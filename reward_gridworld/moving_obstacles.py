"""Simulate moving obstacles and train tabular Q-learning on a local view."""

from collections import Counter, defaultdict
from collections.abc import Callable
from random import Random
from typing import Literal, NamedTuple

import numpy as np

from .gridworld import ACTIONS, GAMMA, HORIZON, LAYOUT, Gridworld
from .learning import LEARNING_RATE, exploration_rate, exploratory_action, greedy_action


OBSTACLE_TRACKS = (
    ((0, 3), (0, 4), (0, 5)),
    ((2, 4), (3, 4), (3, 5)),
    ((5, 1), (5, 2), (5, 3)),
)
OBSTACLE_STAY_PROBABILITY = 0.5
STEP_REWARD = -0.1
DELIVERY_BONUS = 10.0
COLLISION_PENALTY = 10.0
LOCAL_VIEW_RADIUS = 2

Outcome = Literal["delivered", "collision", "timeout"]


class State(NamedTuple):
    robot: int
    obstacles: tuple[int, ...]


class LocalState(NamedTuple):
    robot: int
    relative_obstacles: tuple[tuple[int, int], ...]


class MovingObstacleGridworld:
    def __init__(
        self, layout: tuple[str, ...] = LAYOUT,
        obstacle_tracks: tuple[tuple[tuple[int, int], ...], ...] | None = None,
        *, obstacle_regions: tuple[tuple[tuple[int, int], ...], ...] | None = None,
    ) -> None:
        self.layout = layout
        self.grid = Gridworld(layout)
        index = {cell: i for i, cell in enumerate(self.grid.positions)}
        self._shared_obstacle_region = False
        if obstacle_regions is None:
            tracks = OBSTACLE_TRACKS if obstacle_tracks is None else obstacle_tracks
            self.tracks = tuple(tuple(index[cell] for cell in track) for track in tracks)
            self.obstacle_neighbors = {
                cell: track[max(i - 1, 0):i] + track[i + 1:i + 2]
                for track in self.tracks
                for i, cell in enumerate(track)
            }
        else:
            if obstacle_tracks is not None:
                raise ValueError("Specify obstacle tracks or regions.")
            self.tracks = tuple(tuple(index[cell] for cell in region) for region in obstacle_regions)
            region_counts = Counter(self.tracks)
            self._shared_obstacle_region = len(region_counts) < len(self.tracks)
            if any(count > len(region) for region, count in region_counts.items()):
                raise ValueError("Obstacle regions must have a cell for each starting obstacle.")
            self.obstacle_neighbors = {
                cell: tuple(neighbor for neighbor in self.grid.moves[cell] if neighbor != cell and neighbor in region)
                for region in region_counts
                for cell in region
            }
            if len(self.obstacle_neighbors) != sum(map(len, region_counts)):
                raise ValueError("Obstacle regions can be identical or disjoint.")

    def reset(self, rng: Random) -> State:
        if not self._shared_obstacle_region:
            return State(self.grid.start, tuple(rng.choice(track) for track in self.tracks))
        starts = []
        for region in self.tracks:
            starts.append(rng.choice([cell for cell in region if cell not in starts]))
        return State(self.grid.start, tuple(starts))

    def sample_obstacle_move(self, obstacles: tuple[int, ...], rng: Random) -> tuple[int, ...]:
        proposals = tuple(
            obstacle if rng.random() < OBSTACLE_STAY_PROBABILITY
            else rng.choice(self.obstacle_neighbors[obstacle])
            for obstacle in obstacles
        )
        if not self._shared_obstacle_region:
            return proposals

        # Use the occupancy at the start of the step for all obstacles.
        occupied = set(obstacles)
        destinations = Counter(proposals)
        return tuple(
            after if after not in occupied and destinations[after] == 1 else before
            for before, after in zip(obstacles, proposals, strict=True)
        )

    def step(
        self, state: State, action: int, rng: Random, *, next_obstacles: tuple[int, ...] | None = None,
    ) -> tuple[State, float, Outcome | None]:
        if not 0 <= action < len(ACTIONS):
            raise ValueError(f"Invalid action index: {action}")
        robot = self.grid.moves[state.robot][action]
        if robot == state.robot:
            return state, STEP_REWARD - COLLISION_PENALTY, "collision"

        obstacles = self.sample_obstacle_move(state.obstacles, rng) if next_obstacles is None else next_obstacles
        next_state = State(robot, obstacles)
        swapped = any(
            robot == before and state.robot == after
            for before, after in zip(state.obstacles, obstacles, strict=True)
        )
        if robot in obstacles or swapped:
            return next_state, STEP_REWARD - COLLISION_PENALTY, "collision"
        if robot == self.grid.goal:
            return next_state, STEP_REWARD + DELIVERY_BONUS, "delivered"
        return next_state, STEP_REWARD, None

    def remaining_distance(self, state: State) -> int:
        return self.grid.distance[state.robot]

    def observe(self, state: State) -> State:
        return state

    def one_hot(self, state: State) -> np.ndarray:
        """Return one grid channel each for the robot and the three obstacles."""
        positions = (state.robot, *state.obstacles)
        observation = np.zeros((len(positions), len(self.layout), len(self.layout[0])), dtype=np.uint8)
        for channel, position in enumerate(positions):
            row, col = self.grid.positions[position]
            observation[channel, row, col] = 1
        return observation

    def render(self, state: State) -> str:
        board = [list(row) for row in self.layout]
        for symbol, channel in zip("R123", self.one_hot(state), strict=True):
            row, col = np.argwhere(channel)[0]
            board[row][col] = "X" if board[row][col] == "R" else symbol
        return "\n".join("".join(row) for row in board)


class LocalObstacleView:
    """Project the full simulator state to a robot-centered 5×5 view."""

    def __init__(self, world: MovingObstacleGridworld) -> None:
        self.world = world

    def reset(self, rng: Random) -> State:
        return self.world.reset(rng)

    def step(self, state: State, action: int, rng: Random) -> tuple[State, float, Outcome | None]:
        return self.world.step(state, action, rng)

    def remaining_distance(self, state: State) -> int:
        return self.world.remaining_distance(state)

    def observe(self, state: State) -> LocalState:
        positions = self.world.grid.positions
        row, col = positions[state.robot]
        visible = []
        for obstacle in state.obstacles:
            obstacle_row, obstacle_col = positions[obstacle]
            dr, dc = obstacle_row - row, obstacle_col - col
            if abs(dr) <= LOCAL_VIEW_RADIUS and abs(dc) <= LOCAL_VIEW_RADIUS:
                visible.append((dr, dc))
        return LocalState(state.robot, tuple(sorted(visible)))


QTable = dict[LocalState, list[float]]


def train(
    world: LocalObstacleView,
    training_steps: int,
    seed: int,
    *,
    potential_alpha: float = 0.0,
    exploration_steps: int = 400_000,
    on_step: Callable[[int, QTable], None] | None = None,
) -> QTable:
    """Train for an exact action budget with terminal timeouts."""
    if training_steps < 1 or exploration_steps < 1:
        raise ValueError("Training and exploration budgets must be positive.")
    actions_rng = Random(seed)
    obstacles_rng = Random(f"training:{seed}")
    q: QTable = defaultdict(lambda: [0.0] * len(ACTIONS))
    steps = 0
    while steps < training_steps:
        state = world.reset(obstacles_rng)
        observation = world.observe(state)
        for episode_step in range(1, HORIZON + 1):
            steps += 1
            epsilon = exploration_rate(steps, exploration_steps)
            values = q[observation]
            action = exploratory_action(values, epsilon, actions_rng)
            next_state, reward, outcome = world.step(state, action, obstacles_rng)
            if outcome is None and episode_step == HORIZON:
                outcome = "timeout"
            if potential_alpha:
                current_potential = -potential_alpha * world.remaining_distance(state)
                next_potential = 0.0
                if outcome is None:
                    next_potential = -potential_alpha * world.remaining_distance(next_state)
                reward += GAMMA * next_potential - current_potential
            target = reward
            if outcome is None:
                observation = world.observe(next_state)
                target += GAMMA * max(q[observation])
            values[action] += LEARNING_RATE * (target - values[action])
            if on_step is not None:
                on_step(steps, q)
            if steps == training_steps:
                return q
            if outcome:
                break
            state = next_state
    return q


def evaluate(world: LocalObstacleView, q: QTable, rng: Random) -> tuple[Outcome, list[State]]:
    """Evaluate the frozen table greedily for one obstacle scenario."""
    state = world.reset(rng)
    trace = [state]
    zero_values = [0.0] * len(ACTIONS)
    for _ in range(HORIZON):
        action = greedy_action(q.get(world.observe(state), zero_values))
        state, _, outcome = world.step(state, action, rng)
        trace.append(state)
        if outcome:
            return outcome, trace
    return "timeout", trace
