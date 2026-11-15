"""Deterministic delivery task and its training rewards."""

from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple


LAYOUT = (
    "S.....",
    ".###..",
    "...#..",
    ".#....",
    ".#..#.",
    ".....G",
)
ACTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # East, south, west, north.
GAMMA = 0.99
HORIZON = 100

TransitionTable = list[list[list[tuple[int, float, bool]]]]


class State(NamedTuple):
    position: int
    remaining: int


class Reward(StrEnum):
    BASELINE = "baseline"
    POSITIVE = "positive"
    POTENTIAL = "potential"


@dataclass(frozen=True)
class Condition:
    reward: Reward
    alpha: float = 0.0


class Gridworld:
    def __init__(self) -> None:
        self.positions = tuple(
            (row, col)
            for row, line in enumerate(LAYOUT)
            for col, cell in enumerate(line)
            if cell != "#"
        )
        index = {cell: i for i, cell in enumerate(self.positions)}
        self.start = next(i for (r, c), i in index.items() if LAYOUT[r][c] == "S")
        self.goal = next(i for (r, c), i in index.items() if LAYOUT[r][c] == "G")
        goal_row, goal_col = self.positions[self.goal]
        self.distance = tuple(
            abs(row - goal_row) + abs(col - goal_col) for row, col in self.positions
        )
        self.moves = tuple(
            tuple(index.get((r + dr, c + dc), i) for dr, dc in ACTIONS)
            for i, (r, c) in enumerate(self.positions)
        )

    def reset(self) -> State:
        return State(self.start, HORIZON)

    def step(self, state: State, action: int) -> State:
        if self.done(state):
            raise ValueError("Reset before stepping from a terminal state.")
        if not 0 <= action < len(ACTIONS):
            raise ValueError(f"Invalid action index: {action}")
        return State(self.moves[state.position][action], state.remaining - 1)

    def done(self, state: State) -> bool:
        return state.position == self.goal or state.remaining == 0

    def task_reward(self, next_state: State) -> float:
        return -0.1 + 10.0 * (next_state.position == self.goal)

    def training_reward(
        self, state: State, next_state: State, condition: Condition
    ) -> float:
        task_reward = self.task_reward(next_state)
        if condition.reward is Reward.BASELINE:
            return task_reward

        distance = self.distance[state.position]
        next_distance = self.distance[next_state.position]
        if condition.reward is Reward.POSITIVE:
            return task_reward + condition.alpha * max(0, distance - next_distance)

        potential = -condition.alpha * distance
        next_potential = 0.0 if self.done(next_state) else -condition.alpha * next_distance
        return task_reward + GAMMA * next_potential - potential

    def transitions(self, condition: Condition) -> TransitionTable:
        """Index transitions by remaining time, position, and action."""
        table: TransitionTable = [[]]
        for remaining in range(1, HORIZON + 1):
            layer = []
            for position in range(len(self.positions)):
                state = State(position, remaining)
                actions = []
                if not self.done(state):
                    for action in range(len(ACTIONS)):
                        next_state = self.step(state, action)
                        actions.append((
                            next_state.position,
                            self.training_reward(state, next_state, condition),
                            self.done(next_state),
                        ))
                layer.append(actions)
            table.append(layer)
        return table
