"""Q-learning, greedy policy evaluation, and a finite-horizon reference."""

from collections.abc import Iterator
from dataclasses import dataclass
from random import Random

from .gridworld import ACTIONS, GAMMA, HORIZON, Condition, Gridworld, TransitionTable


DEFAULT_EPISODES = 5_000
LEARNING_RATE = 0.2
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_FRACTION = 0.8

QTable = list[list[list[float]]]


def make_q_table(world: Gridworld) -> QTable:
    return [
        [[0.0] * len(ACTIONS) for _ in world.positions]
        for _ in range(HORIZON + 1)
    ]


def greedy_action(values: list[float]) -> int:
    # Fixed action order keeps greedy evaluation deterministic.
    return values.index(max(values))


@dataclass(frozen=True)
class Rollout:
    trajectory: tuple[int, ...]
    delivered: bool
    task_return: float
    training_return: float
    revisits: int
    blocked_moves: int

    def metrics(self) -> dict[str, float | int]:
        return {
            "task_return": self.task_return,
            "training_return": self.training_return,
            "success": int(self.delivered),
            "timeout": int(not self.delivered),
            "steps": len(self.trajectory) - 1,
            "revisits": self.revisits,
            "blocked_moves": self.blocked_moves,
        }


def evaluate(world: Gridworld, q: QTable, condition: Condition) -> Rollout:
    state = world.reset()
    trajectory = [state.position]
    visited = {state.position}
    task_return = training_return = 0.0
    revisits = blocked_moves = 0
    discount = 1.0

    while not world.done(state):
        action = greedy_action(q[state.remaining][state.position])
        next_state = world.step(state, action)
        task_return += discount * world.task_reward(next_state)
        training_return += discount * world.training_reward(state, next_state, condition)

        position = next_state.position
        moved = position != state.position
        blocked_moves += not moved
        revisits += moved and position in visited
        visited.add(position)
        trajectory.append(position)
        state = next_state
        discount *= GAMMA

    return Rollout(
        tuple(trajectory), state.position == world.goal, task_return,
        training_return, revisits, blocked_moves,
    )


def train(
    world: Gridworld,
    condition: Condition,
    transitions: TransitionTable,
    seed: int,
    episodes: int,
    eval_every: int,
) -> Iterator[tuple[int, Rollout]]:
    rng = Random(seed)
    q = make_q_table(world)
    yield 0, evaluate(world, q, condition)

    # Exploration decays over the first 80% of the episode budget.
    decay_episodes = max(1, int(EPSILON_DECAY_FRACTION * episodes) - 1)
    for episode in range(1, episodes + 1):
        fraction = min((episode - 1) / decay_episodes, 1.0)
        epsilon = EPSILON_START + fraction * (EPSILON_END - EPSILON_START)
        position = world.start
        for remaining in range(HORIZON, 0, -1):
            values = q[remaining][position]
            if rng.random() < epsilon:
                action = rng.randrange(len(ACTIONS))
            else:
                best = max(values)
                action = rng.choice([i for i, value in enumerate(values) if value == best])

            next_position, reward, done = transitions[remaining][position][action]
            target = reward if done else reward + GAMMA * max(q[remaining - 1][next_position])
            values[action] += LEARNING_RATE * (target - values[action])
            if done:
                break
            position = next_position

        if episode % eval_every == 0 or episode == episodes:
            yield episode, evaluate(world, q, condition)


def optimal_q(world: Gridworld, transitions: TransitionTable) -> QTable:
    """Solve backward from zero remaining actions using the same transition model."""
    q = make_q_table(world)
    for remaining in range(1, HORIZON + 1):
        for position, actions in enumerate(transitions[remaining]):
            for action, (next_position, reward, done) in enumerate(actions):
                q[remaining][position][action] = (
                    reward if done else reward + GAMMA * max(q[remaining - 1][next_position])
                )
    return q
