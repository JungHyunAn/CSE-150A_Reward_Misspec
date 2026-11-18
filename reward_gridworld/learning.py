"""Action selection and exploration for tabular Q-learning."""

from random import Random

from .gridworld import ACTIONS


LEARNING_RATE = 0.2
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_FRACTION = 0.8


def greedy_action(values: list[float]) -> int:
    # Fixed action order keeps greedy evaluation deterministic.
    return values.index(max(values))


def exploration_rate(episode: int, episodes: int) -> float:
    # Exploration decays over the first 80% of the episode budget.
    decay_episodes = max(1, int(EPSILON_DECAY_FRACTION * episodes) - 1)
    fraction = min((episode - 1) / decay_episodes, 1.0)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def exploratory_action(values: list[float], epsilon: float, rng: Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(values))
    best = max(values)
    return rng.choice([i for i, value in enumerate(values) if value == best])
