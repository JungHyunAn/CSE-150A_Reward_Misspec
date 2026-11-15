"""Run the reward sweep from the repository root."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from .gridworld import GAMMA, HORIZON, LAYOUT, Condition, Gridworld, Reward
from .learning import (
    DEFAULT_EPISODES, EPSILON_DECAY_FRACTION, EPSILON_END, EPSILON_START, LEARNING_RATE,
    evaluate, optimal_q, train,
)
from .visualization import render


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Use a positive, finite number.")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=positive_int, default=DEFAULT_EPISODES)
    parser.add_argument("--eval-every", type=positive_int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--alphas", type=positive_float, nargs="+", default=[0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--out", type=Path, default=Path("runs/default"))
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct nonnegative integers.")
    if len(set(args.alphas)) != len(args.alphas):
        parser.error("--alphas must be distinct.")
    alphas = sorted(args.alphas)

    baseline = Condition(Reward.BASELINE)
    conditions = [baseline] + [
        Condition(reward, alpha)
        for reward in (Reward.POSITIVE, Reward.POTENTIAL)
        for alpha in alphas
    ]
    world = Gridworld()
    runs = {}
    exact = {}
    evaluation_rows = []
    final_rows = []
    reference_rows = []
    for condition in conditions:
        print(f"Training {condition.reward} (alpha={condition.alpha:g})", flush=True)
        transitions = world.transitions(condition)
        exact[condition] = evaluate(world, optimal_q(world, transitions), condition)
        reference_rows.append({
            "reward": condition.reward, "alpha": condition.alpha,
            **exact[condition].metrics(),
        })
        runs[condition] = {}
        for seed in args.seeds:
            checkpoints = dict(train(
                world, condition, transitions, seed, args.episodes, args.eval_every
            ))
            runs[condition][seed] = checkpoints
            evaluation_rows.extend({
                "reward": condition.reward, "alpha": condition.alpha,
                "seed": seed, "episode": episode, **rollout.metrics(),
            } for episode, rollout in checkpoints.items())
            final_rows.append(evaluation_rows[-1])
        final = [checkpoints[args.episodes] for checkpoints in runs[condition].values()]
        print(
            f"  success {mean(rollout.delivered for rollout in final):.0%} | "
            f"mean steps {mean(len(rollout.trajectory) - 1 for rollout in final):.1f}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("learning.csv", evaluation_rows),
        ("final.csv", final_rows),
        ("optimal.csv", reference_rows),
    ):
        with (args.out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    config = {
        "layout": LAYOUT, "horizon": HORIZON, "gamma": GAMMA,
        "learning_rate": LEARNING_RATE, "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END, "epsilon_decay_fraction": EPSILON_DECAY_FRACTION,
        "episodes": args.episodes, "eval_every": args.eval_every,
        "seeds": args.seeds, "alphas": alphas,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    render(args.out, world, runs, exact[baseline].task_return, alphas, args.seeds[0])
    print(f"Saved results to {args.out.resolve()}")


if __name__ == "__main__":
    main()
