"""Compare baseline, Distance PBRS, and Demo PBRS at matched checkpoints."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

import numpy as np

from .gridworld import GAMMA, HORIZON
from .learning import EPSILON_DECAY_FRACTION, EPSILON_END, EPSILON_START, LEARNING_RATE
from .maze_delivery import DEMO_DISTANCE_WEIGHT, SCENES, MazeDeliveryGridworld, Scene
from .moving_obstacles import (
    COLLISION_PENALTY, DELIVERY_BONUS, LOCAL_VIEW_RADIUS, STEP_REWARD,
    LocalObstacleView, QTable, State, evaluate, train,
)


CHECKPOINTS = (40_000, 100_000, 200_000, 400_000, 600_000, *range(800_000, 1_300_001, 50_000))
SEEDS = tuple(range(20))
EVALUATION_EPISODES = 200
EXPLORATION_STEPS = 400_000
LOOP_STEPS = 20
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 240914
EVALUATION_SEED = "maze-eval:{scene}:{seed}:{scenario}"
OUTCOMES = ("delivered", "collision", "timeout", "two_cell_loop_timeout")


@dataclass(frozen=True)
class Method:
    name: str
    label: str
    potential_alpha: float = 0.0
    demo_guided: bool = False


METHODS = (
    Method("baseline", "Baseline"),
    Method("distance-pbrs", "Distance PBRS", potential_alpha=0.1),
    Method("demo-pbrs", "Demo PBRS", potential_alpha=0.1, demo_guided=True),
)


def is_two_cell_loop(trace: list[State]) -> bool:
    tail = [state.robot for state in trace[-(LOOP_STEPS + 1):]]
    return len(tail) == LOOP_STEPS + 1 and tail[0] != tail[1] and tail[:-2] == tail[2:]


def run_seed(
    scene: Scene,
    method: Method,
    seed: int,
    checkpoints: tuple[int, ...],
    eval_episodes: int,
    exploration_steps: int,
) -> list[dict]:
    view = LocalObstacleView(MazeDeliveryGridworld(scene, demo_guided=method.demo_guided))
    checkpoint_set = frozenset(checkpoints)
    rows = []

    def record(step: int, q: QTable) -> None:
        if step not in checkpoint_set:
            return
        counts = Counter()
        for scenario in range(eval_episodes):
            rng = Random(EVALUATION_SEED.format(scene=scene.name, seed=seed, scenario=scenario))
            outcome, trace = evaluate(view, q, rng)
            counts[outcome] += 1
            if outcome == "timeout" and is_two_cell_loop(trace):
                counts["two_cell_loop_timeout"] += 1
        rows.append({
            "method": method.name, "seed": seed, "training_steps": step,
            "evaluation_episodes": eval_episodes, **{name: counts[name] for name in OUTCOMES},
        })

    train(view, max(checkpoints), seed, potential_alpha=method.potential_alpha,
          exploration_steps=exploration_steps, on_step=record)
    print(f"{method.label}, seed {seed}: {rows[-1]['delivered']}/{eval_episodes} final deliveries", flush=True)
    return rows


def run(
    scene: Scene, checkpoints: tuple[int, ...], seeds: tuple[int, ...],
    eval_episodes: int, exploration_steps: int, workers: int,
) -> list[dict]:
    jobs = [(scene, method, seed, checkpoints, eval_episodes, exploration_steps)
            for method in METHODS for seed in seeds]
    if workers == 1:
        return [row for job in jobs for row in run_seed(*job)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(run_seed, *job) for job in jobs]
        return [row for future in pending for row in future.result()]


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], row["training_steps"]].append(row)
    steps = sorted({step for _, step in groups})
    num_seeds = len({row["seed"] for row in rows})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, num_seeds, size=(BOOTSTRAP_REPLICATES, num_seeds))
    summary = []
    for method in METHODS:
        for step in steps:
            group = sorted(groups[method.name, step], key=lambda row: row["seed"])
            n = np.array([row["evaluation_episodes"] for row in group])
            loops = np.array([row["two_cell_loop_timeout"] for row in group])
            low, high = np.quantile(loops[draws].sum(axis=1) / n[draws].sum(axis=1), (0.025, 0.975))
            total = int(n.sum())
            counts = {outcome: sum(row[outcome] for row in group) for outcome in OUTCOMES}
            timeout, collision = counts["timeout"], counts["collision"]
            summary.append({
                "method": method.name, "training_steps": step, "training_seeds": len(group),
                "evaluation_episodes": total, **counts,
                "success_rate": counts["delivered"] / total,
                "timeout_rate": timeout / total,
                "collision_rate": collision / total,
                "failure_rate": (timeout + collision) / total,
                "two_cell_loop_timeout_rate": counts["two_cell_loop_timeout"] / total,
                "seeds_with_two_cell_loop_timeout": int(np.count_nonzero(loops)),
                "seed_bootstrap_95_low": float(low), "seed_bootstrap_95_high": float(high),
            })
    return summary


def settings(
    scene: Scene, checkpoints: tuple[int, ...], seeds: tuple[int, ...],
    eval_episodes: int, exploration_steps: int,
) -> dict:
    return {
        "scene": asdict(scene), "methods": [asdict(method) for method in METHODS],
        "seeds": seeds, "checkpoints": checkpoints, "evaluation_episodes_per_seed": eval_episodes,
        "evaluation_seed_pattern": EVALUATION_SEED,
        "observation": "robot position and visible obstacle offsets", "view_radius": LOCAL_VIEW_RADIUS,
        "gamma": GAMMA, "learning_rate": LEARNING_RATE, "episode_horizon": HORIZON,
        "task_reward": {"step": STEP_REWARD, "delivery_bonus": DELIVERY_BONUS,
                        "collision_penalty": COLLISION_PENALTY},
        "exploration_steps": exploration_steps, "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END, "epsilon_decay_fraction": EPSILON_DECAY_FRACTION,
        "terminal_timeout": True, "demo_distance_weight": DEMO_DISTANCE_WEIGHT,
        "two_cell_loop_steps": LOOP_STEPS, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=[scene.name for scene in SCENES], default="forked-maze")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=CHECKPOINTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--eval-episodes", type=int, default=EVALUATION_EPISODES)
    parser.add_argument("--exploration-steps", type=int, default=EXPLORATION_STEPS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/forked-maze"))
    args = parser.parse_args()
    checkpoints = tuple(sorted(set(args.checkpoints)))
    seeds = tuple(sorted(set(args.seeds)))
    if min(checkpoints) < 1 or min(args.eval_episodes, args.exploration_steps, args.workers) < 1:
        parser.error("Checkpoints, evaluation and exploration budgets, and worker count must be positive.")
    if min(seeds) < 0:
        parser.error("Training seeds must be nonnegative.")

    scene = next(scene for scene in SCENES if scene.name == args.scene)
    rows = run(scene, checkpoints, seeds, args.eval_episodes, args.exploration_steps, args.workers)
    summary = summarize(rows)
    config = settings(scene, checkpoints, seeds, args.eval_episodes, args.exploration_steps)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "seed-checkpoints.csv", rows)
    write_csv(args.out / "summary.csv", summary)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    from .visualization import plot

    plot(summary, config, args.out)
    print(f"Saved {len(rows)} seed/checkpoint rows and figures to {args.out.resolve()}", flush=True)
