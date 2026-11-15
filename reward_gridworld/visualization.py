"""Render headless, reproducible plots from the greedy evaluations."""

from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

from .gridworld import LAYOUT, Condition, Gridworld, Reward
from .learning import Rollout

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter


Runs = dict[Condition, dict[int, dict[int, Rollout]]]
LABELS = {
    Reward.BASELINE: "Baseline",
    Reward.POSITIVE: "Positive-only",
    Reward.POTENTIAL: "Potential-based",
}
COLORS = {
    Reward.BASELINE: "#45566d",
    Reward.POSITIVE: "#b34324",
    Reward.POTENTIAL: "#087b75",
}
METRICS = (
    ("task_return", "Discounted task return"),
    ("success", "Delivery success rate"),
    ("steps", "Actions per episode"),
    ("revisits", "Repeated entries"),
)


def metric_axes(heading: str, description: str, x_label: str):
    fig, panels = plt.subplots(2, 2, figsize=(10, 6.5))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.79, bottom=0.10, hspace=0.48, wspace=0.31)
    fig.text(0.10, 0.954, heading, fontsize=19, weight="bold", color="#17263b")
    fig.text(0.10, 0.909, description, fontsize=10, color="#4c5e73")
    for ax, (key, label) in zip(panels.flat, METRICS, strict=True):
        ax.set_title(label, loc="left", weight="semibold", fontsize=11)
        ax.set_xlabel(x_label, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e0e6ee")
        ax.set_axisbelow(True)
        if key == "success":
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_ylim(-0.05, 1.05)
    return fig, panels.flat


def abbreviate(condition: Condition) -> str:
    label = LABELS[condition.reward]
    return label if condition.reward is Reward.BASELINE else f"{label} (α={condition.alpha:g})"


def save_learning(out: Path, runs: Runs, reference: float, alpha: float) -> None:
    baseline = Condition(Reward.BASELINE)
    conditions = [baseline, Condition(Reward.POSITIVE, alpha), Condition(Reward.POTENTIAL, alpha)]
    sample = next(iter(runs[baseline].values()))
    episodes = list(sample)
    seed_count = len(runs[baseline])
    fig, axes = metric_axes(
        "Learning curves", f"Greedy evaluation · {seed_count} seeds · Mean ± 1 SD",
        "Training episodes",
    )
    for ax, (key, _) in zip(axes, METRICS, strict=True):
        for condition in conditions:
            values = np.array([
                [rollout.metrics()[key] for rollout in checkpoints.values()]
                for checkpoints in runs[condition].values()
            ])
            center = values.mean(axis=0)
            spread = values.std(axis=0)
            color = COLORS[condition.reward]
            ax.plot(episodes, center, color=color, label=abbreviate(condition), lw=1.6)
            low, high = center - spread, center + spread
            if key == "success":
                low, high = low.clip(0, 1), high.clip(0, 1)
            elif key in {"steps", "revisits"}:
                low = low.clip(min=0)
            ax.fill_between(episodes, low, high, color=color, alpha=0.13, linewidth=0)
        if key == "task_return":
            ax.axhline(reference, color="#182535", lw=1, ls=":", label="Task optimum")
        if key == "success" and all(
            np.array_equal(line.get_ydata(), ax.lines[0].get_ydata()) for line in ax.lines[1:]
        ):
            ax.text(0.5, 0.5, "All curves overlap", transform=ax.transAxes,
                    ha="center", color="#4c5e73", fontsize=10)
        ax.set_xlim(0, episodes[-1])
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.09, 0.895),
               ncol=4, frameon=False, fontsize=9)
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)


def save_sweep(out: Path, runs: Runs, alphas: list[float]) -> None:
    baseline = Condition(Reward.BASELINE)
    seed_count = len(runs[baseline])
    episodes = max(next(iter(runs[baseline].values())))
    fig, axes = metric_axes(
        "Final performance by bonus strength",
        f"After {episodes:,} episodes · {seed_count} seeds · Mean ± 1 SD for shaped rewards",
        "Bonus strength α",
    )
    for ax, (key, _) in zip(axes, METRICS, strict=True):
        means = []
        for reward in (Reward.POSITIVE, Reward.POTENTIAL):
            values = np.array([
                [run[episodes].metrics()[key] for run in runs[Condition(reward, alpha)].values()]
                for alpha in alphas
            ])
            center = values.mean(axis=1)
            means.append(center)
            spread = values.std(axis=1)
            low, high = spread, spread
            if key == "success":
                low, high = np.minimum(spread, center), np.minimum(spread, 1 - center)
            elif key in {"steps", "revisits"}:
                low = np.minimum(spread, center)
            ax.errorbar(alphas, center, yerr=[low, high], marker="o", ms=4, capsize=3,
                        color=COLORS[reward], label=LABELS[reward], lw=1.5)
        baseline_mean = np.mean([run[episodes].metrics()[key] for run in runs[baseline].values()])
        ax.axhline(baseline_mean, color=COLORS[Reward.BASELINE], lw=1.4, ls="--",
                   label="Baseline mean")
        if key == "success" and np.all(np.asarray(means) == baseline_mean):
            ax.text(0.5, 0.5, "All conditions overlap", transform=ax.transAxes,
                    ha="center", color="#4c5e73", fontsize=10)
        ax.set_xticks(alphas, [f"{alpha:g}" for alpha in alphas])
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.09, 0.895),
               ncol=3, frameon=False, fontsize=9)
    fig.savefig(out / "alpha_sweep.png", dpi=160)
    plt.close(fig)


def save_trajectories(out: Path, world: Gridworld, runs: Runs, alpha: float, seed: int) -> None:
    conditions = [
        Condition(Reward.BASELINE), Condition(Reward.POSITIVE, alpha),
        Condition(Reward.POTENTIAL, alpha),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6))
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.10, top=0.73, wspace=0.23)
    fig.text(0.06, 0.94, "Greedy delivery paths", fontsize=19, weight="bold", color="#17263b")
    fig.text(
        0.06, 0.88,
        f"Seed {seed} · Arrows show movement · Numbers show visits",
        fontsize=10, color="#4c5e73",
    )
    for ax, condition in zip(axes, conditions, strict=True):
        checkpoints = runs[condition][seed]
        rollout = checkpoints[max(checkpoints)]
        color = COLORS[condition.reward]
        moved_to = [b for a, b in zip(rollout.trajectory, rollout.trajectory[1:]) if a != b]
        visits = Counter([world.start, *moved_to])
        edges = Counter(
            (a, b) for a, b in zip(rollout.trajectory, rollout.trajectory[1:]) if a != b
        )
        for row, line in enumerate(LAYOUT):
            for col, cell in enumerate(line):
                fill = {"#": "#29384c", "S": "#deebfa", "G": "#daede0"}.get(cell, "#f4f7fa")
                ax.add_patch(Rectangle((col, row), 1, 1, facecolor=fill, edgecolor="white"))
        for (a, b), count in edges.items():
            row_a, col_a = world.positions[a]
            row_b, col_b = world.positions[b]
            dx, dy = col_b - col_a, row_b - row_a
            offset = 0.11 if (b, a) in edges else 0
            x, y = col_a + 0.5 - dy * offset, row_a + 0.5 + dx * offset
            ax.annotate("", xy=(x + 0.82 * dx, y + 0.82 * dy), xytext=(x + 0.18 * dx, y + 0.18 * dy),
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.1 + min(count, 24) / 8})
        for position, (row, col) in enumerate(world.positions):
            cell = LAYOUT[row][col]
            if cell in "SG":
                ax.text(col + 0.12, row + 0.31, cell, color="#182535", weight="bold", fontsize=9)
            if visits[position] > 1:
                ax.text(col + 0.83, row + 0.32, str(visits[position]), color=color,
                        ha="right", fontsize=8, weight="bold")
        ax.set_title(abbreviate(condition), weight="semibold", color=color, pad=10)
        outcome = "Delivered" if rollout.delivered else "Timed out"
        ax.set_xlabel(f"{outcome} in {len(rollout.trajectory) - 1} actions\n"
                      f"{rollout.revisits} repeated entries · {rollout.blocked_moves} blocked moves", fontsize=9)
        ax.set(xlim=(0, len(LAYOUT[0])), ylim=(len(LAYOUT), 0), aspect="equal", xticks=[], yticks=[])
        ax.spines[:].set_visible(False)
    fig.savefig(out / "trajectories.png", dpi=160)
    plt.close(fig)


def render(
    out: Path, world: Gridworld, runs: Runs, reference: float,
    alphas: list[float], seed: int,
) -> None:
    save_learning(out, runs, reference, max(alphas))
    save_sweep(out, runs, alphas)
    save_trajectories(out, world, runs, max(alphas), seed)
