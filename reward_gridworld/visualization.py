"""Render the failure decomposition and two-cell-loop diagnostic."""

from pathlib import Path

import matplotlib
import numpy as np

from .experiment import METHODS

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


METHOD_COLORS = ("#345075", "#bc572c", "#7147a1")
METHOD_MARKERS = ("o", "s", "^")


def plot(rows: list[dict], config: dict, out: Path) -> None:
    steps = sorted({row["training_steps"] for row in rows})
    groups = [
        sorted((row for row in rows if row["method"] == method.name), key=lambda row: row["training_steps"])
        for method in METHODS
    ]
    labels = [f"{step / 1000:,.3f}".rstrip("0").rstrip(".") + "k" for step in steps]
    scenario_count = rows[0]["evaluation_episodes"]
    subtitle = (f"{config['scene']['title']} · {rows[0]['training_seeds']} seeds · "
                f"{scenario_count:,} greedy evaluations per method and checkpoint")
    x = np.arange(len(steps))

    fig, ax = plt.subplots(figsize=(12.4, 5.2), facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.982, bottom=0.155, top=0.739)
    fig.text(0.085, 0.940, "Failure rates by training checkpoint", fontsize=22, weight="bold", color="#17263b")
    fig.text(0.085, 0.873, subtitle, fontsize=11, color="#465a72")
    for index, (offset, group) in enumerate(zip((-0.251, 0, 0.251), groups, strict=True)):
        timeout = np.array([row["timeout_rate"] for row in group])
        collision = [row["collision_rate"] for row in group]
        ax.bar(x + offset, timeout, width=0.245, color="#376aa4", edgecolor="white",
               linewidth=0.5, label="Timeout" if index == 0 else "_nolegend_")
        ax.bar(x + offset, collision, bottom=timeout, width=0.245, color="#ba5834", edgecolor="white",
               linewidth=0.5, label="Collision" if index == 0 else "_nolegend_")
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.set_ylim(0, 1.04)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    style_axes(ax)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.085, 0.835),
               frameon=False, ncol=2, fontsize=11)
    fig.text(0.085, 0.035, "Each group shows Baseline, Distance PBRS, and Demo PBRS from left to right.",
             color="#465a72", fontsize=10)
    fig.savefig(out / "failure-rates.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.4, 4.7), facecolor="white")
    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.19, top=0.725)
    fig.text(0.095, 0.937, "Two-cell-loop timeout rates", fontsize=21, weight="bold", color="#17263b")
    fig.text(0.095, 0.866, subtitle, fontsize=10, color="#465a72")
    for method, group, color, marker in zip(METHODS, groups, METHOD_COLORS, METHOD_MARKERS, strict=True):
        y = [row["two_cell_loop_timeout_rate"] for row in group]
        low = [row["seed_bootstrap_95_low"] for row in group]
        high = [row["seed_bootstrap_95_high"] for row in group]
        ax.fill_between(steps, low, high, color=color, alpha=0.12, linewidth=0)
        ax.plot(steps, y, label=method.label, color=color, marker=marker, markersize=4.5, linewidth=1.8)
    tick_stride = max(1, (len(steps) + 7) // 8)
    tick_indices = sorted({*range(0, len(steps), tick_stride), len(steps) - 1})
    ax.set_xticks([steps[i] for i in tick_indices], [labels[i] for i in tick_indices])
    ax.set_ylim(0, max(0.02, max(row["seed_bootstrap_95_high"] for row in rows) * 1.12))
    style_axes(ax)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.095, 0.827),
               frameon=False, ncol=3, fontsize=10)
    fig.text(0.095, 0.04, "Shading shows pointwise 95% bootstrap intervals over training seeds.",
             color="#465a72", fontsize=9)
    fig.savefig(out / "two-cell-loop-timeouts.png", dpi=180)
    plt.close(fig)


def style_axes(ax) -> None:
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Share of all evaluation episodes", color="#465a72", fontsize=10)
    ax.set_xlabel("Training checkpoint", color="#465a72", fontsize=10, labelpad=8)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#bcc5d2")
    ax.tick_params(axis="both", colors="#465a72", length=0, labelsize=9, pad=5)
