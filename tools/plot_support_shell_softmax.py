#!/usr/bin/env python3
"""
Create a conceptual diagram showing how a support (loading) node distributes
load to shell vertices using softmax weights derived from local scores.

Example:
    python tools/plot_support_shell_softmax.py \
        --output docs/support_node_softmax.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib


def _draw_diagram(ax) -> None:
    from matplotlib.patches import Circle, FancyArrowPatch
    import numpy as np

    ax.set_aspect("equal", "box")
    ax.set_xlim(-3.4, 3.6)
    ax.set_ylim(-2.1, 2.1)
    ax.axis("off")

    support_angle = np.deg2rad(60.0)  # reference angle for the truss support node
    shell_radius = 2.3
    shell_angles = np.deg2rad([20.0, 40.0, 55.0, 75.0, 100.0, 125.0])
    shell_nodes = np.column_stack(
        (shell_radius * np.cos(shell_angles), shell_radius * np.sin(shell_angles))
    )
    tau = np.deg2rad(35.0)
    delta_thetas = shell_angles - support_angle
    weights_raw = np.exp(-0.5 * (delta_thetas / tau) ** 2)
    shell_weights = weights_raw / np.sum(weights_raw)
    ax.plot(
        shell_nodes[:, 0],
        shell_nodes[:, 1],
        linestyle="--",
        linewidth=1.6,
        color="#1f78b4",
        alpha=0.6,
    )
    ax.text(
        1.35,
        2.15,
        "Shell vertices",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#12507a",
    )

    support_radius = 1.1
    support_pos = (
        support_radius * np.cos(support_angle),
        support_radius * np.sin(support_angle),
    )
    support_node = Circle(support_pos, radius=0.12, color="#d62728", zorder=4)
    ax.add_patch(support_node)
    ax.text(
        support_pos[0],
        support_pos[1] - 0.28,
        "Support node\n(loading point)",
        ha="center",
        va="top",
        fontsize=11,
        color="#8c1c13",
    )
    ax.text(
        support_pos[0] + 0.18,
        support_pos[1] + 0.18,
        r"$\theta_s$",
        ha="left",
        va="bottom",
        fontsize=11,
        color="#8c1c13",
    )
    ax.plot(
        [0.0, shell_radius * 1.05 * np.cos(support_angle)],
        [0.0, shell_radius * 1.05 * np.sin(support_angle)],
        linestyle=":",
        linewidth=1.4,
        color="#8c1c13",
        alpha=0.7,
    )
    ax.add_patch(Circle((0.0, 0.0), radius=0.05, color="#8c1c13", alpha=0.6, zorder=2))

    ax.text(
        -1.65,
        0.0,
        r"$w_i = \dfrac{\exp[-(\Delta\theta_i/\tau)^2/2]}{\sum_j \exp[-(\Delta\theta_j/\tau)^2/2]}$",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#bbbbbb",
            alpha=0.9,
        ),
    )

    ax.text(
        -0.2,
        0.62,
        r"$\mathbf{C}(\boldsymbol{\theta})$ row for this support",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )

    support_vec = np.array(support_pos)
    for idx, ((x, y), weight, angle, dtheta) in enumerate(
        zip(shell_nodes, shell_weights, shell_angles, delta_thetas), start=1
    ):
        ax.add_patch(
            Circle((x, y), radius=0.09, color="#1f78b4", alpha=0.95, zorder=5)
        )
        thickness = 1.4 + 6.8 * weight
        direction = np.array([x, y]) - support_vec
        start = support_vec + 0.22 * direction
        end = support_vec + 0.95 * direction
        arrow = FancyArrowPatch(
            posA=start,
            posB=end,
            arrowstyle="-|>",
            mutation_scale=14.0 + weight * 35.0,
            linewidth=thickness,
            color="#1f78b4",
            alpha=0.8,
            zorder=3,
        )
        ax.add_patch(arrow)
        ax.text(
            x,
            y + 0.26,
            rf"$\theta_{{{idx}}}$",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#0a344d",
        )
        ax.text(
            x,
            y - 0.05,
            rf"$\Delta\theta_{{{idx}}} = {np.degrees(abs(dtheta)):.0f}^\circ$",
            ha="center",
            va="top",
            fontsize=10,
            color="#0a344d",
        )
        ax.text(
            x,
            y - 0.32,
            rf"$w_{{{idx}}} = {weight:.2f}$",
            ha="center",
            va="top",
            fontsize=11,
            color="#0a344d",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a conceptual diagram linking local support scores,"
        " softmax weights, and shell vertices."
    )
    parser.add_argument(
        "--output",
        default="docs/support_node_softmax.png",
        help="Path to save the resulting figure (default: docs/support_node_softmax.png)",
    )
    args = parser.parse_args()

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    _draw_diagram(ax)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
