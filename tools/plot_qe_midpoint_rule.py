import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def create_qe_midpoint_figure(out_path: str = "docs/paper/figures/qe_midpoint_rule.pdf") -> None:
    """Render an illustrative figure of the midpoint-rule evaluation of q_e."""
    # Representative line-load variation along a shell boundary edge
    ell_e = 1.25  # edge length in metres
    s_vals = np.linspace(0.0, ell_e, 200)

    # Assume linear variation because hydrostatic pressure depends on depth
    q0 = 34.0  # kN/m at the lower end of the edge
    q1 = 31.0  # kN/m at the upper end of the edge
    q_vals = q0 + (q1 - q0) * (s_vals / ell_e)

    # Exact integral (area under the curve, already hinted in shading)
    q_exact = np.trapezoid(q_vals, s_vals)

    # Midpoint rule approximation
    s_mid = 0.5 * ell_e
    q_mid = q0 + (q1 - q0) * 0.5
    q_midpoint = q_mid * ell_e

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    # Plot q(s)
    ax.plot(s_vals, q_vals, color="#1f77b4", linewidth=2.8, label=r"$q(s) = p(y(s))\,t$")

    # Shade the true integral area
    ax.fill_between(s_vals, 0.0, q_vals, color="#8ecae6", alpha=0.45,
                    label=r"Exact line load $q_e=\int_0^{\ell_e} q(s)\,\mathrm{d}s$")

    # Midpoint rectangle (dashed outline)
    rect = Rectangle((0.0, 0.0), ell_e, q_mid, fill=False,
                     edgecolor="black", linewidth=2.0, linestyle="--")
    ax.add_patch(rect)

    # Annotation for midpoint rectangle
    ax.annotate(r"Midpoint sample $q(s_m)$",
                xy=(s_mid, q_mid),
                xytext=(ell_e * 0.78, q_mid * 1.20),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=1.1),
                fontsize=10)

    # Show midpoint sample location
    ax.scatter([s_mid], [q_mid], color="black", zorder=5)
    ax.text(s_mid, q_mid + 0.8, r"$s_m$", ha="center", va="bottom", fontsize=11)

    # Mark edge length with a double arrow along the baseline
    ax.annotate("", xy=(0.0, -3.0), xytext=(ell_e, -3.0),
                arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.1))
    ax.text(0.5 * ell_e, -3.4, r"$\ell_e$", ha="center", va="top", fontsize=11)

    # Put summary equations outside plotting area to avoid clutter
    fig.text(0.08, 0.92, r"$q_e = \int_{0}^{\ell_e} q(s)\,\mathrm{d}s$", fontsize=12)
    fig.text(0.08, 0.86,
             fr"Midpoint rule: $q(s_m)\,\ell_e = {q_mid:.1f}\times{ell_e:.2f} = {q_midpoint:.1f}\,\mathrm{{kN}}$",
             fontsize=11)

    ax.set_xlabel(r"Arc coordinate $s$ along edge [m]", fontsize=11)
    ax.set_ylabel(r"Line load density $q(s)$ [kN/m]", fontsize=11)
    ax.set_title("Midpoint Approximation of Boundary Edge Line Load $q_e$", fontsize=12, pad=14)

    # Legend with explicit handles
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Rectangle((0, 0), 0, 0, fill=False, edgecolor="black",
                             linestyle="--", linewidth=1.8))
    labels.append(r"Approximation area $q(s_m)\,\ell_e$")
    ax.legend(handles, labels, loc="upper right", frameon=False)

    ax.set_xlim(-0.05, ell_e + 0.05)
    ax.set_ylim(-5.0, max(q_vals) * 1.25)
    ax.grid(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    create_qe_midpoint_figure()
