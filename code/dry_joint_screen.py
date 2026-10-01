from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

RHO = 1800.0  # kg/m^3; screening assumption, not a lunar measurement
MU = 0.60  # dimensionless; screening assumption, not a lunar measurement
B = 0.50  # m
GRAVITIES = {"Earth": 9.81, "Mars": 3.71, "Moon": 1.62}
ASPECT_RATIOS = [0.5, 1.0, 2.0, 4.0]  # h/b
COHESIONS = [0.0, 500.0, 1000.0]  # Pa; scenario values, not measurements


def pressure_ratios(q, g, b=B, aspect=2.0, rho=RHO, mu=MU, cohesion=0.0):
    h = aspect * b
    sliding = q * h / (mu * rho * b * h * g + cohesion * b)
    overturning = q * h / (rho * b**2 * g)
    return sliding, overturning


def pressure_thresholds(g, b=B, aspect=2.0, rho=RHO, mu=MU, cohesion=0.0):
    h = aspect * b
    q_slide = mu * rho * b * g + cohesion * b / h
    q_overturn = rho * b**2 * g / h
    return q_slide, q_overturn, min(q_slide, q_overturn)


def acceleration_thresholds(g, b=B, aspect=2.0, mu=MU):
    h = aspect * b
    return mu * g, g * b / h


def main():
    rows = []
    for env, g in GRAVITIES.items():
        for aspect in ASPECT_RATIOS:
            for cohesion in COHESIONS:
                q_slide, q_overturn, q_crit = pressure_thresholds(g, aspect=aspect, cohesion=cohesion)
                a_slide, a_overturn = acceleration_thresholds(g, aspect=aspect)
                rows.append(
                    {
                    "environment": env,
                        "g_m_s2": g,
                        "aspect_h_over_b": aspect,
                        "cohesion_Pa": cohesion,
                    "q_slide_Pa": q_slide,
                    "q_overturn_Pa": q_overturn,
                    "q_critical_Pa": q_crit,
                    "pressure_mode": "sliding" if q_slide <= q_overturn else "overturning",
                    "a_slide_m_s2": a_slide,
                    "a_overturn_m_s2": a_overturn,
                    "acceleration_mode": "sliding" if a_slide <= a_overturn else "overturning",
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "dry_joint_thresholds.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    g_values = np.logspace(-2, 1.2, 250)
    for cohesion, style in zip(COHESIONS, ["--", "-", ":"]):
        qcrit = [pressure_thresholds(g, aspect=1.0, cohesion=cohesion)[2] for g in g_values]
        axes[0].plot(g_values, qcrit, linestyle=style, label=f"c = {cohesion:g} Pa")
    for g in GRAVITIES.values():
        axes[0].scatter(g, pressure_thresholds(g, aspect=1.0, cohesion=500.0)[2], s=28, zorder=3)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Gravitational acceleration g (m s$^{-2}$)")
    axes[0].set_ylabel("Critical pressure qcrit (Pa)")
    axes[0].set_title("Cohesion shifts the sliding/overturning crossover (h/b = 1)")
    axes[0].legend(frameon=False)
    axes[0].grid(True, which="both", alpha=0.25)

    g = GRAVITIES["Moon"]
    cohesion = 500.0
    q_grid = np.logspace(-1, 4, 220)
    aspect_grid = np.linspace(0.25, 5.0, 180)
    mode = np.zeros((len(aspect_grid), len(q_grid)))
    for i, aspect in enumerate(aspect_grid):
        for j, q in enumerate(q_grid):
            s, o = pressure_ratios(q, g, aspect=aspect, cohesion=cohesion)
            mode[i, j] = 0 if max(s, o) <= 1 else (1 if s >= o else 2)
    mesh = axes[1].pcolormesh(
        q_grid,
        aspect_grid,
        mode,
        shading="auto",
        cmap=ListedColormap(["#66c2a5", "#fc8d62", "#8da0cb"]),
        vmin=0,
        vmax=2,
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Applied lateral pressure q (Pa)")
    axes[1].set_ylabel("Block aspect ratio h/b")
    axes[1].set_title("Moon: stable / sliding / overturning (c = 500 Pa)")
    cbar = fig.colorbar(mesh, ax=axes[1], ticks=[0.33, 1.0, 1.67], pad=0.02)
    cbar.ax.set_yticklabels(["stable", "sliding", "overturning"])
    axes[1].grid(True, which="both", alpha=0.2)

    fig.suptitle("Dry-joint pressure stability screen (assumptions in README)")
    fig.savefig(OUT / "dry_joint_stability_screen.png", dpi=220)
    fig.savefig(OUT / "dry_joint_stability_screen.svg")
    plt.close(fig)

    # Algebraic sanity checks used as a lightweight regression test.
    assert np.isclose(pressure_thresholds(9.81, aspect=2.0, cohesion=0.0)[2] / pressure_thresholds(1.62, aspect=2.0, cohesion=0.0)[2], 9.81 / 1.62)
    assert pressure_thresholds(0.0, aspect=2.0, cohesion=500.0)[0] > 0
    assert np.isclose(acceleration_thresholds(1.62, aspect=2.0)[0] / 1.62, MU)
    print(table.to_string(index=False))
    print("\nPASS: threshold scaling and mode-boundary checks")


if __name__ == "__main__":
    main()
