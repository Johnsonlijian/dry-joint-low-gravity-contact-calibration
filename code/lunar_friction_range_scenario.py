from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lunar_friction_angle_scenarios.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

H = 0.100
WIDTHS = {"concrete_parallel": 0.125, "concrete_perpendicular": 0.250}


def main():
    parameters = pd.read_csv(DATA)
    rows = []
    for orientation, b in WIDTHS.items():
        for n in range(2, 6):
            opening_angle = np.degrees(np.arctan(b / (n * H)))
            for _, parameter in parameters.iterrows():
                phi = float(parameter.friction_angle_deg)
                if abs(phi - opening_angle) <= 2.0:
                    mode = "near_tie"
                else:
                    mode = "sliding" if phi < opening_angle else "toppling"
                rows.append(
                    {
                        "orientation": orientation,
                        "N": n,
                        "opening_angle_deg": opening_angle,
                        "friction_angle_deg": phi,
                        "mu_equivalent": np.tan(np.radians(phi)),
                        "scenario_source_id": parameter.source_id,
                        "parameter_status": parameter.parameter_status,
                        "predicted_mode": mode,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "lunar_friction_range_scenario.csv", index=False)

    critical = (
        result.groupby(["orientation", "N"], as_index=False)
        .agg(
            opening_angle_deg=("opening_angle_deg", "first"),
            min_scenario_friction_angle_deg=("friction_angle_deg", "min"),
            max_scenario_friction_angle_deg=("friction_angle_deg", "max"),
        )
    )
    critical["range_crosses_boundary"] = (
        (critical.min_scenario_friction_angle_deg < critical.opening_angle_deg)
        & (critical.max_scenario_friction_angle_deg > critical.opening_angle_deg)
    )
    critical.to_csv(OUT / "lunar_friction_range_boundary_check.csv", index=False)

    pivot = result.pivot_table(index=["orientation", "N"], columns="friction_angle_deg", values="predicted_mode", aggfunc="first")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True, constrained_layout=True)
    colors = {"sliding": "#2563eb", "toppling": "#dc2626", "near_tie": "#f59e0b"}
    for ax, orientation in zip(axes, WIDTHS):
        subset = result[result.orientation == orientation]
        for _, row in subset.iterrows():
            color = colors[row.predicted_mode]
            ax.scatter(row.friction_angle_deg, row.N, s=85, color=color, edgecolor="black", linewidth=0.4)
            ax.text(row.friction_angle_deg, row.N + 0.10, row.predicted_mode[0].upper(), ha="center", va="bottom", fontsize=8)
        ax.set_title(orientation.replace("concrete_", ""))
        ax.set_xlabel("Scenario friction angle (deg)")
        ax.set_xticks([29.2, 32.0, 36.9, 43.0])
        ax.set_yticks([2, 3, 4, 5])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Stacked courses N")
    fig.suptitle("Material-friction transfer scenario: mode ranking can cross near 32°")
    fig.savefig(OUT / "lunar_friction_range_scenario.png", dpi=220)
    fig.savefig(OUT / "lunar_friction_range_scenario.svg")
    plt.close(fig)

    crossing_count = int(critical.range_crosses_boundary.sum())
    assert crossing_count >= 2
    assert set(result.parameter_status) == {"granular-regolith scenario only", "granular-regolith measured value", "model input only"}
    print(critical.to_string(index=False))
    print(f"\nPASS: {crossing_count} orientation-by-course boundaries are crossed by the scenario range")


if __name__ == "__main__":
    main()
