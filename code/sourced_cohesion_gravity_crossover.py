from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lunar_cohesion_scenarios.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# These are the existing screening values for the single-interface mechanism
# test. They are not a calibrated block-joint material model.
RHO = 1800.0
H = 0.100
MU = 0.60
GRAVITY = {"Earth": 9.80665, "Moon": 1.62}
STACKS = {"single_unit_N1": 1, "four_units_N4": 4}


def main():
    scenarios = pd.read_csv(DATA)
    rows = []
    for _, scenario in scenarios.iterrows():
        for environment, g in GRAVITY.items():
            for stack_label, n in STACKS.items():
                crossover = MU * RHO * (n * H) * g
                eta_low = scenario.cohesion_pa_low / crossover
                eta_high = scenario.cohesion_pa_high / crossover
                if eta_low < 1 < eta_high:
                    regime = "band_crosses_crossover"
                elif eta_high <= 1:
                    regime = "friction_dominant"
                else:
                    regime = "cohesion_dominant"
                rows.append(
                    {
                        "source_id": scenario.source_id,
                        "scenario": scenario.scenario,
                        "environment": environment,
                        "stack": stack_label,
                        "N": n,
                        "g_m_s2": g,
                        "rho_kg_m3": RHO,
                        "h_m": H,
                        "mu": MU,
                        "crossover_c_pa": crossover,
                        "cohesion_low_pa": scenario.cohesion_pa_low,
                        "cohesion_high_pa": scenario.cohesion_pa_high,
                        "eta_low_c_over_cstar": eta_low,
                        "eta_high_c_over_cstar": eta_high,
                        "regime": regime,
                        "parameter_status": scenario.parameter_status,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "sourced_cohesion_gravity_crossover.csv", index=False)

    plot_data = result[result.source_id == "NASA_TM_2016_shallow"].copy()
    labels = ["Earth N=1", "Earth N=4", "Moon N=1", "Moon N=4"]
    keys = [("Earth", 1), ("Earth", 4), ("Moon", 1), ("Moon", 4)]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    for idx, (environment, n) in enumerate(keys):
        subset = plot_data[(plot_data.environment == environment) & (plot_data.N == n)]
        low = float(subset.eta_low_c_over_cstar.iloc[0])
        high = float(subset.eta_high_c_over_cstar.iloc[0])
        ax.vlines(idx, low, high, color="#2563eb", linewidth=5)
        ax.scatter([idx, idx], [low, high], color="#2563eb", s=45, zorder=3)
    ax.axhline(1, color="#dc2626", linestyle="--", linewidth=1.5, label="c / c* = 1 crossover")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Cohesion ratio c / c*")
    ax.set_title("NASA regolith-cohesion transfer scenario")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(OUT / "sourced_cohesion_gravity_crossover.png", dpi=220)
    fig.savefig(OUT / "sourced_cohesion_gravity_crossover.svg")
    plt.close(fig)

    shallow_moon_n1 = result[
        (result.source_id == "NASA_TM_2016_shallow")
        & (result.environment == "Moon")
        & (result.N == 1)
    ].iloc[0]
    shallow_earth_n1 = result[
        (result.source_id == "NASA_TM_2016_shallow")
        & (result.environment == "Earth")
        & (result.N == 1)
    ].iloc[0]
    assert shallow_earth_n1.regime == "friction_dominant"
    assert shallow_moon_n1.regime == "cohesion_dominant"
    assert result.shape[0] == 8
    print(result.to_string(index=False))
    print("\nPASS: shallow cohesion scenario changes regime from Earth N=1 to Moon N=1")


if __name__ == "__main__":
    main()
