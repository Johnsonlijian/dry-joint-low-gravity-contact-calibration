from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

RHO = 1800.0  # kg/m^3, screening assumption
MU = 0.60  # dimensionless, screening assumption
B = 0.50  # m
H = 0.20  # m; h/b = 0.4, squat-block baseline for mode competition
G = 1.62  # m/s^2, lunar scenario
COHESIONS = [0.0, 500.0, 1000.0]  # Pa; scenario values
N_VALUES = np.arange(1, 11)


def thresholds(n, g=G, b=B, h=H, rho=RHO, mu=MU, cohesion=0.0):
    tan_slide = mu
    tan_open = b / (n * h)
    q_slide = mu * rho * b * g + cohesion * b / (n * h)
    q_open = rho * b**2 * g / (n * h)
    return tan_slide, tan_open, q_slide, q_open


def main():
    rows = []
    for n in N_VALUES:
        for cohesion in COHESIONS:
            tan_slide, tan_open, q_slide, q_open = thresholds(n, cohesion=cohesion)
            rows.append(
                {
                    "block_count_N": n,
                    "cohesion_Pa": cohesion,
                    "tan_alpha_slide": tan_slide,
                    "tan_alpha_open": tan_open,
                    "pressure_slide_Pa": q_slide,
                    "pressure_open_Pa": q_open,
                    "pressure_mode": "sliding" if q_slide <= q_open else "opening",
                    "inertial_mode": "sliding" if tan_slide <= tan_open else "opening",
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "stack_contact_thresholds.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for cohesion, style in zip(COHESIONS, ["--", "-", ":"]):
        q_open = [thresholds(n, cohesion=cohesion)[3] for n in N_VALUES]
        q_slide = [thresholds(n, cohesion=cohesion)[2] for n in N_VALUES]
        q_critical = np.minimum(q_open, q_slide)
        axes[0].plot(N_VALUES, q_critical, style, label=f"c = {cohesion:g} Pa")
    axes[0].set_xlabel("Number of aligned blocks N")
    axes[0].set_ylabel("Critical pressure (Pa), Moon")
    axes[0].set_title("Height dilutes the base cohesion contribution")
    axes[0].set_yscale("log")
    axes[0].set_xticks(N_VALUES)
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(N_VALUES, [thresholds(n)[1] for n in N_VALUES], "o-", label="opening")
    axes[1].axhline(MU, color="tab:red", linestyle="--", label="sliding")
    axes[1].set_xlabel("Number of aligned blocks N")
    axes[1].set_ylabel("tan(alpha) threshold")
    axes[1].set_title("Pseudo-static mode competition")
    axes[1].set_xticks(N_VALUES)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("Aligned dry-joint stack baseline (screening model)")
    fig.savefig(OUT / "stack_contact_baseline.png", dpi=220)
    fig.savefig(OUT / "stack_contact_baseline.svg")
    plt.close(fig)

    # Checks: one block is the single-block opening threshold; the mode changes
    # when N*h/b crosses 1/mu.
    assert np.isclose(thresholds(1)[1], B / H)
    n_star = B / (H * MU)
    assert thresholds(int(np.floor(n_star)), cohesion=0.0)[1] >= MU
    assert thresholds(int(np.ceil(n_star)), cohesion=0.0)[1] <= MU
    assert thresholds(10, cohesion=1000.0)[2] > thresholds(10, cohesion=0.0)[2]
    print(table.to_string(index=False))
    print("\nPASS: stack-height, cohesion and mode-boundary checks")


if __name__ == "__main__":
    main()
