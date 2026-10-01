from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "costa2024_identifiability_pairs.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def logit(p):
    return np.log(p / (1 - p))


def main():
    pair = pd.read_csv(DATA)
    p_parallel = float(pair.loc[pair.orientation == "concrete_parallel", "slide_fraction"].iloc[0])
    p_perpendicular = float(pair.loc[pair.orientation == "concrete_perpendicular", "slide_fraction"].iloc[0])
    logit_gap = logit(p_parallel) - logit(p_perpendicular)

    # Diagnostic model: P(sliding) = sigmoid((M + delta_orientation)/s).
    # Only the offset difference is identifiable from these two counts.
    scales = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    rows = []
    for scale in scales:
        separation = scale * logit_gap
        rows.append(
            {
                "angular_variability_scale_deg": scale,
                "logit_gap": logit_gap,
                "required_orientation_offset_separation_deg": separation,
                "centered_parallel_offset_deg": separation / 2,
                "centered_perpendicular_offset_deg": -separation / 2,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "minimal_variability_required_offsets.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.plot(
        result.angular_variability_scale_deg,
        result.required_orientation_offset_separation_deg,
        "o-",
        color="#7c3aed",
    )
    ax.set_xlabel("Assumed angular variability scale s (deg)")
    ax.set_ylabel("Required orientation-offset separation (deg)")
    ax.set_title("Diagnostic bound from the two mixed-mode cases")
    ax.grid(alpha=0.25)
    ax.text(
        0.05,
        0.92,
        f"logit gap = {logit_gap:.2f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    fig.savefig(OUT / "minimal_variability_required_offsets.png", dpi=220)
    fig.savefig(OUT / "minimal_variability_required_offsets.svg")
    plt.close(fig)

    assert logit_gap > 1.6
    assert np.allclose(result.required_orientation_offset_separation_deg, scales * logit_gap)
    print(result.to_string(index=False))
    print("\nPASS: offset separation scales linearly with the assumed variability scale")


if __name__ == "__main__":
    main()
