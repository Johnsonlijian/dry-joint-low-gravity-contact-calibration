from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "costa2024_table2.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

H = 0.100
MU = 0.60
WIDTHS = {"concrete_parallel": 0.125, "concrete_perpendicular": 0.250}


def wilson_interval(successes, total, z=1.96):
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return centre - half_width, centre + half_width


def main():
    raw = pd.read_csv(DATA)
    counts = (
        raw.groupby(["orientation", "N", "mode"], as_index=False)["n"]
        .sum()
        .pivot(index=["orientation", "N"], columns="mode", values="n")
        .fillna(0)
        .reset_index()
    )
    selected = counts[
        ((counts.orientation == "concrete_parallel") & (counts.N == 2))
        | ((counts.orientation == "concrete_perpendicular") & (counts.N == 4))
    ].copy()
    selected["total"] = selected["sliding"] + selected["toppling"]
    selected["slide_fraction"] = selected["sliding"] / selected["total"]
    selected[["ci_low", "ci_high"]] = selected.apply(
        lambda row: pd.Series(wilson_interval(row.sliding, row.total)), axis=1
    )
    selected["b_m"] = selected.orientation.map(WIDTHS)
    selected["h_m"] = H
    selected["geometry_ratio_b_over_Nh"] = selected.b_m / (selected.N * H)
    selected["opening_angle_deg"] = np.degrees(np.arctan(selected.geometry_ratio_b_over_Nh))
    selected["sliding_angle_deg"] = np.degrees(np.arctan(MU))
    selected.to_csv(OUT / "costa2024_identifiability_pairs.csv", index=False)

    ratio_range = selected.geometry_ratio_b_over_Nh.max() - selected.geometry_ratio_b_over_Nh.min()
    fraction_difference = abs(
        selected.loc[selected.orientation == "concrete_parallel", "slide_fraction"].iloc[0]
        - selected.loc[selected.orientation == "concrete_perpendicular", "slide_fraction"].iloc[0]
    )
    summary = pd.DataFrame(
        [
            {
                "comparison": "parallel_N2_vs_perpendicular_N4",
                "geometry_ratio_range": ratio_range,
                "opening_angle_range_deg": selected.opening_angle_deg.max() - selected.opening_angle_deg.min(),
                "sliding_angle_deg": selected.sliding_angle_deg.iloc[0],
                "absolute_slide_fraction_difference": fraction_difference,
                "interpretation": "same scalar geometry score but different observed mode fractions",
            }
        ]
    )
    summary.to_csv(OUT / "costa2024_identifiability_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    labels = ["parallel, N=2", "perpendicular, N=4"]
    x = np.arange(len(selected))
    ax.errorbar(
        x,
        selected.slide_fraction,
        yerr=[selected.slide_fraction - selected.ci_low, selected.ci_high - selected.slide_fraction],
        fmt="o",
        color="#1d4ed8",
        capsize=5,
        markersize=7,
        label="sliding fraction (95% Wilson interval)",
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Observed sliding fraction")
    ax.set_title("Geometrically equivalent near-ties have different mode fractions")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.5,
        0.08,
        "b/(Nh) = 0.625 for both; Δ fraction = 0.40",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    fig.savefig(OUT / "costa2024_identifiability_pairs.png", dpi=220)
    fig.savefig(OUT / "costa2024_identifiability_pairs.svg")
    plt.close(fig)

    assert ratio_range < 1e-12
    assert abs(selected.opening_angle_deg.max() - selected.opening_angle_deg.min()) < 1e-12
    assert fraction_difference > 0.35
    print(selected.to_string(index=False))
    print(f"\nPASS: geometry score identical; observed sliding-fraction difference = {fraction_difference:.2f}")


if __name__ == "__main__":
    main()
