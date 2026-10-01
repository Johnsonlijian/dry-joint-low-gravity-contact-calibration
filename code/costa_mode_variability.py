from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "costa2024_table2.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    data = pd.read_csv(DATA)
    counts = (
        data.groupby(["orientation", "N", "mode"], as_index=False)["n"]
        .sum()
        .pivot(index=["orientation", "N"], columns="mode", values="n")
        .fillna(0)
        .reset_index()
    )
    for mode in ["sliding", "toppling"]:
        if mode not in counts:
            counts[mode] = 0
    counts["total"] = counts["sliding"] + counts["toppling"]
    counts["sliding_fraction"] = counts["sliding"] / counts["total"]
    counts["toppling_fraction"] = counts["toppling"] / counts["total"]
    counts.to_csv(OUT / "costa2024_mode_fractions.csv", index=False)

    assert set(counts.total) == {30}
    assert counts.sliding_fraction.between(0, 1).all()
    assert counts.toppling_fraction.between(0, 1).all()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True, constrained_layout=True)
    colors = {"sliding": "#3b82f6", "toppling": "#ef4444"}
    for ax, orientation in zip(axes, ["concrete_parallel", "concrete_perpendicular"]):
        subset = counts[counts.orientation == orientation].sort_values("N")
        ax.bar(subset.N, subset.sliding_fraction, color=colors["sliding"], label="sliding")
        ax.bar(
            subset.N,
            subset.toppling_fraction,
            bottom=subset.sliding_fraction,
            color=colors["toppling"],
            label="toppling",
        )
        for _, row in subset.iterrows():
            ax.text(row.N, 0.5, f"{int(row.sliding)}/{int(row.total)}\n{int(row.toppling)}/{int(row.total)}", ha="center", va="center", fontsize=8)
        ax.set_title(orientation.replace("concrete_", ""))
        ax.set_xlabel("Stacked courses N")
        ax.set_xticks([2, 3, 4, 5])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Observed mode fraction")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Costa et al. (2024): mixed collapse modes")
    fig.savefig(OUT / "costa2024_mode_fractions.png", dpi=220)
    fig.savefig(OUT / "costa2024_mode_fractions.svg")
    plt.close(fig)

    print(counts.to_string(index=False))
    print("\nPASS: each orientation-by-course group contains 30 observations")


if __name__ == "__main__":
    main()
