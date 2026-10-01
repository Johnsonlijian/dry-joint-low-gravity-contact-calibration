from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "costa2024_table2.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

WANG_MU = 0.60


def main():
    raw = pd.read_csv(DATA)
    sliding = raw[(raw["mode"] == "sliding") & (raw.n > 0)].copy()
    sliding["dominance_fraction"] = sliding.n / 30.0
    sliding["mu_eff"] = np.tan(np.radians(sliding.observed_alpha_deg))
    sliding["mu_sd_first_order"] = (
        (1 / np.cos(np.radians(sliding.observed_alpha_deg)) ** 2)
        * np.radians(sliding.observed_sd_deg)
    )
    sliding["delta_mu_vs_wang_0p6"] = sliding.mu_eff - WANG_MU
    sliding["calibration_role"] = np.where(
        sliding.dominance_fraction >= 0.5,
        "sliding-dominant proxy",
        "mixed-case diagnostic only",
    )
    sliding.to_csv(OUT / "costa2024_effective_contact_mu_proxy.csv", index=False)

    eligible = sliding[sliding.dominance_fraction >= 0.5].copy()
    summary_rows = []
    for orientation, subset in eligible.groupby("orientation"):
        weights = subset.n / subset.n.sum()
        weighted_mu = float((subset.mu_eff * weights).sum())
        weighted_sd = float((subset.mu_sd_first_order * weights).sum())
        summary_rows.append(
            {
                "orientation": orientation,
                "eligible_groups": len(subset),
                "weighted_mu_eff": weighted_mu,
                "weighted_within_group_sd_proxy": weighted_sd,
                "delta_vs_wang_mu_0p6": weighted_mu - WANG_MU,
            }
        )
    summary = pd.DataFrame(summary_rows)
    parallel_mu = float(summary.loc[summary.orientation == "concrete_parallel", "weighted_mu_eff"].iloc[0])
    perpendicular_mu = float(summary.loc[summary.orientation == "concrete_perpendicular", "weighted_mu_eff"].iloc[0])
    difference = perpendicular_mu - parallel_mu
    summary["orientation_difference_vs_parallel"] = summary.weighted_mu_eff - parallel_mu
    summary.to_csv(OUT / "costa2024_effective_contact_mu_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    x = np.arange(len(sliding))
    colors = ["#2563eb" if role == "sliding-dominant proxy" else "#f59e0b" for role in sliding.calibration_role]
    ax.errorbar(
        x,
        sliding.mu_eff,
        yerr=sliding.mu_sd_first_order,
        fmt="none",
        ecolor="#374151",
        capsize=3,
        linewidth=1,
    )
    ax.scatter(x, sliding.mu_eff, c=colors, s=65, edgecolor="black", linewidth=0.4)
    ax.axhline(WANG_MU, color="#dc2626", linestyle="--", label="Wang scenario input μ=0.60")
    ax.set_xticks(x, [f"{row.orientation.replace('concrete_', '')}\nN={row.N}" for _, row in sliding.iterrows()])
    ax.set_ylabel("Effective friction proxy μ_eff = tan(mean collapse angle)")
    ax.set_title("Costa benchmark: effective contact-friction proxy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "costa2024_effective_contact_mu_proxy.png", dpi=220)
    fig.savefig(OUT / "costa2024_effective_contact_mu_proxy.svg")
    plt.close(fig)

    assert len(eligible) == 3
    assert difference > 0.03
    print(sliding.to_string(index=False))
    print("\n" + summary.to_string(index=False))
    print(f"\nPASS: perpendicular-minus-parallel weighted mu difference = {difference:.3f}")


if __name__ == "__main__":
    main()
