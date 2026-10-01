from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "costa2024_table2.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def pooled_within_mode_sd(subset):
    valid = subset[(subset.n > 1) & subset.observed_sd_deg.notna()].copy()
    degrees_of_freedom = (valid.n - 1).sum()
    variance = (((valid.n - 1) * valid.observed_sd_deg**2).sum()) / degrees_of_freedom
    return float(np.sqrt(variance)), int(degrees_of_freedom)


def main():
    raw = pd.read_csv(DATA)
    pairs = [
        ("concrete_parallel", 2),
        ("concrete_perpendicular", 4),
    ]
    rows = []
    for orientation, courses in pairs:
        subset = raw[(raw.orientation == orientation) & (raw.N == courses)]
        sd, df = pooled_within_mode_sd(subset)
        rows.append(
            {
                "orientation": orientation,
                "N": courses,
                "pooled_within_mode_sd_deg": sd,
                "within_mode_degrees_of_freedom": df,
                "sliding_count": int(subset.loc[subset["mode"] == "sliding", "n"].sum()),
                "total_count": int(subset.n.sum()),
            }
        )
    result = pd.DataFrame(rows)
    combined_sd, combined_df = pooled_within_mode_sd(
        raw[((raw.orientation == "concrete_parallel") & (raw.N == 2))
            | ((raw.orientation == "concrete_perpendicular") & (raw.N == 4))]
    )
    logit_gap = np.log((22 / 8) / (10 / 20))
    result["required_offset_separation_deg"] = result.pooled_within_mode_sd_deg * logit_gap
    result.to_csv(OUT / "costa2024_source_variability_proxy.csv", index=False)
    pd.DataFrame(
        [
            {
                "proxy": "combined_mixed_case_within_mode_sd",
                "sd_proxy_deg": combined_sd,
                "degrees_of_freedom": combined_df,
                "logit_gap": logit_gap,
                "required_offset_separation_deg": combined_sd * logit_gap,
                "status": "source-grounded proxy; not a calibrated contact-law scale",
            }
        ]
    ).to_csv(OUT / "costa2024_source_variability_proxy_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    x = np.arange(len(result))
    width = 0.36
    ax.bar(x - width / 2, result.pooled_within_mode_sd_deg, width, label="within-mode SD proxy", color="#0f766e")
    ax.bar(x + width / 2, result.required_offset_separation_deg, width, label="implied offset separation", color="#f59e0b")
    ax.set_xticks(x, ["parallel, N=2", "perpendicular, N=4"])
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Source-grounded variability proxy for the mixed-mode cases")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(OUT / "costa2024_source_variability_proxy.png", dpi=220)
    fig.savefig(OUT / "costa2024_source_variability_proxy.svg")
    plt.close(fig)

    assert combined_sd > 1.0
    assert result.required_offset_separation_deg.between(1.0, 4.0).all()
    print(result.to_string(index=False))
    print(f"\nPASS: combined within-mode SD proxy = {combined_sd:.3f} deg")


if __name__ == "__main__":
    main()
