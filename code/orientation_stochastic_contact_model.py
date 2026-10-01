from math import erf, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "costa2024_table2.csv"
PROXY = ROOT / "outputs" / "costa2024_effective_contact_mu_proxy.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def normal_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return centre - half_width, centre + half_width


def main():
    raw = pd.read_csv(DATA)
    proxy = pd.read_csv(PROXY)
    eligible = proxy[proxy.calibration_role == "sliding-dominant proxy"].copy()

    # The pooled model is the scalar-contact negative control. Both models use
    # exactly the same calibration groups; only the orientation term differs.
    pooled_mu = np.average(eligible.mu_eff, weights=eligible.n)
    pooled_sd = np.average(eligible.mu_sd_first_order, weights=eligible.n)

    # Build the two orientation summaries explicitly so the script works with
    # pandas versions before and after the groupby.apply include_groups change.
    orientation_rows = []
    for orientation, group in eligible.groupby("orientation"):
        orientation_rows.append(
            {
                "orientation": orientation,
                "mu_mean": np.average(group.mu_eff, weights=group.n),
                "mu_sd_proxy": np.average(group.mu_sd_first_order, weights=group.n),
            }
        )
    orientation_parameters = pd.DataFrame(orientation_rows)

    rows = []
    targets = [("concrete_parallel", 2, "in-sample orientation fit"), ("concrete_perpendicular", 4, "held-out course count")]
    for orientation, n, validation_role in targets:
        b = 0.125 if orientation == "concrete_parallel" else 0.250
        critical_mu = b / (n * 0.100)
        params = orientation_parameters[orientation_parameters.orientation == orientation].iloc[0]
        predicted_fraction = normal_cdf((critical_mu - params.mu_mean) / params.mu_sd_proxy)
        pooled_fraction = normal_cdf((critical_mu - pooled_mu) / pooled_sd)
        subset = raw[(raw.orientation == orientation) & (raw.N == n)]
        observed_sliding = int(subset.loc[subset["mode"] == "sliding", "n"].sum())
        observed_total = int(subset.n.sum())
        observed_fraction = observed_sliding / observed_total
        ci_low, ci_high = wilson_interval(observed_sliding, observed_total)
        rows.append(
            {
                "orientation": orientation,
                "N": n,
                "validation_role": validation_role,
                "mu_mean": params.mu_mean,
                "mu_sd_proxy": params.mu_sd_proxy,
                "critical_mu_for_sliding": critical_mu,
                "predicted_sliding_fraction": predicted_fraction,
                "pooled_scalar_predicted_fraction": pooled_fraction,
                "pooled_scalar_absolute_error": abs(pooled_fraction - observed_fraction),
                "observed_sliding_fraction": observed_fraction,
                "observed_wilson95_low": ci_low,
                "observed_wilson95_high": ci_high,
                "absolute_fraction_error": abs(predicted_fraction - observed_fraction),
                "observed_sliding_count": observed_sliding,
                "observed_total_count": observed_total,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "orientation_stochastic_contact_predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "pooled scalar contact",
                "mu_mean": pooled_mu,
                "mu_sd_proxy": pooled_sd,
                "calibration_groups": len(eligible),
            },
            {
                "model": "orientation-specific contact",
                "mu_mean": np.nan,
                "mu_sd_proxy": np.nan,
                "calibration_groups": len(eligible),
            },
        ]
    ).to_csv(OUT / "orientation_stochastic_contact_model_comparison.csv", index=False)
    orientation_parameters.to_csv(OUT / "orientation_stochastic_contact_parameters.csv", index=False)

    heldout = result[result.validation_role == "held-out course count"].iloc[0]
    heldout_params = orientation_parameters[
        orientation_parameters.orientation == heldout.orientation
    ].iloc[0]
    sensitivity_rows = []
    for multiplier in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        s_used = heldout_params.mu_sd_proxy * multiplier
        prediction = normal_cdf(
            (heldout.critical_mu_for_sliding - heldout_params.mu_mean) / s_used
        )
        sensitivity_rows.append(
            {
                "s_mu_multiplier": multiplier,
                "s_mu_used": s_used,
                "predicted_fraction": prediction,
                "observed_fraction": heldout.observed_sliding_fraction,
                "absolute_error": abs(prediction - heldout.observed_sliding_fraction),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUT / "orientation_stochastic_contact_sensitivity.csv", index=False)

    x = np.arange(len(result))
    width = 0.24
    fig, (ax, ax_sens) = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    observed = result.observed_sliding_fraction.to_numpy()
    yerr = np.vstack(
        [observed - result.observed_wilson95_low, result.observed_wilson95_high - observed]
    )
    ax.bar(x - width, observed, width, label="observed", color="#2563eb")
    ax.errorbar(x - width, observed, yerr=yerr, fmt="none", ecolor="#1e3a8a", capsize=3, lw=1)
    ax.bar(x, result.pooled_scalar_predicted_fraction, width, label="pooled scalar μ", color="#94a3b8")
    ax.bar(x + width, result.predicted_sliding_fraction, width, label="orientation-specific μ", color="#f59e0b")
    ax.set_xticks(x, ["parallel, N=2", "perpendicular, N=4"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Sliding fraction")
    ax.set_title("Held-out comparison of scalar and orientation-specific contact models")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax_sens.plot(
        sensitivity.s_mu_multiplier,
        sensitivity.predicted_fraction,
        marker="o",
        color="#f59e0b",
        label="orientation-specific μ",
    )
    ax_sens.axhline(
        heldout.observed_sliding_fraction,
        color="#2563eb",
        lw=1.4,
        label="observed",
    )
    ax_sens.axhspan(
        heldout.observed_wilson95_low,
        heldout.observed_wilson95_high,
        color="#2563eb",
        alpha=0.12,
        label="95% Wilson interval",
    )
    ax_sens.set_xlabel("Multiplier on proxy spread $s_\\mu$")
    ax_sens.set_ylabel("Held-out sliding fraction")
    ax_sens.set_ylim(0, 1)
    ax_sens.set_xticks(sensitivity.s_mu_multiplier)
    ax_sens.grid(alpha=0.25)
    ax_sens.legend(frameon=False, fontsize=8)
    ax_sens.set_title("Sensitivity to proxy spread")
    fig.savefig(OUT / "orientation_stochastic_contact_predictions.png", dpi=220)
    fig.savefig(OUT / "orientation_stochastic_contact_predictions.svg")
    plt.close(fig)

    assert len(result) == 2
    assert result.loc[result.validation_role == "held-out course count", "absolute_fraction_error"].iloc[0] < 0.05
    assert result.loc[result.validation_role == "held-out course count", "pooled_scalar_absolute_error"].iloc[0] > result.loc[result.validation_role == "held-out course count", "absolute_fraction_error"].iloc[0]
    print(orientation_parameters.to_string(index=False))
    print(f"\nPooled scalar model: mu={pooled_mu:.6f}, s_mu={pooled_sd:.6f}")
    print("\n" + result.to_string(index=False))
    improvement = heldout.pooled_scalar_absolute_error - heldout.absolute_fraction_error
    print(f"\nPASS: held-out orientation model error < 0.05 and improves pooled scalar model by {improvement:.3f}")
    print("\nSensitivity to proxy spread:")
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
