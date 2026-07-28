"""Compute eta_90 (u90) noise-tolerance values per metric and model family.

For each (metric, method) the noise-scaling fit at the largest cell number is
used to compute u90 -- the UMIs/cell needed to reach 90% of the achievable
information -- with 2-sigma error bars propagated from the fit uncertainties.

Writes a LaTeX table (incl. STATE) to analysis/figures/u90_table.txt and prints it.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(HERE, "final_results")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)

df = pd.read_csv(os.path.join(FINAL, "collect_mi_results.csv"))
noise_df = pd.read_csv(os.path.join(FINAL, "noise_scaling.csv"))

# Display order / labels for the four auxiliary-MI metrics and the model families.
METRIC_ORDER = ["author_day", "clone", "ng_idx", "protein_counts"]
METRIC_LABEL = {
    "author_day": "Temporal MI",
    "clone": "Clonal MI",
    "ng_idx": "Spatial MI",
    "protein_counts": "Protein MI",
}
METHOD_ORDER = ["Geneformer", "PCA", "RandomProjection", "SCVI", "State"]
METHOD_LABEL = {
    "Geneformer": "Geneformer",
    "PCA": "PCA",
    "RandomProjection": "Random Projection",
    "SCVI": "SCVI",
    "State": "STATE",
}
Q = 0.95


def uq(ubar, Imax, q=Q):
    I = Imax + np.log2(q)
    return ubar * ((2 ** (2 * I) - 1) / (2 ** (2 * Imax) - 2 ** (2 * I)))


def uq_error(ubar, Imax, ubar_error, Imax_error, q=Q):
    I = Imax + np.log2(q)
    du_dubar = (2 ** (2 * I) - 1) / (2 ** (2 * Imax) - 2 ** (2 * I))
    numerator = 2 ** (2 * I) - 1
    denominator = 2 ** (2 * Imax) - 2 ** (2 * I)
    d_num = 2 ** (2 * I) * 2 * np.log(2)
    d_denom = 2 ** (2 * Imax) * 2 * np.log(2) - 2 ** (2 * I) * 2 * np.log(2)
    du_dImax = ubar * (d_num * denominator - numerator * d_denom) / (denominator ** 2)
    return np.sqrt((du_dubar * ubar_error) ** 2 + (du_dImax * Imax_error) ** 2)


# u90 (value + 2-sigma error) per (metric, method) at the largest cell number.
u90_dict, u90_error_dict = {}, {}
for metric in METRIC_ORDER:
    u90_dict[metric], u90_error_dict[metric] = {}, {}
    for method in METHOD_ORDER:
        subset = noise_df[(noise_df["metric"] == metric) & (noise_df["method"] == method)]
        if subset.empty:
            u90_dict[metric][method] = np.nan
            u90_error_dict[metric][method] = np.nan
            continue
        subset = subset[subset["size"] == subset["size"].max()]
        means = subset.mean(numeric_only=True)
        u90_dict[metric][method] = uq(means["fitted_u_bar"], means["fitted_I_max"])
        u90_error_dict[metric][method] = uq_error(
            means["fitted_u_bar"], means["fitted_I_max"],
            means["u_bar_error"], means["I_max_error"],
        )

max_umis = df.groupby("signal")["umis_per_cell"].max().to_dict()

# ── LaTeX table ─────────────────────────────────────────────────────────────
lines = []
lines.append("\\begin{table}[h]")
lines.append("\\centering")
lines.append("\\begin{tabular}{l" + "r" * (len(METHOD_ORDER) + 1) + "}")
lines.append("\\hline")
lines.append("Metric & " + " & ".join(METHOD_LABEL[m] for m in METHOD_ORDER) + " & Actual UMIs \\\\")
lines.append("\\hline")
lines.append("")
for metric in METRIC_ORDER:
    cells = []
    for method in METHOD_ORDER:
        v, e = u90_dict[metric][method], u90_error_dict[metric][method]
        cells.append("--" if np.isnan(v) else f"${v:.1f} \\pm {e:.1f}$")
    actual = max_umis.get(metric)
    actual_str = f"{actual:.0f}" if isinstance(actual, (int, float)) else str(actual)
    lines.append(f"{METRIC_LABEL[metric]} & " + " & ".join(cells) + f" & {actual_str} \\\\")
lines.append("")
lines.append("\\hline")
lines.append("\\end{tabular}")
lines.append("\\caption{$\\eta_{90}$ values by metric and model family, with $\\pm2\\sigma$.}")
lines.append("\\label{tab:u90}")
lines.append("\\end{table}")

table = "\n".join(lines)
print(table)

out_path = os.path.join(FIGURES, "u90_table.txt")
with open(out_path, "w") as f:
    f.write(table + "\n")
print(f"\nwrote {os.path.relpath(out_path, HERE)}")