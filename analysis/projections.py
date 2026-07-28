"""Project the MI gain from a 10x increase in cells or in UMIs/cell (prints only).

Uses the fitted scaling parameters in analysis/final_results/:
  - cell_scaling.csv : cell-number scaling  I(N) = I_inf - (N / N0) ** (-s)
  - noise_scaling.csv: information scaling   I(u) = I_max - 0.5*log2((1 + u/u_bar) / (u/u_bar + 2**(-2*I_max)))
and the observed maxima (cells / umis_per_cell) from collect_mi_results.csv.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(HERE, "final_results")

df = pd.read_csv(os.path.join(FINAL, "collect_mi_results.csv"))


def cell_number_scaling(x, N0, s, I_inf):
    """Cell-number scaling: I(x) = I_inf - (x / N0) ** (-s)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = x / N0
        scaling_term = np.where(ratio > 0, np.exp(-s * np.log(ratio)), 0.0)
        return np.where(x > 0, I_inf - scaling_term, np.nan)


def info_scaling(u, u_bar, I_max):
    """Information (noise) scaling function."""
    eps = 1e-12
    u = np.asarray(u, dtype=float)
    u_bar_safe = np.where(u_bar == 0, eps, u_bar)
    u_over = u / u_bar_safe
    denom = np.where(u_over + 2 ** (-2 * I_max) == 0, eps, u_over + 2 ** (-2 * I_max))
    ratio = np.where((1 + u_over) / denom <= 0, eps, (1 + u_over) / denom)
    return I_max - 0.5 * np.log2(ratio)


# ── 10x more cells ──────────────────────────────────────────────────────────
print("=== projected MI gain from 10x cells ===")
res_df = pd.read_csv(os.path.join(FINAL, "cell_scaling.csv"))
res_df = res_df[(res_df["metric"] != "celltype.l3") & (res_df["quality"] == 1)]

rows = []
for row in res_df.itertuples():
    N0, s, I_inf = row.N0, row.s, row.I_inf
    metric, dataset, method = row.metric, row.dataset, row.method
    max_cells = df[(df["dataset"] == dataset) & (df["algorithm"] == method) & (df["signal"] == metric)]["size"].max()
    cells_10x = max_cells * 10
    I_max = cell_number_scaling(max_cells, N0, s, I_inf)
    I_10x = cell_number_scaling(cells_10x, N0, s, I_inf)
    print(f"{dataset} {method} {metric}: {I_max:.3f} -> {I_10x:.3f} ({(I_10x - I_max) / I_max * 100:.1f}%)")
    rows.append({"method": method, "perc_increase": (I_10x - I_max) / I_max * 100})

projection_df = pd.DataFrame(rows)
print("\nmean projected increase (10x cells):")
print(projection_df.groupby("method")["perc_increase"].mean())

# ── 10x more UMIs/cell ──────────────────────────────────────────────────────
print("\n=== projected MI gain from 10x UMIs/cell ===")
noise_df = pd.read_csv(os.path.join(FINAL, "noise_scaling.csv"))
noise_df = noise_df[noise_df["metric"] != "celltype.l3"]
noise_df = noise_df[noise_df["size"].isin(noise_df.groupby("dataset")["size"].max().values)]

noise_rows = []
for row in noise_df.itertuples():
    u_bar, I_max = row.fitted_u_bar, row.fitted_I_max
    metric, dataset, method = row.metric, row.dataset, row.method
    max_umis = df[(df["dataset"] == dataset) & (df["algorithm"] == method) & (df["signal"] == metric)]["umis_per_cell"].max()
    umis_10x = max_umis * 10
    I_max_val = info_scaling(max_umis, u_bar, I_max)
    I_10x = info_scaling(umis_10x, u_bar, I_max)
    print(f"{dataset} {method} {metric}: {I_max_val:.3f} -> {I_10x:.3f} ({(I_10x - I_max_val) / I_max_val * 100:.1f}%)")
    noise_rows.append({"method": method, "perc_increase": (I_10x - I_max_val) / I_max_val * 100})

noise_proj = pd.DataFrame(noise_rows)
print("\nmean projected increase (10x UMIs):")
print(noise_proj.groupby("method")["perc_increase"].mean())