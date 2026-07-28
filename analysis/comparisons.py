"""Compare noise-scaling parameters across model families (prints only).

Reports, per auxiliary-MI metric (at the largest cell number):
  - PCA vs Geneformer ratio of fitted u_bar (noise robustness)
  - SCVI vs Geneformer difference in fitted I_max
  - PCA vs Geneformer difference in fitted I_max
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(HERE, "final_results")

df = pd.read_csv(os.path.join(FINAL, "collect_mi_results.csv"))
noise_df = pd.read_csv(os.path.join(FINAL, "noise_scaling.csv"))

print("=== PCA vs Geneformer: fitted u_bar ratio ===")
for metric in noise_df["metric"].unique():
    subset = noise_df[noise_df["metric"] == metric]
    subset = subset[subset["size"] == subset["size"].max()]
    means = subset.groupby("method").mean(numeric_only=True)
    print("Metric: ", metric)
    print("Geneformer ubar: ", means.loc["Geneformer", "fitted_u_bar"])
    print("PCA ubar: ", means.loc["PCA", "fitted_u_bar"])
    print("ratio: ", means.loc["PCA", "fitted_u_bar"] / means.loc["Geneformer", "fitted_u_bar"])
    print()

print("=== SCVI vs Geneformer: fitted I_max difference ===")
for metric in noise_df["metric"].unique():
    subset = noise_df[noise_df["metric"] == metric]
    subset = subset[subset["size"] == subset["size"].max()]
    means = subset.groupby("method").mean(numeric_only=True)
    print("Metric: ", metric)
    print("Geneformer Imax: ", means.loc["Geneformer", "fitted_I_max"])
    print("scvi Imax: ", means.loc["SCVI", "fitted_I_max"])
    print("difference: ", means.loc["SCVI", "fitted_I_max"] - means.loc["Geneformer", "fitted_I_max"])
    print()

print("=== PCA vs Geneformer: fitted I_max difference ===")
for metric in noise_df["metric"].unique():
    subset = noise_df[noise_df["metric"] == metric]
    subset = subset[subset["size"] == subset["size"].max()]
    means = subset.groupby("method").mean(numeric_only=True)
    print("Metric: ", metric)
    print("Geneformer Imax: ", means.loc["Geneformer", "fitted_I_max"])
    print("PCA Imax: ", means.loc["PCA", "fitted_I_max"])
    print("difference: ", means.loc["PCA", "fitted_I_max"] - means.loc["Geneformer", "fitted_I_max"])
    print()