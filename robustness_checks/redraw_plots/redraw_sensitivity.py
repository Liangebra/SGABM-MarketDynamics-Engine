"""Redraw sensitivity-analysis figures for Section 5.6.1 (SCI-Q1 style).

Two figures, both using the document's scenario palette (colors_8):
  Fig 5-7: Final-year carbon reduction response to the grid emission factor
           (GRID_EMISSION_FACTOR +/-20%) across scenarios S0-S6.
  Fig 5-8: Final-year capacity utilization response to the investment
           coefficient (INVESTMENT_COEFFICIENT +/-20%) across scenarios S0-S6.

Style: academic serif fonts, clean light grid, scenario-colored grouped bars
with lightness-graded variants, value labels, professional legend.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

# ---------------------------------------------------------------------------
# Paths & palette
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
# Use production-era sensitivity data (output/production/sensitivity/)
ENGINE_OUT = os.path.join(ROOT, "SGABM-MarketDynamics-Engine-main", "output")
CSV = os.path.join(ENGINE_OUT, "production", "sensitivity", "sensitivity_summary.csv")
OUT_DIR = os.path.join(ROOT, "英文图")
os.makedirs(OUT_DIR, exist_ok=True)

# Document scenario palette (colors_8 from redraw_trends_3x3.R)
SCEN_COLORS = {
    "S0": "#f57c6e",
    "S1": "#f2b56e",
    "S2": "#fbe79e",
    "S3": "#84c3b7",
    "S4": "#88d7da",
    "S5": "#71b8ed",
    "S6": "#b8aeea",
}
SCEN_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.edgecolor": "#4d4d4d",
    "axes.linewidth": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def lighten(hex_color: str, factor: float) -> str:
    """Blend a hex color toward white (factor 0..1)."""
    r, g, b = to_rgb(hex_color)
    mix = lambda c: c + (1.0 - c) * factor  # noqa: E731
    return (mix(r), mix(g), mix(b))


def darken(hex_color: str, factor: float) -> str:
    """Blend a hex color toward black (factor 0..1)."""
    r, g, b = to_rgb(hex_color)
    mix = lambda c: c * (1.0 - factor)  # noqa: E731
    return (mix(r), mix(g), mix(b))


def style_ax(ax: plt.Axes) -> None:
    """Apply the shared academic styling to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linestyle="--", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=4, colors="#4d4d4d")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    # Production CSV uses 'scenario_id'; older CSVs used 'scenario'
    if "scenario_id" in df.columns and "scenario" not in df.columns:
        df = df.rename(columns={"scenario_id": "scenario"})
    df["scenario"] = pd.Categorical(df["scenario"], categories=SCEN_ORDER,
                                    ordered=True)
    return df


def pivot(df: pd.DataFrame, variants: list[str], metric: str) -> pd.DataFrame:
    sub = df[df["variant"].isin(variants)]
    return sub.pivot(index="scenario", columns="variant", values=metric)


def plot_carbon_response(df: pd.DataFrame) -> str:
    """Fig 5-7: final-year carbon reduction vs grid emission factor."""
    variants = ["EF_-20", "baseline", "EF_+20"]
    labels = ["GEF \u221220%", "Baseline", "GEF +20%"]
    pv = pivot(df, variants, "final_carbon_reduction")  # tCO2e
    pv = pv / 1e6  # -> million tCO2e (matches Table 5-2 / chapter scale)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    n_scen = len(SCEN_ORDER)
    n_var = len(variants)
    width = 0.26
    x = np.arange(n_scen)

    for i, (var, lab) in enumerate(zip(variants, labels)):
        vals = pv[var].values
        color = [SCEN_COLORS[s] for s in SCEN_ORDER]
        if var.startswith("EF_-"):
            face = [lighten(c, 0.45) for c in color]
            edge = color
        elif var.startswith("EF_+"):
            face = [darken(c, 0.22) for c in color]
            edge = [darken(c, 0.35) for c in color]
        else:
            face = color
            edge = [darken(c, 0.15) for c in color]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      color=face, edgecolor=edge, linewidth=0.9,
                      label=lab, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + v * 0.02 + 0.004,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8.5,
                    color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(SCEN_ORDER)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Final-year carbon reduction (million tCO$_2$e)")
    ax.set_ylim(0, pv.values.max() * 1.18)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    style_ax(ax)
    ax.legend(title="Grid emission factor", loc="upper left",
              frameon=False, ncol=1)
    ax.set_title("Sensitivity of carbon reduction to the grid emission factor",
                 pad=12)

    out = os.path.join(OUT_DIR, "Fig5-7_Sensitivity_Carbon.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# NOTE: Fig 5-8 is generated by
#   SGABM-MarketDynamics-Engine-main/figures_paper/make_fig58_heatmap_v2.py
# (original GREEN HEATMAP style, matching 5.6.docx). The bar-chart version that
# previously overwrote Fig5-8_Sensitivity_CapacityUtil.png has been removed so the
# heatmap remains the canonical figure for Fig. 5-8.


def main() -> None:
    df = load_data()
    p1 = plot_carbon_response(df)
    print(f"[saved] {p1}")


if __name__ == "__main__":
    main()