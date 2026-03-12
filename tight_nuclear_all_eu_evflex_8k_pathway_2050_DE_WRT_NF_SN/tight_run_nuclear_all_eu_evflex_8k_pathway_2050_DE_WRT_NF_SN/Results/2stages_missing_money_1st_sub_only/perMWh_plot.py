"""
Purpose
Plot aggregated per-MWh indicators by technology.

Input
missing_money_table_5tech_aggregated.csv

Processing
1. Read aggregated 5-technology table.
2. Use bars for:
       - Rev1_per_MWh
       - Sub1_per_MWh
       - Rev2_per_MWh
3. Use lines for:
       - MM_with_sub_per_MWh
       - MM_without_sub_per_MWh

Output
missing_money_5tech_perMWh.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_NAME = "missing_money_table_5tech_aggregated.csv"
OUT_PNG = "missing_money_5tech_perMWh.png"

def main():
    script_dir = Path(__file__).resolve().parent
    path = script_dir / CSV_NAME
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    tech_col = "Technology"
    bar_cols = [
        "Rev1_per_MWh",
        "Sub1_per_MWh",
        "Rev2_per_MWh",
    ]
    line_cols = [
        "MM_with_sub_per_MWh",
        "MM_without_sub_per_MWh",
    ]

    needed = [tech_col] + bar_cols + line_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available: {df.columns.tolist()}")

    for c in bar_cols + line_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.reset_index(drop=True)
    x = np.arange(len(df))

    fig, axL = plt.subplots(figsize=(18, 6), dpi=300)

    width = 0.22
    axL.bar(
        x - width,
        df["Rev1_per_MWh"],
        width=width,
        alpha=0.85,
        label="Rev1/Gen1"
    )

    axL.bar(
        x,
        df["Sub1_per_MWh"],
        width=width,
        alpha=0.85,
        label="Sub1/Gen1"
    )

    axL.bar(
        x + width,
        df["Rev2_per_MWh"],
        width=width,
        alpha=0.85,
        label="Rev2/Gen2"
    )

    axL.set_xlabel("Technology")
    axL.set_ylabel("Revenue / Subsidy per MWh (CHF/MWh)")
    axL.set_xticks(x)
    axL.set_xticklabels(df[tech_col].astype(str).tolist())
    axL.grid(axis="y", alpha=0.3)

    axR = axL.twinx()
    axR.plot(
        x,
        df["MM_with_sub_per_MWh"],
        marker="o",
        linewidth=2,
        label="MM with sub / MWh"
    )

    axR.plot(
        x,
        df["MM_without_sub_per_MWh"],
        marker="s",
        linewidth=2,
        label="MM w/o sub / MWh"
    )
    axR.set_ylabel("Missing money per MWh (CHF/MWh)")
    axR.axhline(0.0, linestyle="--", linewidth=1.2)

    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, loc="upper left", frameon=True)

    axL.set_title("Per-MWh Revenue, Subsidy, and Missing Money by Technology")

    fig.tight_layout()
    fig.savefig(script_dir / OUT_PNG, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()