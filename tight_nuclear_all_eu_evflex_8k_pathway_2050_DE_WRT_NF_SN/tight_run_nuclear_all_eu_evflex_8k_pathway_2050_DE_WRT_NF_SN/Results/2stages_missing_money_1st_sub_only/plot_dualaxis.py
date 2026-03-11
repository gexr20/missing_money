"""
Purpose
Visualize missing money results by technology using a dual-axis plot.

Input
missing_money_table_5tech_aggregated.csv
Technology-level table containing revenue, subsidy, and missing money
values for each technology.

Processing
1. Compute subsidy intensity:
       Subsidy 1st / Revenue 1st (%).
2. Convert missing money values to million CHF.
3. Plot:
       - Bars (left axis): Missing money with and without subsidy.
       - Line (right axis): Subsidy 1st / Revenue 1st (%).

Output
missing_money_5tech_dualaxis.png
Dual-axis figure showing missing money (bars) and subsidy intensity (line)
by technology.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_NAME = "missing_money_table_5tech_aggregated.csv"
OUT_PNG = "missing_money_5tech_dualaxis.png"

def main():
    script_dir = Path(__file__).resolve().parent
    path = script_dir / CSV_NAME
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    tech_col = "Technology"
    rev1_col = "Revenue 1st"
    sub1_col = "Subsidy 1st"
    mm_with_col = "Missing money with subsidy"
    mm_wo_col = "Missing money without"

    needed = [tech_col, rev1_col, sub1_col, mm_with_col, mm_wo_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available: {df.columns.tolist()}")

    for c in [rev1_col, sub1_col, mm_with_col, mm_wo_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Right axis line: Sub1/Rev1 (%)
    df["ratio_1st_pct"] = np.where(df[rev1_col].abs() > 0, df[sub1_col] / df[rev1_col] * 100.0, np.nan)

    # Left axis bars: Missing money in 10^6 CHF
    df["mm_with_million_chf"] = df[mm_with_col] / 1e6
    df["mm_wo_million_chf"] = df[mm_wo_col] / 1e6

    df = df.reset_index(drop=True)
    x = np.arange(len(df))

    fig, axL = plt.subplots(figsize=(18, 5), dpi=300)

    # Left axis: missing money bars
    width = 0.28
    axL.bar(
        x - width/2, df["mm_wo_million_chf"], width=width, alpha=0.80,
        label="Missing money without subsidy (Mio CHF)"
    )
    axL.bar(
        x + width/2, df["mm_with_million_chf"], width=width, alpha=0.80,
        label="Missing money with subsidy (Mio CHF)"
    )
    axL.set_ylabel("Missing money (Mio CHF)")
    axL.set_xlabel("Technology")
    axL.set_xticks(x)
    axL.set_xticklabels(df[tech_col].astype(str).tolist())
    axL.grid(axis="y", alpha=0.3)

    # y=0 dashed reference line (on missing money axis)
    axL.axhline(0.0, linestyle="--", linewidth=1.2)

    # Right axis: ratio line (%)
    axR = axL.twinx()
    axR.plot(
        x, df["ratio_1st_pct"], marker="o", linewidth=2,
        label="Subsidy 1st / Revenue 1st (%)"
    )
    axR.set_ylabel("Subsidy 1st / Revenue 1st (%)")

    # Combined legend
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, loc="upper left", frameon=True)

    axL.set_title("Missing Money (bars) and Subsidy/Revenue Ratio (line) by Technology")

    fig.tight_layout()
    fig.savefig(script_dir / OUT_PNG, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()