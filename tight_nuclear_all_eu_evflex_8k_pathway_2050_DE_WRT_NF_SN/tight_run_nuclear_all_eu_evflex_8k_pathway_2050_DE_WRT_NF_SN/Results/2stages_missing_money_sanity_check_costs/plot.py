"""
Purpose
Visualize per-MWh revenue, subsidy and missing money by technology.

Input
missing_money_w_cost_6techs.csv

Processing
Bars:
    MM_rev_per_MWh
    MM_margin_per_MWh

Lines:
    Diff_1_per_MWh
    Diff_2_per_MWh

Output
missing_money_perMWh_dualaxis.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_NAME = "missing_money_w_cost_6techs.csv"
OUT_PNG = "missing_money_perMWh_dualaxis.png"


def main():

    script_dir = Path(__file__).resolve().parent
    df = pd.read_csv(script_dir / CSV_NAME)

    x = np.arange(len(df))

    fig, axL = plt.subplots(figsize=(18, 5), dpi=300)

    width = 0.22

    axL.bar(
        x - width,
        df["MM_rev_per_MWh"],
        width=width,
        label="MM_rev/MWh"
    )

    axL.bar(
        x,
        df["MM_margin_per_MWh"],
        width=width,
        label="MM_margin/MWh"
    )
    
    axL.set_ylabel("Missing money in different calculations (CHF/MWh)")
    axL.set_xticks(x)
    axL.set_xticklabels(df["Technology"])
    axL.grid(axis="y", alpha=0.3)

    axR = axL.twinx()

    axR.plot(
        x,
        df["Diff_1_per_MWh"],
        marker="o",
        linewidth=2,
        label="Diff_1/MWh"
    )

    axR.plot(
        x,
        df["Diff_2_per_MWh"],
        marker="s",
        linewidth=2,
        label="Diff_2/MWh"
    )

    axR.set_ylabel("Profit from market in 2 stages (CHF/MWh)")
    axR.axhline(0, linestyle="--")

    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()

    axL.legend(h1 + h2, l1 + l2, loc="upper left")

    plt.title("Revenue / Subsidy and Missing Money per MWh by Technology")

    plt.tight_layout()
    plt.savefig(script_dir / OUT_PNG)
    plt.close()


if __name__ == "__main__":
    main()