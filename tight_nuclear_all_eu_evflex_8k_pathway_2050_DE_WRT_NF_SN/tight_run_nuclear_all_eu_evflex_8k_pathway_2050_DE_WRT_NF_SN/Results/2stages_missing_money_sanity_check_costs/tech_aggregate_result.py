"""
Purpose
Aggregate generator-level missing money results into six technology groups
(Wind, PV, Battery, Gas-CC, Waste, Hydro) and compute summary indicators.

Input
missing_money_w_cost.csv
Generator-level results including Technology, generation, revenues,
subsidies, variable costs, investment costs, and missing-money metrics.

Processing
1. Map original technologies to six aggregated categories.
2. Convert numeric columns and sum values by technology.
3. Compute additional indicators:
       Diff_1 = Rev1 + Sub1 − VarCost1 − InvCost
       Diff_2 = Rev2 − VarCost2 − InvCost
4. Keep selected columns and order technologies:
       Wind, PV, Battery, Gas-CC, Waste, Hydro.

Output
missing_money_w_cost_6techs.csv
Aggregated results by technology including generation, revenues,
costs, missing-money indicators, and Diff_1 / Diff_2.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

IN_FILE = BASE_DIR / "missing_money_w_cost.csv"
OUT_FILE = BASE_DIR / "missing_money_w_cost_6techs.csv"

# 1) read
df = pd.read_csv(IN_FILE)

if "Technology" not in df.columns:
    raise KeyError(f"[{IN_FILE.name}] Missing column 'Technology'. Columns: {df.columns.tolist()}")

# 2) 6-tech mapping
TECH_MAP = {
    # Wind / PV
    "WindOn": "Wind",
    "PV-alpine": "PV",
    "PV-roof": "PV",
    # Battery
    "Battery-TSO": "Battery",
    # Gas-CC
    "GasCC-CCS": "Gas-CC",
    "GasCC-Syn": "Gas-CC",
    # Waste
    "Waste": "Waste",
    # Hydro
    "Dam": "Hydro",
    "Pump-Open": "Hydro",
    "RoR": "Hydro",
}

df = df[df["Technology"].isin(TECH_MAP)].copy()
df["Tech_6"] = df["Technology"].map(TECH_MAP)

# 3) aggregate numeric columns (sum)
num_cols = [c for c in df.columns if c not in ["Technology", "Tech_6"]]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

df_agg = df.groupby("Tech_6", as_index=False)[num_cols].sum()

df_agg = df_agg.rename(columns={"Tech_6": "Technology"})
df_agg["Diff_1"] = df_agg["Rev1"] + df_agg["Sub1"] - df_agg["VarCost1"] - df_agg["InvCost"]
df_agg["Diff_2"] = df_agg["Rev2"] - df_agg["VarCost2"] - df_agg["InvCost"]

keep_cols = [
    "Technology",
    "Gen1",
    "Gen2",
    "Rev1",
    "Rev2",
    "Sub1",
    "VarCost1",
    "VarCost2",
    "InvCost",
    "MM_rev_based",
    "MM_margin_based",
    "Diff_1",
    "Diff_2",
]

df_agg = df_agg[keep_cols]

# 4) order + rename
order = ["Wind", "PV", "Battery", "Gas-CC", "Waste", "Hydro"]
df_agg["Technology"] = pd.Categorical(df_agg["Technology"], categories=order, ordered=True)
df_agg = df_agg.sort_values("Technology").reset_index(drop=True)

df_agg.to_csv(OUT_FILE, index=False)

print("Saved:", OUT_FILE)
print(df_agg)