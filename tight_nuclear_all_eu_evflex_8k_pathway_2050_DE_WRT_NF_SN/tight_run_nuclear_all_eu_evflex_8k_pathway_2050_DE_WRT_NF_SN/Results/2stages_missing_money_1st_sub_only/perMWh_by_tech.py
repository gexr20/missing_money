"""
Purpose
Aggregate technology-level missing money results into 5 main categories
and add revenue/subsidy intensity indicators per MWh.

Input
missing_money_table_by_technology.csv
annual_generation_by_technology.csv

Processing
1. Merge money table with annual generation table by Technology.
2. Map detailed technologies into 5 main categories.
3. Sum revenue, generation, subsidy, and missing money by category.
4. Add unit indicators:
       - Rev1_per_MWh = Revenue 1st / Generation 1st
       - Sub1_per_MWh = Subsidy 1st / Generation 1st
       - Rev2_per_MWh = Revenue 2nd / Generation 2nd
       - MM_with_sub_per_MWh = Missing money with subsidy / Generation 2nd
       - MM_without_sub_per_MWh = Missing money without subsidy / Generation 2nd

Output
missing_money_table_5tech_aggregated.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MM_FILE = BASE_DIR / "missing_money_table_by_technology.csv"
GEN_FILE = BASE_DIR / "annual_generation_by_technology.csv"
OUT_FILE = BASE_DIR / "missing_money_table_5tech_aggregated.csv"

# 1) read missing money table (no generation columns)
mm = pd.read_csv(MM_FILE)

# 2) read annual generation table (has Generation 1st/2nd)
gen = pd.read_csv(GEN_FILE)

# 3) merge to attach generation columns
df = mm.merge(gen, on="Technology", how="left")
df["Generation 1st"] = df["Generation 1st"].fillna(0.0)
df["Generation 2nd"] = df["Generation 2nd"].fillna(0.0)

# 4) 5-tech mapping
TECH_MAP = {
    "WindOn": "Wind",
    "PV-alpine": "PV",
    "PV-roof": "PV",
    "Waste": "Waste",
    "GasCC-CCS": "Gas CC",
    "GasCC-Syn": "Gas CC",
    "Dam": "Hydro",
    "Pump-Open": "Hydro",
    "RoR": "Hydro",
}

df = df[df["Technology"].isin(TECH_MAP)].copy()
df["Tech_5"] = df["Technology"].map(TECH_MAP)

agg_cols = [
    "Revenue 1st",
    "Generation 1st",
    "Revenue 2nd",
    "Generation 2nd",
    "Subsidy 1st",
    "Subsidy 2nd",
    "Missing money with subsidy",
    "Missing money without",
]

df_agg = df.groupby("Tech_5", as_index=False)[agg_cols].sum()

order = ["Wind", "PV", "Waste", "Gas CC", "Hydro"]
df_agg["Tech_5"] = pd.Categorical(df_agg["Tech_5"], categories=order, ordered=True)
df_agg = df_agg.sort_values("Tech_5").rename(columns={"Tech_5": "Technology"})

# 5) add value-per-MWh columns
df_agg["Rev1_per_MWh"] = np.where(
    df_agg["Generation 1st"] != 0,
    df_agg["Revenue 1st"] / df_agg["Generation 1st"],
    np.nan
)

df_agg["Sub1_per_MWh"] = np.where(
    df_agg["Generation 1st"] != 0,
    df_agg["Subsidy 1st"] / df_agg["Generation 1st"],
    np.nan
)

df_agg["Rev2_per_MWh"] = np.where(
    df_agg["Generation 2nd"] != 0,
    df_agg["Revenue 2nd"] / df_agg["Generation 2nd"],
    np.nan
)

df_agg["MM_with_sub_per_MWh"] = np.where(
    df_agg["Generation 2nd"] != 0,
    df_agg["Missing money with subsidy"] / df_agg["Generation 2nd"],
    np.nan
)

df_agg["MM_without_sub_per_MWh"] = np.where(
    df_agg["Generation 2nd"] != 0,
    df_agg["Missing money without"] / df_agg["Generation 2nd"],
    np.nan
)

df_agg.to_csv(OUT_FILE, index=False)

print("Saved:", OUT_FILE)
print(df_agg)