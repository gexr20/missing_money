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

import pandas as pd
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

df_agg.to_csv(OUT_FILE, index=False)

print("Saved:", OUT_FILE)
print(df_agg)