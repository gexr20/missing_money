"""
Purpose:
Compare revenue-based and margin-based missing money results at the technology level,
and check the impact of variable costs, investment costs, and RES subsidies.

Processing:
- Time range limited to 0–8759.
- Aggregates generator-level results to Technology level.
- Computes totals for:
  Gen1, Gen2, Rev1, Rev2, Sub1, VarCost1, VarCost2, InvCost.

Sanity check indicators:
- MM_rev_based
- MM_margin_based
- ProfitDiff_inv_once
- ProfitDiff_inv_twice
- Delta_margin_vs_rev
- VarCost1_minus_VarCost2
- MM_rev_over_(Inv+Var2)
- MM_margin_over_(Inv+Var2)

Output files:
- missing_money_w_cost.csv
- missing_money_w_cost_with_total.csv

"""

from pathlib import Path
import pandas as pd

# =========================
# CONFIG (keep aligned with tech_1st_sub_only_missing_money.py)
# =========================
OUT_NAME = "2stages_missing_money"
VERSION = "sanity_check_costs"
COUNTRY_FILTER = "CH"

BASE_DIR = Path(__file__).resolve().parent / "tight_run_nuclear_all_eu_evflex_8k_pathway_2050_DE_WRT_NF_SN"

DATA_GENERATORS = BASE_DIR / "CentIv_2050" / "mappings" / "Data_generators.csv"

# Prices
ELPRICE_CH = BASE_DIR / "CentIv_2050" / "ElPrice_hourly_CH.xlsx"
NODAL_DUAL = BASE_DIR / "InvestmentRun_2050" / "NodalConstraint_one_CH_dual.csv"

# Generation
GENERATION_WIDE_2ND = BASE_DIR / "CentIv_2050" / "GenerationPerGen_hourly_ALL_LP.xlsx"
GEN_SHEET_2ND = "Generation_MWh"

GENERATION_LONG_1ST = BASE_DIR / "InvestmentRun_2050" / "PowerGenerated.csv"

# Optional subsidy dual (CHF/MWh, scalar)
RES_DUAL = BASE_DIR / "InvestmentRun_2050" / "RESCon_dual.csv"

TIME_MIN = 0
TIME_MAX = 8759

TECH_ORDER = [
    "Battery-TSO",
    "DAC",
    "Dam",
    "FuelCell",
    "GasCC-CCS",
    "GasCC-Syn",
    "Nuclear",
    "PV-alpine",
    "PV-roof",
    "Pump-Open",
    "RoR",
    "Waste",
    "WindOn",
]


# =========================
# Helpers (kept local for “parallel script” use)
# =========================
def load_data_generators_ch_meta_costs(path: Path, country_code: str = "CH") -> pd.DataFrame:
    """
    Need columns:
      - first column: GenID
      - Country, Technology, GenType
      - InvCost, TotVarCost
    Return CH only:
      GenID(int), Technology(str), GenType(str), InvCost(float), TotVarCost(float)
    """
    df = pd.read_csv(path)
    genid_col = df.columns[0]

    required = ["Country", "Technology", "GenType", "InvCost", "TotVarCost"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Data_generators.csv missing required columns: {missing}")

    out = df[[genid_col, "Country", "Technology", "GenType", "InvCost", "TotVarCost"]].copy()
    out = out.rename(columns={genid_col: "GenID"})

    out["GenID"] = pd.to_numeric(out["GenID"], errors="coerce")
    out["Country"] = out["Country"].astype(str).str.upper()
    out["Technology"] = out["Technology"].astype(str)
    out["GenType"] = out["GenType"].astype(str)

    out["InvCost"] = pd.to_numeric(out["InvCost"], errors="coerce").fillna(0.0).astype(float)
    out["TotVarCost"] = pd.to_numeric(out["TotVarCost"], errors="coerce").fillna(0.0).astype(float)

    out = out.dropna(subset=["GenID"]).copy()
    out["GenID"] = out["GenID"].astype(int)

    out = out.loc[out["Country"] == str(country_code).upper()].copy()
    out = out.drop_duplicates(subset=["GenID"]).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No generators found with Country == {country_code}")
    return out


def load_price_elprice_ch00(path: Path) -> pd.DataFrame:
    df0 = pd.read_excel(path, header=None, engine="openpyxl")
    time = pd.to_numeric(df0.iloc[1:, 0], errors="coerce")
    price = pd.to_numeric(df0.iloc[1:, 1], errors="coerce")
    out = pd.DataFrame({"time": time, "price_2nd": price}).dropna(subset=["time", "price_2nd"]).copy()
    out["time"] = out["time"].astype(int)
    out["price_2nd"] = out["price_2nd"].astype(float)
    out = out[(out["time"] >= TIME_MIN) & (out["time"] <= TIME_MAX)].reset_index(drop=True)
    return out


def load_price_dual_1st_two_cols(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.iloc[:, :2].copy()
    out.columns = ["time", "price_1st"]
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["price_1st"] = pd.to_numeric(out["price_1st"], errors="coerce")
    out = out.dropna(subset=["time", "price_1st"]).copy()
    out["time"] = out["time"].astype(int)
    out["price_1st"] = out["price_1st"].astype(float)
    out = out[(out["time"] >= TIME_MIN) & (out["time"] <= TIME_MAX)].reset_index(drop=True)
    return out


def load_generation_long_3cols(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, low_memory=False)
    if df.shape[1] < 3:
        raise ValueError(f"{path.name}: expected >=3 cols (GenID,time,generation).")
    out = df.iloc[:, :3].copy()
    out.columns = ["GenID", "time", "generation"]
    out["GenID"] = pd.to_numeric(out["GenID"], errors="coerce")
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["generation"] = pd.to_numeric(out["generation"], errors="coerce")
    out = out.dropna(subset=["GenID", "time", "generation"]).copy()
    out["GenID"] = out["GenID"].astype(int)
    out["time"] = out["time"].astype(int)
    out["generation"] = out["generation"].astype(float)
    out = out[(out["time"] >= TIME_MIN) & (out["time"] <= TIME_MAX)].copy()
    return out


def load_generation_wide_selected(path: Path, sheet: str, gen_ids: list[int]) -> pd.DataFrame:
    df0 = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")
    gen_row = pd.to_numeric(df0.iloc[0, 1:], errors="coerce")
    col_map: dict[int, int] = {}
    for j, gid in enumerate(gen_row, start=1):
        if pd.notna(gid):
            col_map[int(gid)] = j

    time_series = pd.to_numeric(df0.iloc[3:, 0], errors="coerce").dropna().astype(int).values
    n_t = len(time_series)

    blocks = []
    for g in gen_ids:
        j = col_map.get(int(g))
        if j is None:
            continue
        vals = pd.to_numeric(df0.iloc[3 : 3 + n_t, j], errors="coerce").fillna(0.0).values
        blocks.append(pd.DataFrame({"GenID": int(g), "time": time_series, "generation": vals}))

    if not blocks:
        return pd.DataFrame(columns=["GenID", "time", "generation"])

    out = pd.concat(blocks, ignore_index=True)
    out["generation"] = pd.to_numeric(out["generation"], errors="coerce").fillna(0.0).astype(float)
    out = out[(out["time"] >= TIME_MIN) & (out["time"] <= TIME_MAX)].copy()
    return out


def load_res_dual_rate_optional(path: Path) -> float:
    if not path.exists():
        return 0.0
    df = pd.read_csv(path, header=None)
    if df.shape[0] < 2 or df.shape[1] < 2:
        raise ValueError("RESCon_dual.csv format not recognized (need >=2 rows and >=2 cols).")
    val = pd.to_numeric(df.iloc[1, 1], errors="coerce")
    if pd.isna(val):
        raise ValueError("RESCon_dual.csv: cannot parse scalar at B2 (row2,col2).")
    return float(val)


def force_tech_order(df: pd.DataFrame) -> pd.DataFrame:
    full = pd.DataFrame({"Technology": TECH_ORDER})
    return full.merge(df, on="Technology", how="left").fillna(0.0)


# =========================
# Core sanity-check computations
# =========================
def compute_tech_sanity_table(
    meta: pd.DataFrame,          # GenID, Technology, GenType, InvCost, TotVarCost
    gen1: pd.DataFrame,          # GenID,time,generation
    gen2: pd.DataFrame,          # GenID,time,generation
    p1: pd.DataFrame,            # time,price_1st
    p2: pd.DataFrame,            # time,price_2nd
    res_rate: float,             # CHF/MWh (scalar)
) -> pd.DataFrame:
    # Attach metadata
    g1 = gen1.merge(meta, on="GenID", how="inner").merge(p1, on="time", how="left")
    g2 = gen2.merge(meta, on="GenID", how="inner").merge(p2, on="time", how="left")
    g1["price_1st"] = g1["price_1st"].fillna(0.0)
    g2["price_2nd"] = g2["price_2nd"].fillna(0.0)

    # Basic quantities
    g1["Rev1"] = g1["price_1st"] * g1["generation"]
    g2["Rev2"] = g2["price_2nd"] * g2["generation"]

    g1["VarCost1"] = g1["TotVarCost"] * g1["generation"]
    g2["VarCost2"] = g2["TotVarCost"] * g2["generation"]

    # Subsidy only on 1st stage, RES gens only
    is_res_1 = g1["GenType"].astype(str).str.upper().eq("RES")
    g1["Sub1"] = 0.0
    if res_rate != 0.0:
        g1.loc[is_res_1, "Sub1"] = float(res_rate) * g1.loc[is_res_1, "generation"]

    # Revenue-based missing money (your current approach)
    # MM_rev = Rev2 + Sub1 - Rev1
    mm_rev = (
        g1.groupby("Technology", as_index=False)[["Rev1", "Sub1", "VarCost1", "generation"]].sum()
        .rename(columns={"generation": "Gen1"})
        .merge(
            g2.groupby("Technology", as_index=False)[["Rev2", "VarCost2", "generation"]].sum().rename(
                columns={"generation": "Gen2"}
            ),
            on="Technology",
            how="outer",
        )
        .fillna(0.0)
    )
    mm_rev["MM_rev_based"] = mm_rev["Rev2"] + mm_rev["Sub1"] - mm_rev["Rev1"]

    # Margin-based missing money (matches your derived expression)
    # MM_margin = (Rev2 - VarCost2) - (Rev1 + Sub1 - VarCost1)
    #          = (p2 - tv)*Gen2 - (p1 + res_rate(res) - tv)*Gen1
    mm_rev["MM_margin_based"] = (mm_rev["Rev2"] - mm_rev["VarCost2"] + mm_rev["Sub1"]) -  (mm_rev["Rev1"] - mm_rev["VarCost1"])

    # Investment cost aggregation (technology-level sum of InvCost over generators)
    inv_by_tech = meta.groupby("Technology", as_index=False)[["InvCost"]].sum()

    out = mm_rev.merge(inv_by_tech, on="Technology", how="left").fillna(0.0)

    # Optional profit views including InvCost
    # Interpretations (sanity bounds):
    # - If you "charge" InvCost in each stage: Profit2 - Profit1 includes -2*InvCost
    # - If you "charge" InvCost once (annualized): Profit2 - Profit1 includes -InvCost
    out["ProfitDiff_inv_twice"] = (out["Rev2"] - (out["VarCost2"] + out["InvCost"])) - (
        (out["Rev1"] + out["Sub1"]) - (out["VarCost1"] + out["InvCost"])
    )
    out["ProfitDiff_inv_once"] = (out["Rev2"] - out["VarCost2"]) - ((out["Rev1"] + out["Sub1"]) - out["VarCost1"]) - out["InvCost"]

    # Diagnostics / comparisons
    out["Delta_margin_vs_rev"] = out["MM_margin_based"] - out["MM_rev_based"]
    out["VarCost1_minus_VarCost2"] = out["VarCost1"] - out["VarCost2"]

    # Ratios (avoid div-by-zero)
    denom = (out["InvCost"] + out["VarCost2"]).replace(0.0, pd.NA)
    out["MM_rev_over_(Inv+Var2)"] = (out["MM_rev_based"] / denom).fillna(0.0)
    out["MM_margin_over_(Inv+Var2)"] = (out["MM_margin_based"] / denom).fillna(0.0)

    # Order + select columns
    out = force_tech_order(out)

    out = out[
        [
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
            "ProfitDiff_inv_once",
            "ProfitDiff_inv_twice",
            "Delta_margin_vs_rev",
            "VarCost1_minus_VarCost2",
            "MM_rev_over_(Inv+Var2)",
            "MM_margin_over_(Inv+Var2)",
        ]
    ]
    return out


# =========================
# Main
# =========================
def main():
    results_root = BASE_DIR / "Results"
    results_dir = results_root / f"{OUT_NAME}_{VERSION}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # required inputs (RES_DUAL optional)
    for p in [
        DATA_GENERATORS,
        ELPRICE_CH,
        NODAL_DUAL,
        GENERATION_WIDE_2ND,
        GENERATION_LONG_1ST,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    meta = load_data_generators_ch_meta_costs(DATA_GENERATORS, COUNTRY_FILTER)

    gen_ids_all = meta["GenID"].astype(int).tolist()

    gen1 = load_generation_long_3cols(GENERATION_LONG_1ST)
    gen1 = gen1[gen1["GenID"].isin(set(gen_ids_all))].copy()

    gen2 = load_generation_wide_selected(GENERATION_WIDE_2ND, GEN_SHEET_2ND, gen_ids_all)

    p1 = load_price_dual_1st_two_cols(NODAL_DUAL)
    p2 = load_price_elprice_ch00(ELPRICE_CH)

    res_rate = load_res_dual_rate_optional(RES_DUAL)

    table = compute_tech_sanity_table(meta=meta, gen1=gen1, gen2=gen2, p1=p1, p2=p2, res_rate=res_rate)

    out_csv = results_dir / "missing_money_w_cost.csv"
    table.to_csv(out_csv, index=False)

    # quick consistency printouts
    print("Saved:", out_csv)
    print("RESCon (CHF/MWh):", res_rate)
    print(table)

    # Optional: also save a totals row
    totals = table.drop(columns=["Technology"]).sum(numeric_only=True).to_frame().T
    totals.insert(0, "Technology", "TOTAL")
    out_tot = pd.concat([table, totals], ignore_index=True)
    out_tot_csv = results_dir / "missing_money_w_cost_with_total.csv"
    out_tot.to_csv(out_tot_csv, index=False)
    print("Saved:", out_tot_csv)


if __name__ == "__main__":
    main()