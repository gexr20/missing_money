"""
Purpose:
Evaluate the effect of 1st-stage RES subsidies on missing money by comparing
technology revenues between the 1st and 2nd market stages. :contentReference[oaicite:0]{index=0}

Input data:
- Data_generators.csv: first column = GenID; reads Country, Technology, and GenType.
  Only CH generators are used and results are aggregated by Technology.

Generation data:
- 1st stage generation: InvestmentRun_2050/PowerGenerated.csv
  Long format (GenID, time, generation) for all generators.

- 2nd stage generation: GenerationPerGen_hourly_ALL_LP.xlsx (sheet "Generation_MWh").
  Wide table format: row 0 from column B contains GenID, and from row 3 column A contains time.
- 1st-stage RES generation profile is derived from PowerGenerated.csv
  by filtering generators with GenType == RES.

Price data:
- 1st stage price: NodalConstraint_one_CH_dual.csv
  Uses the first two columns (time, price_1st).
- 2nd stage price: ElPrice_hourly_CH.xlsx
  Column A = time, column B = CH00 price.

Subsidy:
- RES subsidy rate read from InvestmentRun_2050/RESCon_dual.csv (cell B2).
- Applied only to RES generators in the 1st stage.
- No subsidy is applied in the 2nd stage.

Missing money calculation:
- Missing money without subsidy = Revenue 2nd − Revenue 1st
- Missing money with subsidy = Revenue 2nd + Subsidy 1st − Revenue 1st

Output files:
- missing_money_table_by_technology.csv
- annual_generation_by_technology.csv
"""

from pathlib import Path
import pandas as pd

# =========================
# CONFIG
# =========================
OUT_NAME = "2stages_missing_money"
VERSION = "1st_sub_only"
COUNTRY_FILTER = "CH"

BASE_DIR = Path(__file__).resolve().parent / "tight_run_nuclear_all_eu_evflex_8k_pathway_2050_DE_WRT_NF_SN"

DATA_GENERATORS = BASE_DIR / "CentIv_2050" / "mappings" / "Data_generators.csv"

# Prices
ELPRICE_CH = BASE_DIR / "CentIv_2050" / "ElPrice_hourly_CH.xlsx"
NODAL_DUAL = BASE_DIR / "InvestmentRun_2050" / "NodalConstraint_one_CH_dual.csv"

# Generation
GENERATION_WIDE_2ND = BASE_DIR / "CentIv_2050" / "GenerationPerGen_hourly_ALL_LP.xlsx"
GEN_SHEET_2ND = "Generation_MWh"

GENERATION_LONG_1ST = BASE_DIR / "InvestmentRun_2050" / "PowerGenerated.csv"          # ALL gens: GenID,time,generation
GENERATION_RES_LONG_1ST = BASE_DIR / "InvestmentRun_2050" / "PowerProductionRES.csv"  # RES gens only: GenID,time,generation (for 1st-stage subsidy)

# Optional subsidy dual (CHF/MWh, scalar)
RES_DUAL = BASE_DIR / "InvestmentRun_2050" / "RESCon_dual.csv"

TIME_MIN = 0
TIME_MAX = 8759

# We keep the mapping only for validation (no aggregation now).
TECH_BUCKET_MAP = {
    "PV-alpine": "PV",
    "PV-roof": "PV",
    "WindOn": "WindOn",
    "Waste": "Waste",
    "GasCC-CCS": "GasCC",
    "GasCC-Syn": "GasCC",
    "Battery-TSO": "Ignore",
    "DAC": "Ignore",
    "Dam": "Ignore",
    "FuelCell": "Ignore",
    "Nuclear": "Ignore",
    "Pump-Open": "Ignore",
    "RoR": "Ignore",
}

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
# Helpers
# =========================
def load_data_generators_ch_meta(path: Path, country_code: str = "CH") -> pd.DataFrame:
    """
    Need columns:
    - first column: GenID
    - Country
    - Technology
    - GenType (for RES identification)
    Return: GenID(int), Technology(str), GenType(str) for CH only
    """
    df = pd.read_csv(path)
    genid_col = df.columns[0]

    for c in ["Country", "Technology", "GenType"]:
        if c not in df.columns:
            raise ValueError(f"Data_generators.csv missing required column: {c}")

    out = df[[genid_col, "Country", "Technology", "GenType"]].copy().rename(columns={genid_col: "GenID"})
    out["GenID"] = pd.to_numeric(out["GenID"], errors="coerce")
    out["Country"] = out["Country"].astype(str).str.upper()
    out["Technology"] = out["Technology"].astype(str)
    out["GenType"] = out["GenType"].astype(str)

    out = out.dropna(subset=["GenID"]).copy()
    out["GenID"] = out["GenID"].astype(int)

    out = out.loc[out["Country"] == str(country_code).upper()].copy()
    out = out.drop_duplicates(subset=["GenID"]).reset_index(drop=True)

    if out.empty:
        raise ValueError(f"No generators found with Country == {country_code}")
    return out


def load_price_elprice_ch00(path: Path) -> pd.DataFrame:
    """
    ElPrice_hourly_CH.xlsx:
    - col A: time from row 2
    - col B: CH00 price from row 2
    Return: time(int), price_2nd(float)
    """
    df0 = pd.read_excel(path, header=None, engine="openpyxl")
    time = pd.to_numeric(df0.iloc[1:, 0], errors="coerce")
    price = pd.to_numeric(df0.iloc[1:, 1], errors="coerce")
    out = pd.DataFrame({"time": time, "price_2nd": price}).dropna(subset=["time", "price_2nd"]).copy()
    out["time"] = out["time"].astype(int)
    out["price_2nd"] = out["price_2nd"].astype(float)
    out = out[(out["time"] >= TIME_MIN) & (out["time"] <= TIME_MAX)].reset_index(drop=True)
    return out


def load_price_dual_1st_two_cols(path: Path) -> pd.DataFrame:
    """
    1st-stage price: take first two columns as (time, price_1st)
    Return: time(int), price_1st(float)
    """
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
    """
    Long generation csv:
      col1=GenID, col2=time, col3=generation
    Return: GenID(int), time(int), generation(float)
    """
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
    """
    Wide generation excel:
    - row0, colB..: GenID headers
    - row3.., colA: time
    Return long: GenID,time,generation for selected gen_ids.
    """
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
    """
    RESCon_dual.csv (CHF/MWh) optional:
    - missing -> 0.0
    - else read B2 (row2,col2) as scalar
    """
    if not path.exists():
        return 0.0
    df = pd.read_csv(path, header=None)
    if df.shape[0] < 2 or df.shape[1] < 2:
        raise ValueError("RESCon_dual.csv format not recognized (need >=2 rows and >=2 cols).")
    val = pd.to_numeric(df.iloc[1, 1], errors="coerce")
    if pd.isna(val):
        raise ValueError("RESCon_dual.csv: cannot parse scalar at B2 (row2,col2).")
    return float(val)


def revenue_by_technology(
    gen_ts: pd.DataFrame,
    price: pd.DataFrame,
    price_col: str,
    meta_tech: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return: Technology, revenue
    """
    df = gen_ts.merge(meta_tech[["GenID", "Technology"]], on="GenID", how="inner")
    df = df.merge(price[["time", price_col]], on="time", how="left")
    df[price_col] = df[price_col].fillna(0.0)
    df["revenue"] = df[price_col] * df["generation"]
    return df.groupby("Technology", as_index=False)[["revenue"]].sum()


def subsidy_by_technology_from_gen(
    res_rate: float,
    gen_ts: pd.DataFrame,
    res_gen_ids: set[int],
    meta_tech: pd.DataFrame,
) -> pd.DataFrame:
    """
    Subsidy total (CHF) by Technology for RES GenIDs only:
      res_rate * generation
    Return: Technology, subsidy
    """
    if res_rate == 0.0:
        return pd.DataFrame(columns=["Technology", "subsidy"])

    df = gen_ts[gen_ts["GenID"].isin(res_gen_ids)].merge(
        meta_tech[["GenID", "Technology"]], on="GenID", how="inner"
    )
    df["subsidy"] = float(res_rate) * df["generation"]
    return df.groupby("Technology", as_index=False)[["subsidy"]].sum()


def build_missing_money_table(
    gen_1st_all: pd.DataFrame,
    gen_2nd_all: pd.DataFrame,
    gen_1st_res_profile: pd.DataFrame,
    price_1st: pd.DataFrame,
    price_2nd: pd.DataFrame,
    meta_tech: pd.DataFrame,  # must contain GenID, Technology, GenType
    res_gen_ids: set[int],
    res_rate: float,
) -> pd.DataFrame:
    # Revenue by Technology
    r1 = revenue_by_technology(gen_1st_all, price_1st, "price_1st", meta_tech).rename(
        columns={"revenue": "Revenue 1st"}
    )
    r2 = revenue_by_technology(gen_2nd_all, price_2nd, "price_2nd", meta_tech).rename(
        columns={"revenue": "Revenue 2nd"}
    )
    out = r1.merge(r2, on="Technology", how="outer").fillna(0.0)

    # Subsidy by Technology (RES only) — ONLY 1st stage (RESCon_dual exists only in 1st stage)
    s1 = subsidy_by_technology_from_gen(res_rate, gen_1st_res_profile, res_gen_ids, meta_tech).rename(
        columns={"subsidy": "Subsidy 1st"}
    )
    out = out.merge(s1, on="Technology", how="left")
    out["Subsidy 1st"] = out["Subsidy 1st"].fillna(0.0)

    # 2nd stage subsidy not activated
    out["Subsidy 2nd"] = 0.0

    # Missing money
    # missing money without subsidy = Revenue 2nd - Revenue 1st
    # missing money with subsidy = Revenue 2nd + Subsidy 1st - Revenue 1st
    out["Missing money without"] = out["Revenue 2nd"] - out["Revenue 1st"]
    out["Missing money with subsidy"] = out["Revenue 2nd"] + out["Subsidy 1st"] - out["Revenue 1st"]

    # Ensure all required Technology rows exist (even if 0)
    full = pd.DataFrame({"Technology": TECH_ORDER})
    out = full.merge(out, on="Technology", how="left").fillna(0.0)

    out = out[
        [
            "Technology",
            "Revenue 1st",
            "Revenue 2nd",
            "Subsidy 1st",
            "Subsidy 2nd",
            "Missing money with subsidy",
            "Missing money without",
        ]
    ]
    return out


def build_annual_generation_by_technology(
    gen_1st_all: pd.DataFrame,
    gen_2nd_all: pd.DataFrame,
    df_meta_ch: pd.DataFrame,
) -> pd.DataFrame:
    """
    Annual generation per Technology (no aggregation), and force TECH_ORDER rows.
    """
    g1 = gen_1st_all.merge(df_meta_ch[["GenID", "Technology"]], on="GenID", how="left")
    g1_agg = (
        g1.groupby("Technology", as_index=False)[["generation"]]
        .sum()
        .rename(columns={"generation": "Generation 1st"})
    )

    g2 = gen_2nd_all.merge(df_meta_ch[["GenID", "Technology"]], on="GenID", how="left")
    g2_agg = (
        g2.groupby("Technology", as_index=False)[["generation"]]
        .sum()
        .rename(columns={"generation": "Generation 2nd"})
    )

    out = g1_agg.merge(g2_agg, on="Technology", how="outer").fillna(0.0)

    # force all rows
    full = pd.DataFrame({"Technology": TECH_ORDER})
    out = full.merge(out, on="Technology", how="left").fillna(0.0)

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
        GENERATION_RES_LONG_1ST,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    # Meta (CH only)
    df_meta_ch = load_data_generators_ch_meta(DATA_GENERATORS, COUNTRY_FILTER)

    # Optional: validate that all Technology values are covered by TECH_BUCKET_MAP
    # (does not affect calculation; helps avoid silent missing categories)
    tmp = df_meta_ch.copy()
    tmp["__chk__"] = tmp["Technology"].map(TECH_BUCKET_MAP)
    if tmp["__chk__"].isna().any():
        missing = tmp.loc[tmp["__chk__"].isna(), "Technology"].unique().tolist()
        raise RuntimeError(f"Technology not defined in TECH_BUCKET_MAP (extend it): {missing}")

    # For per-Technology output, keep all CH generators (do NOT drop Ignore)
    meta_tech = df_meta_ch[["GenID", "Technology", "GenType"]].copy()

    # RES GenIDs strictly by GenType == RES (includes Waste if your meta says so)
    res_gen_ids = set(
        meta_tech.loc[meta_tech["GenType"].astype(str).str.upper() == "RES", "GenID"].astype(int).tolist()
    )

    gen_ids_all_ch = meta_tech["GenID"].astype(int).tolist()

    # Generation
    gen_1st_all = load_generation_long_3cols(GENERATION_LONG_1ST)
    gen_1st_all = gen_1st_all[gen_1st_all["GenID"].isin(set(gen_ids_all_ch))].copy()

    gen_2nd_all = load_generation_wide_selected(GENERATION_WIDE_2ND, GEN_SHEET_2ND, gen_ids_all_ch)

    # 1st-stage RES profile for subsidy (keep only CH RES gen ids)
    # gen_1st_res_profile = load_generation_long_3cols(GENERATION_RES_LONG_1ST)
    # gen_1st_res_profile = gen_1st_res_profile[gen_1st_res_profile["GenID"].isin(res_gen_ids)].copy()

    # 1st-stage RES profile for subsidy: derive from PowerGenerated.csv (gen_1st_all)
    gen_1st_res_profile = gen_1st_all[gen_1st_all["GenID"].isin(res_gen_ids)].copy()

    # Prices
    price_1st = load_price_dual_1st_two_cols(NODAL_DUAL)
    price_2nd = load_price_elprice_ch00(ELPRICE_CH)

    # Subsidy rate (CHF/MWh)
    res_rate = load_res_dual_rate_optional(RES_DUAL)

    # Missing money table by Technology
    table = build_missing_money_table(
        gen_1st_all=gen_1st_all,
        gen_2nd_all=gen_2nd_all,
        gen_1st_res_profile=gen_1st_res_profile,
        price_1st=price_1st,
        price_2nd=price_2nd,
        meta_tech=meta_tech,
        res_gen_ids=res_gen_ids,
        res_rate=res_rate,
    )
    out_csv = results_dir / "missing_money_table_by_technology.csv"
    table.to_csv(out_csv, index=False)

    # Annual generation check table by Technology
    annual_gen_table = build_annual_generation_by_technology(
        gen_1st_all=gen_1st_all,
        gen_2nd_all=gen_2nd_all,
        df_meta_ch=df_meta_ch,
    )
    annual_gen_csv = results_dir / "annual_generation_by_technology.csv"
    annual_gen_table.to_csv(annual_gen_csv, index=False)

    print("Saved:", out_csv)
    print("Saved:", annual_gen_csv)
    print("RES subsidy rate (CHF/MWh):", res_rate)
    print(table)


if __name__ == "__main__":
    main()