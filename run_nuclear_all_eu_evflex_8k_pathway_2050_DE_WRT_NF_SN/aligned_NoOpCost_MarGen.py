"""
Purpose:
Identify one marginal generator per hour for CH generators under both 1st-stage
and 2nd-stage prices, without using opportunity cost.

Input data:
- Data_generators.csv: reads CH generators with TotVarCost and Pmax.
- GenerationPerGen_hourly_ALL_LP.xlsx: 2nd-stage hourly generation.
- NodalConstraint_one_CH_dual.csv: 1st-stage price.
- ElPrice_hourly_CH.xlsx: 2nd-stage price.

Processing:
- Keep only CH generators.
- Apply a minimum generation threshold to define marginal-eligible units.
- For each hour, select the generator with the smallest cost-price gap among
  eligible online units; if none exists, fall back to the overall closest unit.
- Also mark whether the selected unit is saturated and within the ±10% price band.

Output:
- CH generator metadata and generation time series.
- 1st-stage and 2nd-stage price tables.
- Marginal generator results and hourly summary tables for both stages.
- A sanity-check CSV.

Output folder:
Results/<script_name>_<VERSION>/.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
VERSION = "v1"
COUNTRY_FILTER = "CH"

BASE_DIR = Path(__file__).resolve().parent / "run_nuclear_all_eu_evflex_8k_pathway_2050_DE_WRT_NF_SN"

DATA_GENERATORS = BASE_DIR / "CentIv_2050" / "mappings" / "Data_generators.csv"

# 2nd stage price (aggregated CH bus CH00)
ELPRICE_CH = BASE_DIR / "CentIv_2050" / "ElPrice_hourly_CH.xlsx"

# 1st stage price (dual of nodal constraint), need /100 (NOTE: this script keeps your original loading as-is)
NODAL_DUAL = BASE_DIR / "InvestmentRun_2050" / "NodalConstraint_one_CH_dual.csv"

GENERATION_WIDE = BASE_DIR / "CentIv_2050" / "GenerationPerGen_hourly_ALL_LP.xlsx"
GEN_SHEET = "Generation_MWh"

# -------- unified boundaries (match opp-cost case) --------
# marginal eligibility filter (avoid unconverged points)
GEN_MIN_MWH = 1.0
GEN_MIN_PCT_PMAX = 0.02

# numeric (for stable comparisons / ties)
MC_LE_PRICE_EPS = 1e-9
TIE_ABS_EPS = 1e-7

# keep your original band/sat meaning
PRICE_BAND = 0.10        # TotVarCost ∈ price * [0.9, 1.1]
SAT_EPS = 1e-4           # generation >= (1-SAT_EPS)*Pmax 视为饱和


# =========================
# Helpers
# =========================
def _get_col_by_excel_letter(df: pd.DataFrame, letter: str) -> str:
    idx = ord(letter.upper()) - ord("A")
    if idx < 0 or idx >= len(df.columns):
        raise ValueError(f"Column letter {letter} out of range. ncols={len(df.columns)}")
    return df.columns[idx]


def load_data_generators_ch(path: Path, country_code: str = "CH") -> pd.DataFrame:
    """
    从 Data_generators.csv 读出并筛选 CH generators：
    - GenID：第一列
    - Country：列名必须存在（用于筛 CH）
    - TotVarCost：优先列名 TotVarCost，否则用 S 列
    - Pmax：优先列名 Pmax，否则用 T 列（按你之前描述）
    返回：GenID(int), Country(str), TotVarCost(float), Pmax(float)
    """
    df = pd.read_csv(path)

    genid_col = df.columns[0]
    if "Country" not in df.columns:
        raise ValueError("Data_generators.csv missing required column: Country")

    if "TotVarCost" in df.columns:
        totvar_col = "TotVarCost"
    else:
        totvar_col = _get_col_by_excel_letter(df, "S")

    if "Pmax" in df.columns:
        pmax_col = "Pmax"
    else:
        pmax_col = _get_col_by_excel_letter(df, "T")

    out = df[[genid_col, "Country", totvar_col, pmax_col]].copy()
    out = out.rename(columns={
        genid_col: "GenID",
        totvar_col: "TotVarCost",
        pmax_col: "Pmax",
    })

    out["GenID"] = pd.to_numeric(out["GenID"], errors="coerce")
    out["TotVarCost"] = pd.to_numeric(out["TotVarCost"], errors="coerce")
    out["Pmax"] = pd.to_numeric(out["Pmax"], errors="coerce")
    out["Country"] = out["Country"].astype(str)

    out = out.dropna(subset=["GenID", "TotVarCost", "Pmax", "Country"]).copy()
    out["GenID"] = out["GenID"].astype(int)

    out = out.loc[out["Country"] == country_code].copy()
    out = out.drop_duplicates(subset=["GenID"]).reset_index(drop=True)
    return out


def load_price_elprice_ch00(path: Path) -> pd.DataFrame:
    """
    读取 CentIv_2050/ElPrice_hourly_CH.xlsx（截图结构）：
    - A列：time (0..8759)，从第2行开始
    - B1：'CH00'
    - B列：price，从第2行开始
    返回：time(int), price(float) 其中列名为 price_2nd
    """
    df0 = pd.read_excel(path, header=None, engine="openpyxl")

    time = pd.to_numeric(df0.iloc[1:, 0], errors="coerce")
    price = pd.to_numeric(df0.iloc[1:, 1], errors="coerce")

    out = pd.DataFrame({"time": time, "price_2nd": price}).dropna(subset=["time", "price_2nd"]).copy()
    out["time"] = out["time"].astype(int)
    return out


def load_price_dual_1st(path: Path) -> pd.DataFrame:
    """
    读取 1st-stage dual price:
    col0 = time
    col1 = dual value
    返回: time(int), price_1st(float)
    """
    df = pd.read_csv(path)

    out = df.iloc[:, :2].copy()
    out.columns = ["time", "price_1st"]

    out["time"] = pd.to_numeric(out["time"], errors="coerce").astype(int)
    out["price_1st"] = pd.to_numeric(out["price_1st"], errors="coerce")

    return out.dropna(subset=["time", "price_1st"]).reset_index(drop=True)


def load_generation_wide_selected(path: Path, sheet: str, gen_ids: list[int]) -> pd.DataFrame:
    """
    解析 GenerationPerGen_hourly_ALL_LP.xlsx 的宽表结构：
    - 第0行(B列开始)是 GenID
    - 第3行开始(A列)是 time=0..8759
    只抽取 gen_ids 对应的列，返回长表：GenID, time, generation
    """
    df0 = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")

    gen_row = pd.to_numeric(df0.iloc[0, 1:], errors="coerce")  # row0, from colB
    col_map = {}
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
        vals = pd.to_numeric(df0.iloc[3:3 + n_t, j], errors="coerce").fillna(0.0).values
        blocks.append(pd.DataFrame({"GenID": int(g), "time": time_series, "generation": vals}))

    if not blocks:
        return pd.DataFrame(columns=["GenID", "time", "generation"])

    return pd.concat(blocks, ignore_index=True)


# -------- unified eligibility helpers --------
def _min_gen_threshold(pmax: float) -> float:
    if pmax is None or not np.isfinite(pmax) or pmax <= 0:
        return np.inf
    return float(max(GEN_MIN_MWH, GEN_MIN_PCT_PMAX * pmax))


def is_marginal_eligible(gen_mwh: float, pmax: float) -> bool:
    if gen_mwh is None or not np.isfinite(gen_mwh):
        return False
    return (gen_mwh > 0.0) and (gen_mwh >= _min_gen_threshold(pmax))


def pick_marginal_per_hour(
    price_df: pd.DataFrame,
    gens_df: pd.DataFrame,
    gen_ts: pd.DataFrame,
    price_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    输出保持你原来的结构：
    - 每小时一台机组（优先 eligible&on；否则 fallback 到全体最小 abs_gap）
    """
    df = price_df[["time", price_col]].merge(gens_df, how="cross")
    df = df.merge(gen_ts, on=["GenID", "time"], how="left")
    df["generation"] = df["generation"].fillna(0.0)

    # ---- UPDATED: unified eligibility (replaces GEN_POS_EPS) ----
    df["is_on"] = df["generation"] > 0.0
    df["eligible_gen_level"] = df.apply(lambda r: is_marginal_eligible(r["generation"], r["Pmax"]), axis=1)
    df["is_saturated"] = df["generation"] >= (1 - SAT_EPS) * df["Pmax"]

    df["gap"] = df["TotVarCost"] - df[price_col]
    df["abs_gap"] = df["gap"].abs()

    lower = df[price_col] * (1 - PRICE_BAND)
    upper = df[price_col] * (1 + PRICE_BAND)
    df["in_band_10pct"] = (df["TotVarCost"] >= (lower - TIE_ABS_EPS)) & (df["TotVarCost"] <= (upper + TIE_ABS_EPS))

    # candidates: on & eligible (same spirit as opp-cost sparse points)
    df_on = df.loc[df["is_on"] & df["eligible_gen_level"]].copy()

    if df_on.empty:
        idx = df.groupby("time")["abs_gap"].idxmin()
        marginal = df.loc[idx].copy()
        pick_note = "no_eligible_on_generator_any_hour"
    else:
        idx_on = df_on.groupby("time")["abs_gap"].idxmin()
        m_on = df_on.loc[idx_on].copy()

        covered = set(m_on["time"].tolist())
        df_rest = df.loc[~df["time"].isin(covered)].copy()
        if not df_rest.empty:
            idx_rest = df_rest.groupby("time")["abs_gap"].idxmin()
            m_rest = df_rest.loc[idx_rest].copy()
            marginal = pd.concat([m_on, m_rest], ignore_index=True)
        else:
            marginal = m_on
        pick_note = "prefer_is_on_and_eligible"

    marginal = marginal.sort_values("time").reset_index(drop=True)

    hourly_summary = marginal[[
        "time", price_col, "GenID", "Country", "TotVarCost", "Pmax",
        "generation", "is_on", "eligible_gen_level", "is_saturated",
        "gap", "abs_gap", "in_band_10pct"
    ]].copy()

    return marginal, hourly_summary, pick_note


# =========================
# Main
# =========================
def main():
    results_root = BASE_DIR / "Results"
    script_name = Path(__file__).stem
    results_dir = results_root / f"{script_name}_{VERSION}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) CH generators only (NOT by bus)
    gens_ch = load_data_generators_ch(DATA_GENERATORS, COUNTRY_FILTER)  # Country == CH
    gens_ch.to_csv(results_dir / "gens_CH.csv", index=False)

    gen_ids = gens_ch["GenID"].tolist()
    if not gen_ids:
        raise ValueError("No CH generators found (Country==CH).")

    # 2) generation time series for CH gens
    gen_ts = load_generation_wide_selected(GENERATION_WIDE, GEN_SHEET, gen_ids)
    gen_ts.to_csv(results_dir / "generation_CH_long.csv", index=False)

    # 3) prices
    price_1st = load_price_dual_1st(NODAL_DUAL)        # time, price_1st
    price_2nd = load_price_elprice_ch00(ELPRICE_CH)    # time, price_2nd

    price_1st.to_csv(results_dir / "price_1st_dual.csv", index=False)
    price_2nd.to_csv(results_dir / "price_2nd_CH00.csv", index=False)

    # 4) marginal under 1st-stage price
    m1, s1, note1 = pick_marginal_per_hour(
        price_df=price_1st,
        gens_df=gens_ch[["GenID", "Country", "TotVarCost", "Pmax"]],
        gen_ts=gen_ts,
        price_col="price_1st",
    )
    m1.to_csv(results_dir / "marginal_1st_stage.csv", index=False)
    s1.to_csv(results_dir / "hourly_summary_1st_stage.csv", index=False)

    # 5) marginal under 2nd-stage price
    m2, s2, note2 = pick_marginal_per_hour(
        price_df=price_2nd,
        gens_df=gens_ch[["GenID", "Country", "TotVarCost", "Pmax"]],
        gen_ts=gen_ts,
        price_col="price_2nd",
    )
    m2.to_csv(results_dir / "marginal_2nd_stage.csv", index=False)
    s2.to_csv(results_dir / "hourly_summary_2nd_stage.csv", index=False)

    # optional sanity checks
    pd.DataFrame({
        "item": ["n_CH_gens", "n_hours_1st", "n_hours_2nd", "pick_rule_1st", "pick_rule_2nd"],
        "value": [len(gen_ids), price_1st["time"].nunique(), price_2nd["time"].nunique(), note1, note2],
    }).to_csv(results_dir / "sanity_check.csv", index=False)

    print("Saved to:", results_dir)


if __name__ == "__main__":
    main()