"""
Purpose
Efficiently compute marginal generators and opportunity-cost adjusted
marginal costs for the Swiss system using sparse generation points.

Core logic
1. Filter generators to Country == CH.
2. Load hourly generation and keep only meaningful operating points:
       generation ≥ 1 MWh AND generation ≥ 2% of Pmax.
3. Compute effective marginal cost:
       EffectiveMC = TotVarCost + (− dual_sum)
   where dual_sum aggregates battery and hydro SOC dual values.
4. Identify marginal generators for each hour:
       among units with EffectiveMC ≤ price,
       choose the unit(s) with the highest EffectiveMC.
5. Perform the calculation for both stages:
       stage1 → InvestmentRun_2050 dual files
       stage2 → CentIv_2050 dual files.

Outputs
- gens_CH_meta.csv: CH generator metadata
- gen_CH_long_raw.csv: CH generation time series
- gen_CH_sparse_points.csv: filtered operating points
- price_1st.csv, price_2nd.csv: hourly prices
- stage1/ and stage2/ folders:
    - dual_sum_CH.csv
    - points_effective_mc_sparse.csv
    - marginals_per_hour_multi.csv
    - hourly_diagnostics.csv
- price1st_battcand.png / price1st_battcand.csv:
    comparison of 1st-stage price and battery candidate SOC dual.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "v1"
COUNTRY_FILTER = "CH"

# -------- paths (adjust BASE_DIR) --------
BASE_DIR = Path(__file__).resolve().parent / "run_nuclear_all_eu_evflex_8k_pathway_2050_DE_NRT_NF_SN"

DATA_GENERATORS = BASE_DIR / "CentIv_2050" / "mappings" / "Data_generators.csv"
GENERATION_WIDE = BASE_DIR / "CentIv_2050" / "GenerationPerGen_hourly_ALL_LP.xlsx"
GEN_SHEET = "Generation_MWh"

# prices
ELPRICE_CH = BASE_DIR / "CentIv_2050" / "ElPrice_hourly_CH.xlsx"  # 2nd stage (CH00)
NODAL_DUAL = BASE_DIR / "InvestmentRun_2050" / "NodalConstraint_one_CH_dual.csv"  # 1st stage
NODAL_DUAL_SCALE = 100.0 / 100.0  # adjust if needed

# dual files exist under both InvestmentRun_2050 and CentIv_2050
BATT_DUAL_FILES = ["SoCBattCandCon_dual.csv", "SoCBattCon_dual.csv"]
HYDRO_DUAL_FILES = ["SoCHydroCon1_dual.csv", "SoCHydroDayCon1_dual.csv", "SoCHydroDamCon_dual.csv"]

# dual -> opportunity cost (per your rule: TotVarCost + negative*dual)
DUAL_TO_OPPCOST_SIGN = -1.0
DUAL_SCALE = 1.0

# marginal eligibility filter (avoid unconverged points)
GEN_MIN_MWH = 1.0
GEN_MIN_PCT_PMAX = 0.02

# numeric
MC_LE_PRICE_EPS = 1e-9
TIE_ABS_EPS = 1e-4


# =========================
# IO helpers
# =========================
def _get_col_by_excel_letter(df: pd.DataFrame, letter: str) -> str:
    idx = ord(letter.upper()) - ord("A")
    if idx < 0 or idx >= len(df.columns):
        raise ValueError(f"Column letter {letter} out of range. ncols={len(df.columns)}")
    return df.columns[idx]


def load_ch_generators_meta(path: Path, country_code: str = "CH") -> pd.DataFrame:
    """Data_generators.csv -> CH generators meta: GenID, Country, TotVarCost, Pmax"""
    df = pd.read_csv(path)

    genid_col = df.columns[0]
    if "Country" not in df.columns:
        raise ValueError("Data_generators.csv missing required column: Country")

    totvar_col = "TotVarCost" if "TotVarCost" in df.columns else _get_col_by_excel_letter(df, "S")
    pmax_col = "Pmax" if "Pmax" in df.columns else _get_col_by_excel_letter(df, "T")

    out = df[[genid_col, "Country", totvar_col, pmax_col]].copy()
    out = out.rename(columns={genid_col: "GenID", totvar_col: "TotVarCost", pmax_col: "Pmax"})

    out["GenID"] = pd.to_numeric(out["GenID"], errors="coerce")
    out["TotVarCost"] = pd.to_numeric(out["TotVarCost"], errors="coerce")
    out["Pmax"] = pd.to_numeric(out["Pmax"], errors="coerce")
    out["Country"] = out["Country"].astype(str)

    out = out.dropna(subset=["GenID", "TotVarCost", "Pmax", "Country"]).copy()
    out["GenID"] = out["GenID"].astype(int)
    out = out.loc[out["Country"] == country_code].drop_duplicates(subset=["GenID"]).reset_index(drop=True)
    return out


def load_price_elprice_ch00(path: Path) -> pd.DataFrame:
    """2nd stage: ElPrice_hourly_CH.xlsx (no header). A:time, B:CH00 price from row 2."""
    df0 = pd.read_excel(path, header=None, engine="openpyxl")
    time = pd.to_numeric(df0.iloc[1:, 0], errors="coerce")
    price = pd.to_numeric(df0.iloc[1:, 1], errors="coerce")
    out = pd.DataFrame({"time": time, "price_2nd": price}).dropna(subset=["time", "price_2nd"]).copy()
    out["time"] = out["time"].astype(int)
    return out


def load_price_dual_1st(path: Path, scale: float = 1.0) -> pd.DataFrame:
    """1st stage: NodalConstraint_one_CH_dual.csv -> take first 2 cols: time, dual (scaled)."""
    df = pd.read_csv(path)
    out = df.iloc[:, :2].copy()
    out.columns = ["time", "price_1st"]
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["price_1st"] = pd.to_numeric(out["price_1st"], errors="coerce") * float(scale)
    out = out.dropna(subset=["time", "price_1st"]).copy()
    out["time"] = out["time"].astype(int)
    return out


def load_generation_wide_selected(path: Path, sheet: str, gen_ids: list[int]) -> pd.DataFrame:
    """Wide excel -> long for selected GenIDs.
    Format:
      - row0 (B..): GenID
      - row3.. (A): time
    Returns: GenID(int), time(int), generation(float)
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
        vals = pd.to_numeric(df0.iloc[3:3 + n_t, j], errors="coerce").fillna(0.0).values
        blocks.append(pd.DataFrame({"GenID": int(g), "time": time_series, "generation": vals}))

    if not blocks:
        return pd.DataFrame(columns=["GenID", "time", "generation"])

    return pd.concat(blocks, ignore_index=True)


def load_dual_long(path: Path) -> pd.DataFrame:
    """Dual csv structure: col0 GenID, col1 time, col2 dual (from row2)."""
    df = pd.read_csv(path)
    out = df.iloc[:, :3].copy()
    out.columns = ["GenID", "time", "dual"]
    out["GenID"] = pd.to_numeric(out["GenID"], errors="coerce")
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["dual"] = pd.to_numeric(out["dual"], errors="coerce")
    out = out.dropna(subset=["GenID", "time", "dual"]).copy()
    out["GenID"] = out["GenID"].astype(int)
    out["time"] = out["time"].astype(int)
    return out


def load_duals_sum(stage_dir: Path, filenames: list[str], ch_gen_set: set[int]) -> pd.DataFrame:
    """Load + filter to CH GenIDs, then sum dual by (GenID,time)."""
    parts = []
    missing = []
    for fn in filenames:
        p = stage_dir / fn
        if not p.exists():
            missing.append(str(p))
            continue
        d = load_dual_long(p)
        if not d.empty:
            d = d.loc[d["GenID"].isin(ch_gen_set)].copy()
            parts.append(d[["GenID", "time", "dual"]])

    if missing:
        raise FileNotFoundError("Missing dual files:\n" + "\n".join(missing))

    if not parts:
        return pd.DataFrame(columns=["GenID", "time", "dual"])

    df = pd.concat(parts, ignore_index=True)
    return df.groupby(["GenID", "time"], as_index=False)["dual"].sum()


# =========================
# Core logic (fast)
# =========================
def build_sparse_points_for_marginal(gen_ts: pd.DataFrame, gens_meta: pd.DataFrame) -> pd.DataFrame:
    """Keep only meaningful generation points and attach static meta (Pmax, TotVarCost)."""
    df = gen_ts.merge(gens_meta[["GenID", "Pmax", "TotVarCost", "Country"]], on="GenID", how="left")
    df["Pmax"] = pd.to_numeric(df["Pmax"], errors="coerce")
    df["TotVarCost"] = pd.to_numeric(df["TotVarCost"], errors="coerce")
    df = df.dropna(subset=["Pmax", "TotVarCost"]).copy()

    df["min_gen_threshold"] = np.maximum(GEN_MIN_MWH, GEN_MIN_PCT_PMAX * df["Pmax"])
    df["eligible_gen_level"] = df["generation"] >= df["min_gen_threshold"]
    df["is_on"] = df["generation"] > 0.0

    # only keep eligible & on points for marginal analysis (sparse)
    df = df.loc[df["is_on"] & df["eligible_gen_level"]].copy()
    return df


def attach_opportunity_cost(df_points: pd.DataFrame, dual_sum: pd.DataFrame) -> pd.DataFrame:
    """EffectiveMC = TotVarCost + (-dual_sum). Non-storage -> dual_sum=0."""
    df = df_points.merge(dual_sum, on=["GenID", "time"], how="left")
    df["dual"] = df["dual"].fillna(0.0)
    df["OppCost"] = DUAL_TO_OPPCOST_SIGN * DUAL_SCALE * df["dual"]
    df["EffectiveMC"] = df["TotVarCost"] + df["OppCost"]

    # optional label: treat any gen with nonzero/available dual rows as storage-like
    # (if a non-storage never appears in dual files, its dual stays 0)
    df["has_dual_row"] = df["dual"].ne(0.0)
    return df


def pick_marginals_multi(df: pd.DataFrame, price_df: pd.DataFrame, price_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized multi-marginal selection per time:
       among points with EffectiveMC <= price, pick those with max EffectiveMC (ties).
    """
    d = df.merge(price_df[["time", price_col]], on="time", how="inner").copy()
    d["mc_le_price"] = d["EffectiveMC"] <= (d[price_col] + MC_LE_PRICE_EPS)
    feasible = d.loc[d["mc_le_price"]].copy()

    if feasible.empty:
        return feasible, pd.DataFrame(columns=["time", "n_points", "n_feasible", "picked_max_effective_mc"])

    feasible["max_mc_below"] = feasible.groupby("time")["EffectiveMC"].transform("max")
    picked = feasible.loc[(feasible["EffectiveMC"] >= (feasible["max_mc_below"] - TIE_ABS_EPS))].copy()

    diag = (
        d.groupby("time", as_index=False)
        .agg(
            n_points=("GenID", "count"),
            n_feasible=("mc_le_price", "sum"),
        )
        .merge(
            picked.groupby("time", as_index=False)["EffectiveMC"].max().rename(columns={"EffectiveMC": "picked_max_effective_mc"}),
            on="time",
            how="left",
        )
    )
    return picked, diag


def plot_price_vs_battcand_dual(results_dir: Path, price_1st: pd.DataFrame, battcand_dual_1st: pd.DataFrame) -> None:
    """Plot 1st stage price and battery candidate dual (sum over CH IDs, SN typically 1)."""
    ddual = battcand_dual_1st.groupby("time", as_index=False)["dual"].sum().rename(columns={"dual": "batt_cand_dual"})

    dfp = price_1st.merge(ddual, on="time", how="left").fillna({"batt_cand_dual": 0.0})
    # Plot with negative of dual value to make it comparable to positiv price
    ddual["batt_cand_dual"] = -ddual["batt_cand_dual"]
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(dfp["time"], dfp["price_1st"], label="price_1st")
    ax1.set_xlabel("time")
    ax1.set_ylabel("price_1st")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(dfp["time"], dfp["batt_cand_dual"], label="batt_cand_dual")
    ax2.set_ylabel("battery candidate dual")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(results_dir / "price1st_battcand.png", dpi=200)
    plt.close(fig)

    dfp.to_csv(results_dir / "price1st_battcand.csv", index=False)


def run_stage(stage_name: str, stage_dir: Path, df_points_base: pd.DataFrame, price_df: pd.DataFrame, price_col: str,
              ch_gen_set: set[int], results_dir: Path) -> None:
    """Stage run on sparse points only."""
    # dual sum for this stage (battery + hydro, CH filtered while reading)
    dual_batt = load_duals_sum(stage_dir, BATT_DUAL_FILES, ch_gen_set)
    dual_hydro = load_duals_sum(stage_dir, HYDRO_DUAL_FILES, ch_gen_set)
    dual_sum = pd.concat([dual_batt, dual_hydro], ignore_index=True)
    if not dual_sum.empty:
        dual_sum = dual_sum.groupby(["GenID", "time"], as_index=False)["dual"].sum()
    else:
        dual_sum = pd.DataFrame(columns=["GenID", "time", "dual"])

    # attach opp cost only on sparse points
    df_points = attach_opportunity_cost(df_points_base, dual_sum)

    # pick marginals
    picked, diag = pick_marginals_multi(df_points, price_df, price_col)

    out_dir = results_dir / stage_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dual_sum.to_csv(out_dir / "dual_sum_CH.csv", index=False)
    df_points.to_csv(out_dir / "points_effective_mc_sparse.csv", index=False)
    picked.sort_values(["time", "EffectiveMC", "GenID"], ascending=[True, False, True]).to_csv(
        out_dir / "marginals_per_hour_multi.csv", index=False
    )
    diag.to_csv(out_dir / "hourly_diagnostics.csv", index=False)


def main():
    results_root = BASE_DIR / "Results"
    script_name = Path(__file__).stem
    results_dir = results_root / f"{script_name}_{VERSION}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) CH generator meta (static)
    gens_ch = load_ch_generators_meta(DATA_GENERATORS, COUNTRY_FILTER)
    if gens_ch.empty:
        raise ValueError(f"No generators found for Country=={COUNTRY_FILTER}.")
    gens_ch.to_csv(results_dir / "gens_CH_meta.csv", index=False)
    ch_gen_set = set(gens_ch["GenID"].tolist())

    # 2) CH generation long (still biggest IO; but we only extract CH columns)
    gen_ts = load_generation_wide_selected(GENERATION_WIDE, GEN_SHEET, gens_ch["GenID"].tolist())
    gen_ts.to_csv(results_dir / "gen_CH_long_raw.csv", index=False)

    # 3) Build sparse points for marginal analysis (removes tiny/unreliable points)
    df_points_base = build_sparse_points_for_marginal(gen_ts, gens_ch)
    df_points_base.to_csv(results_dir / "gen_CH_sparse_points.csv", index=False)

    # 4) Prices
    price_1st = load_price_dual_1st(NODAL_DUAL, scale=NODAL_DUAL_SCALE)
    price_2nd = load_price_elprice_ch00(ELPRICE_CH)
    price_1st.to_csv(results_dir / "price_1st.csv", index=False)
    price_2nd.to_csv(results_dir / "price_2nd.csv", index=False)

    # 5) Plot: 1st stage price + battery candidate dual (1st stage dir)
    stage1_dir = BASE_DIR / "InvestmentRun_2050"
    battcand_dual_1st = load_duals_sum(stage1_dir, ["SoCBattCandCon_dual.csv"], ch_gen_set)
    plot_price_vs_battcand_dual(results_dir, price_1st, battcand_dual_1st)

    # 6) Run stage1 & stage2 with their respective dual directories
    run_stage(
        stage_name="stage1",
        stage_dir=stage1_dir,
        df_points_base=df_points_base,
        price_df=price_1st,
        price_col="price_1st",
        ch_gen_set=ch_gen_set,
        results_dir=results_dir,
    )

    run_stage(
        stage_name="stage2",
        stage_dir=BASE_DIR / "CentIv_2050",
        df_points_base=df_points_base,
        price_df=price_2nd,
        price_col="price_2nd",
        ch_gen_set=ch_gen_set,
        results_dir=results_dir,
    )

    print("Saved to:", results_dir)


if __name__ == "__main__":
    main()