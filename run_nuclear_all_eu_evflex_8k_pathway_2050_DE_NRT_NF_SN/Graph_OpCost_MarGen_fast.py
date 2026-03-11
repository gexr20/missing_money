"""
Graph_OpCost_MarGen_fast.py

Plot opportunity cost signal for a selected hydro generator.

Data structure (same BASE_DIR scenario folder):
- BASE_DIR/run_* scenario directory
- Results/<script_name>/ output figure

Input files
- Data_generators.csv: generator metadata (GenID, Country, TotVarCost, Pmax)
- GenerationPerGen_hourly_ALL_LP.xlsx: hourly generation (wide format)
- ElPrice_hourly_CH.xlsx: 2nd stage price (CH00 bus, column B)
- NodalConstraint_one_CH_dual.csv: 1st stage price (dual of nodal constraint)
- SoCHydroDamCon_dual.csv: SOC dual values for hydro dam generators

Logic
- Select a single generator (GenID=19 Dam)
- Load SOC dual from both stages
- Convert SOC dual to opportunity cost signal (-dual)
- Plot time series comparison:
    price_1st
    price_2nd
    -SoCHydroDam dual (1st stage)
    -SoCHydroDam dual (2nd stage)

Output
- Results/Graph_OpCost_MarGen_fast/gen19_price12_damSOCdual.png

Note
- Script only generates the figure (no additional CSV outputs).
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

# your target generator
TARGET_GENID = 19
TARGET_DAM_DUAL_FILE = "SoCHydroDamCon_dual.csv"


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


def load_genid_dual_from_file(stage_dir: Path, filename: str, genid: int) -> pd.DataFrame:
    """Load a single dual file and filter to GenID; returns time, dual."""
    p = stage_dir / filename
    if not p.exists():
        raise FileNotFoundError(p)
    d = load_dual_long(p)
    d = d.loc[d["GenID"] == int(genid), ["time", "dual"]].copy()
    if d.empty:
        return pd.DataFrame(columns=["time", "dual"])
    # if duplicates, sum by time
    d = d.groupby("time", as_index=False)["dual"].sum()
    d["time"] = d["time"].astype(int)
    return d


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

    df = df.loc[df["is_on"] & df["eligible_gen_level"]].copy()
    return df


def attach_opportunity_cost(df_points: pd.DataFrame, dual_sum: pd.DataFrame) -> pd.DataFrame:
    """EffectiveMC = TotVarCost + (-dual_sum). Non-storage -> dual_sum=0."""
    df = df_points.merge(dual_sum, on=["GenID", "time"], how="left")
    df["dual"] = df["dual"].fillna(0.0)
    df["OppCost"] = DUAL_TO_OPPCOST_SIGN * DUAL_SCALE * df["dual"]
    df["EffectiveMC"] = df["TotVarCost"] + df["OppCost"]
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
        .agg(n_points=("GenID", "count"), n_feasible=("mc_le_price", "sum"))
        .merge(
            picked.groupby("time", as_index=False)["EffectiveMC"].max().rename(columns={"EffectiveMC": "picked_max_effective_mc"}),
            on="time",
            how="left",
        )
    )
    return picked, diag


def plot_gen19_dam_soc_dual_vs_prices(
    results_dir: Path,
    price_1st: pd.DataFrame,
    price_2nd: pd.DataFrame,
    dual_1st: pd.DataFrame,
    dual_2nd: pd.DataFrame,
    genid: int,
) -> None:

    df = price_1st.merge(price_2nd, on="time", how="outer")
    df = df.merge(dual_1st.rename(columns={"dual": "dam_dual_1st"}), on="time", how="left")
    df = df.merge(dual_2nd.rename(columns={"dual": "dam_dual_2nd"}), on="time", how="left")

    df = df.sort_values("time").reset_index(drop=True)
    df["dam_dual_1st"] = df["dam_dual_1st"].fillna(0.0)
    df["dam_dual_2nd"] = df["dam_dual_2nd"].fillna(0.0)

    # negated for opp cost interpretation
    df["neg_dam_dual_1st"] = -df["dam_dual_1st"]
    df["neg_dam_dual_2nd"] = -df["dam_dual_2nd"]

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # prices
    ax1.plot(df["time"], df["price_1st"], label="price_1st")
    ax1.plot(df["time"], df["price_2nd"], label="price_2nd")
    ax1.set_xlabel("time")
    ax1.set_ylabel("price")
    ax1.legend(loc="upper left")

    # duals
    ax2 = ax1.twinx()
    ax2.plot(
        df["time"],
        df["neg_dam_dual_1st"],
        color="green",
        label=f"-SoCHydroDam dual (1st), GenID={genid}",
    )
    ax2.plot(
        df["time"],
        df["neg_dam_dual_2nd"],
        color="red",
        label=f"-SoCHydroDam dual (2nd), GenID={genid}",
    )
    ax2.set_ylabel("negated SOC dual")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(results_dir / f"gen{genid}_price12_damSOCdual.png", dpi=300)
    plt.close(fig)
    

def run_stage(
    stage_name: str,
    stage_dir: Path,
    df_points_base: pd.DataFrame,
    price_df: pd.DataFrame,
    price_col: str,
    ch_gen_set: set[int],
    results_dir: Path,
) -> None:
    """Stage run on sparse points only."""
    dual_batt = load_duals_sum(stage_dir, BATT_DUAL_FILES, ch_gen_set)
    dual_hydro = load_duals_sum(stage_dir, HYDRO_DUAL_FILES, ch_gen_set)
    dual_sum = pd.concat([dual_batt, dual_hydro], ignore_index=True)
    if not dual_sum.empty:
        dual_sum = dual_sum.groupby(["GenID", "time"], as_index=False)["dual"].sum()
    else:
        dual_sum = pd.DataFrame(columns=["GenID", "time", "dual"])

    df_points = attach_opportunity_cost(df_points_base, dual_sum)

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
    results_dir = results_root / script_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # prices
    price_1st = load_price_dual_1st(NODAL_DUAL, scale=NODAL_DUAL_SCALE)
    price_2nd = load_price_elprice_ch00(ELPRICE_CH)

    # duals
    stage1_dir = BASE_DIR / "InvestmentRun_2050"
    stage2_dir = BASE_DIR / "CentIv_2050"

    dam_dual_1st = load_genid_dual_from_file(stage1_dir, TARGET_DAM_DUAL_FILE, TARGET_GENID)
    dam_dual_2nd = load_genid_dual_from_file(stage2_dir, TARGET_DAM_DUAL_FILE, TARGET_GENID)

    # only plot
    plot_gen19_dam_soc_dual_vs_prices(
        results_dir=results_dir,
        price_1st=price_1st,
        price_2nd=price_2nd,
        dual_1st=dam_dual_1st,
        dual_2nd=dam_dual_2nd,
        genid=TARGET_GENID,
    )

    print("Figure saved to:", results_dir)


if __name__ == "__main__":
    main()