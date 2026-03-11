"""
Purpose:
Analyze saturated generators in the 1st-stage market and visualize their
generation contribution by technology together with the 1st-stage price.

Input data:
- Data_generators.csv: reads CH generators with Pmax and Technology.
- PowerGenerated.csv: 1st-stage generation (GenID, time, generation).
- NodalConstraint_one_CH_dual.csv: 1st-stage price (time, price_1st).

Processing:
- For each predefined TIME_WINDOW, select the middle MID_HOURS period.
- Identify saturated generators when:
  generation ≥ 0.99 * Pmax and generation > 1e-3.
- Aggregate saturated generation by Technology for each hour.

Output:
- CSV time series of saturated generation by technology.
- Stacked bar plot of saturated generation with 1st-stage price on the
  secondary axis.

Output folder:
Results/<script_name>_<VERSION>_<MID_HOURS>/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt

from Final_Generation_cap_check_1st import TIME_WINDOWS

VERSION = "v1"
MID_HOURS = 120

SAT_RATIO = 0.99
GEN_EPS = 1e-3

FIGSIZE = (20, 5)
DPI = 400

TECH_ORDER = ["Dam", "PV-alpine", "PV-roof", "Pump-Open", "Waste", "WindOn"]

TECH_COLOR = {
    "PV-roof": "#ff7f0e",     # darker orange
    "PV-alpine": "#ffbb78",   # lighter orange

    "Waste": "red",
    "WindOn": "green",

    "Dam": "darkblue",
    "RoR": "skyblue",

    "Pump-Open": "purple",
}

def mid_subwindow(window: Tuple[int, int], hours: int) -> Tuple[int, int]:
    """在 inclusive [t0,t1] 内取中间 hours 小时（inclusive）。"""
    t0, t1 = int(window[0]), int(window[1])
    if t1 < t0:
        raise ValueError(f"Bad window: {window}")

    length = t1 - t0 + 1
    if hours >= length:
        return t0, t1

    center = t0 + length // 2
    start = center - hours // 2
    end = start + hours - 1

    if start < t0:
        start = t0
        end = start + hours - 1
    if end > t1:
        end = t1
        start = end - hours + 1

    return int(start), int(end)


# ---- loaders (1st stage) ----
def load_generation_long_3cols(path: Path, time_min: int = 0, time_max: int = 8759) -> pd.DataFrame:
    """
    PowerGenerated.csv long format:
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

    out = out[(out["time"] >= time_min) & (out["time"] <= time_max)].copy()
    return out


def load_price_dual_1st_two_cols(path: Path, time_min: int = 0, time_max: int = 8759) -> pd.DataFrame:
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

    out = out[(out["time"] >= time_min) & (out["time"] <= time_max)].reset_index(drop=True)
    return out


def load_ch_meta_with_tech(path_genmeta: Path) -> tuple[pd.DataFrame, set[int]]:
    """
    Data_generators.csv:
      - first column: GenID
      - Country, Pmax, Technology
    Return:
      df_meta_ch with columns: GenID, Country, Pmax, Tech_type
      ch_gen_set
    """
    df = pd.read_csv(path_genmeta)
    genid_col = df.columns[0]

    need = {genid_col, "Country", "Pmax", "Technology"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Data_generators.csv missing columns: {sorted(miss)}")

    out = df.loc[df["Country"].astype(str).eq("CH"), [genid_col, "Country", "Pmax", "Technology"]].copy()
    out = out.rename(columns={genid_col: "GenID", "Technology": "Tech_type"})

    out["GenID"] = pd.to_numeric(out["GenID"], errors="coerce")
    out["Pmax"] = pd.to_numeric(out["Pmax"], errors="coerce")
    out["Tech_type"] = out["Tech_type"].astype(str)

    out = out.dropna(subset=["GenID", "Pmax"]).copy()
    out["GenID"] = out["GenID"].astype(int)
    out["Country"] = out["Country"].astype(str)

    dup = out["GenID"].duplicated(keep=False)
    if dup.any():
        bad = out.loc[dup].sort_values("GenID")
        raise ValueError(
            "Duplicated GenID rows for CH in Data_generators.csv (expected unique per generator).\n"
            f"{bad.head(30).to_string(index=False)}"
        )

    ch_gen_set = set(out["GenID"].tolist())
    if not ch_gen_set:
        raise ValueError("No CH generators found in Data_generators.csv (Country==CH).")

    return out.reset_index(drop=True), ch_gen_set


# ---- core logic ----
def build_sat_generation_by_tech(df_long: pd.DataFrame, window: Tuple[int, int]) -> pd.DataFrame:
    t0, t1 = window
    w = df_long[(df_long["time"] >= t0) & (df_long["time"] <= t1)].copy()

    idx = pd.Index(range(t0, t1 + 1), name="time")
    if w.empty:
        return pd.DataFrame(index=idx)

    lower = w["Pmax"] * 0.99
    w["in_band"] = (w["generation"] >= lower) & (w["generation"] > 1e-3)

    w = w[w["in_band"]].copy()
    if w.empty:
        return pd.DataFrame(index=idx)

    w["Tech_type"] = w["Tech_type"].astype(str).fillna("UNKNOWN")

    g = (
        w.groupby(["time", "Tech_type"], as_index=False)["generation"]
        .sum()
        .rename(columns={"generation": "sat_generation"})
    )
    wide = g.pivot(index="time", columns="Tech_type", values="sat_generation").fillna(0.0)
    wide = wide.reindex(idx).fillna(0.0)
    return wide


def plot_stacked_bar_with_1st_price(wide: pd.DataFrame, price_1st: pd.Series, title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    times = wide.index.to_numpy()
    x = list(range(len(times)))

    cols = list(wide.columns)
    ordered = [t for t in TECH_ORDER if t in cols] + [t for t in cols if t not in TECH_ORDER]

    bottom = [0.0] * len(times)
    for tech in ordered:
        y = wide[tech].to_numpy()
        ax.bar(
            x,
            y,
            bottom=bottom,
            width=0.9,
            label=tech,
            color=TECH_COLOR.get(tech),
        )
        bottom = (pd.Series(bottom) + pd.Series(y)).to_list()

    ax.set_xlabel("time")
    ax.set_ylabel("Saturated generation (stacked by Tech)")
    ax.set_title(title)

    step = max(1, len(times) // 12)
    xticks = list(range(0, len(times), step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(times[i]) for i in xticks], rotation=0)

    ax.legend(loc="upper left", ncol=2, fontsize=8, title="Technology")

    ax2 = ax.twinx()
    price_aligned = price_1st.reindex(wide.index)
    ax2.plot(x, price_aligned.values, linestyle="-", linewidth=1.2, label="1st stage price")
    ax2.set_ylabel("1st stage price")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent / "run_nuclear_all_eu_evflex_8k_pathway_2050_DE_WRT_NF_SN"

    results_root = base_dir / "Results"
    results_dir = results_root / f"{Path(__file__).stem}_{VERSION}_{MID_HOURS}"
    results_dir.mkdir(parents=True, exist_ok=True)

    path_genmeta = base_dir / "CentIv_2050" / "mappings" / "Data_generators.csv"
    path_gen_1st = base_dir / "InvestmentRun_2050" / "PowerGenerated.csv"
    path_price_1st = base_dir / "InvestmentRun_2050" / "NodalConstraint_one_CH_dual.csv"

    for p in [path_genmeta, path_gen_1st, path_price_1st]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    # meta (CH) with tech + Pmax
    df_meta_ch, ch_gen_set = load_ch_meta_with_tech(path_genmeta)

    # 1st-stage generation long (ALL gens), then filter CH gens
    df_long = load_generation_long_3cols(path_gen_1st)
    df_long = df_long[df_long["GenID"].isin(ch_gen_set)].copy()

    # attach meta -> bring Tech_type & Pmax
    df_long = df_long.merge(df_meta_ch[["GenID", "Tech_type", "Pmax"]], on="GenID", how="inner")

    # 1st-stage price
    dfp = load_price_dual_1st_two_cols(path_price_1st)
    s_price_1st = dfp.set_index("time")["price_1st"].reindex(range(0, 8760))
    if s_price_1st.isna().any():
        miss = int(s_price_1st.isna().sum())
        raise ValueError(f"[{path_price_1st.name}] Missing hours after reindex: {miss}")
    s_price_1st.name = "price_1st"

    # per window
    for idx, win in enumerate(TIME_WINDOWS, start=1):
        mid_t0, mid_t1 = mid_subwindow(win, MID_HOURS)
        wide = build_sat_generation_by_tech(df_long, (mid_t0, mid_t1))

        out_csv = results_dir / f"time_series_window{idx}_mid{MID_HOURS}_sat_generation_by_tech_1st.csv"
        wide.reset_index().to_csv(out_csv, index=False)

        out_png = results_dir / f"plot_window{idx}_mid{MID_HOURS}_SatGen_with_1stprice.png"
        plot_stacked_bar_with_1st_price(
            wide=wide,
            price_1st=s_price_1st,
            title=f"window{idx} (middle {MID_HOURS}h: t={mid_t0}..{mid_t1})",
            out_png=out_png,
        )

    (results_dir / "run_info.txt").write_text(
        "\n".join(
            [
                f"BASE_DIR={base_dir}",
                f"GENMETA={path_genmeta}",
                f"GEN_1ST={path_gen_1st}",
                f"PRICE_1ST={path_price_1st}",
                f"TIME_WINDOWS={TIME_WINDOWS}",
                f"MID_HOURS={MID_HOURS}",
                f"SAT_RATIO={SAT_RATIO}",
                f"GEN_EPS={GEN_EPS}",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()