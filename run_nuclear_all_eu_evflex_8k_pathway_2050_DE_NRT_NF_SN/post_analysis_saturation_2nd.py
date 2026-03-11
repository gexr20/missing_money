"""
Purpose:
Analyze saturated generators in the 2nd-stage market and visualize their
generation contribution by technology together with the 2nd-stage price.

Input data:
- Data_generators.csv: reads CH generators with Pmax and Technology.
- GenerationPerGen_hourly_ALL_LP.xlsx: 2nd-stage generation data.
- ElPrice_hourly_CH.xlsx: 2nd-stage electricity price (CH00).

Processing:
- For each TIME_WINDOW, select the middle MID_HOURS period.
- Identify saturated generators when:
  generation ≥ 0.99 * Pmax and generation > 1e-3.
- Aggregate saturated generation by Technology for each hour.

Output:
- CSV time series of saturated generation by technology.
- Stacked bar plot of saturated generation with 2nd-stage price on the
  secondary axis.

Output folder:
Results/<script_name>_<VERSION>_<MID_HOURS>/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt

from Final_Generation_cap_check_2nd import (
    load_ch_meta,
    load_generation_long,
    TIME_WINDOWS,
)

VERSION = "v4"

MID_HOURS = 120

SAT_RATIO = 0.99
GEN_EPS = 1e-3

FIGSIZE = (20, 5)
DPI = 400

N_HOURS = 8760
ELPRICE_SHEET = "CH00" 
ELPRICE_COL = "ElPrice" 

TECH_ORDER = ["Dam", "PV-alpine", "PV-roof", "Pump-Open", "Waste", "WindOn"]

TECH_COLOR = {
    "PV-roof": "#ff7f0e", 
    "PV-alpine": "#ffbb78", 

    "Waste": "red",
    "WindOn": "green",

    "Dam": "darkblue",
    "RoR": "skyblue",

    "Pump-Open": "purple",
}

def mid_subwindow(window: Tuple[int, int], hours: int) -> Tuple[int, int]:
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


def read_2nd_price_ch00(path_xlsx: Path) -> pd.Series:
    xls = pd.ExcelFile(path_xlsx, engine="openpyxl")
    sheet_to_use = ELPRICE_SHEET if ELPRICE_SHEET in xls.sheet_names else xls.sheet_names[0]

    df = pd.read_excel(path_xlsx, sheet_name=sheet_to_use, header=0, engine="openpyxl")
    if df.empty:
        raise ValueError(f"[{path_xlsx.name}] Empty sheet: {sheet_to_use}")

    # first column is time (0..8759) by your convention
    time_col = df.columns[0]

    # choose price column
    price_col = ELPRICE_COL
    if price_col not in df.columns:
        # case-insensitive match
        low_map = {str(c).strip().lower(): c for c in df.columns}
        if str(price_col).strip().lower() in low_map:
            price_col = low_map[str(price_col).strip().lower()]
        else:
            # fallback: first numeric-like column excluding time_col
            cand_cols = [c for c in df.columns if c != time_col]
            # try numeric conversion score
            best = None
            best_non_na = -1
            for c in cand_cols:
                s = pd.to_numeric(df[c], errors="coerce")
                nn = int(s.notna().sum())
                if nn > best_non_na:
                    best_non_na = nn
                    best = c
            if best is None or best_non_na <= 0:
                raise ValueError(
                    f"[{path_xlsx.name}] Cannot infer price column. Columns: {df.columns.tolist()}"
                )
            price_col = best

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[time_col, price_col]).copy()
    df[time_col] = df[time_col].astype(int)

    s = df.set_index(time_col)[price_col].sort_index()
    s = s.reindex(range(N_HOURS))

    if s.isna().any():
        miss = int(s.isna().sum())
        raise ValueError(
            f"[{path_xlsx.name}] Missing hours after reindex: {miss} NaN hours. "
            f"Sheet used: {sheet_to_use}, time_col: {time_col}, price_col: {price_col}"
        )
    s.name = "price_2nd"
    return s


def build_sat_generation_by_tech(
    df_long: pd.DataFrame,
    window: Tuple[int, int],
) -> pd.DataFrame:
    t0, t1 = window
    w = df_long[(df_long["time"] >= t0) & (df_long["time"] <= t1)].copy()
    if w.empty:
        idx = pd.Index(range(t0, t1 + 1), name="time")
        return pd.DataFrame(index=idx)

    w["in_band"] = (w["generation"] >= SAT_RATIO * w["Pmax"]) & (w["generation"] > GEN_EPS)
    w = w[w["in_band"]].copy()

    idx = pd.Index(range(t0, t1 + 1), name="time")
    if w.empty:
        return pd.DataFrame(index=idx)

    if "Tech_type" not in w.columns:
        w["Tech_type"] = "UNKNOWN"
    w["Tech_type"] = w["Tech_type"].astype(str).fillna("UNKNOWN")

    g = (
        w.groupby(["time", "Tech_type"], as_index=False)["generation"]
        .sum()
        .rename(columns={"generation": "sat_generation"})
    )

    wide = g.pivot(index="time", columns="Tech_type", values="sat_generation").fillna(0.0)
    wide = wide.reindex(idx).fillna(0.0)
    return wide


def plot_stacked_bar_with_2nd_price(
    wide: pd.DataFrame,
    price_2nd: pd.Series,
    title: str,
    out_png: Path,
) -> None:
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
    ax.set_ylabel("Saturated generation (stacked by Tech_type)")
    ax.set_title(title)

    step = max(1, len(times) // 12)
    xticks = list(range(0, len(times), step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(times[i]) for i in xticks], rotation=0)

    ax.legend(loc="upper left", ncol=2, fontsize=8, title="Technology")

    ax2 = ax.twinx()
    price_aligned = price_2nd.reindex(wide.index)

    ax2.plot(
        x,
        price_aligned.values,
        linestyle="-",
        linewidth=1.2,
        label="2nd stage price",
    )
    ax2.set_ylabel("2nd stage price")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent / "run_nuclear_all_eu_evflex_8k_pathway_2050_DE_NRT_NF_SN"

    results_root = base_dir / "Results"
    results_dir = results_root / f"{Path(__file__).stem}_{VERSION}_{MID_HOURS}"
    results_dir.mkdir(parents=True, exist_ok=True)

    path_genmeta = base_dir / "CentIv_2050" / "mappings" / "Data_generators.csv"
    path_gen_ts = base_dir / "CentIv_2050" / "GenerationPerGen_hourly_ALL_LP.xlsx"
    path_elprice = base_dir / "CentIv_2050" / "ElPrice_hourly_CH.xlsx"

    for p in [path_genmeta, path_gen_ts, path_elprice]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    s_price_2nd = read_2nd_price_ch00(path_elprice)

    df_meta_ch, ch_gen_set = load_ch_meta(path_genmeta)

    df_long = load_generation_long(path_gen_ts, ch_gen_set)

    df_long = df_long.merge(df_meta_ch, on="GenID", how="inner")

    df_long["time"] = pd.to_numeric(df_long["time"], errors="coerce")
    df_long["generation"] = pd.to_numeric(df_long["generation"], errors="coerce")
    df_long["Pmax"] = pd.to_numeric(df_long["Pmax"], errors="coerce")
    df_long = df_long.dropna(subset=["time", "generation", "Pmax"])
    df_long["time"] = df_long["time"].astype(int)

    for idx, win in enumerate(TIME_WINDOWS, start=1):
        mid_t0, mid_t1 = mid_subwindow(win, MID_HOURS)

        wide = build_sat_generation_by_tech(df_long, (mid_t0, mid_t1))

        out_csv = results_dir / f"time_series_window{idx}_mid{MID_HOURS}_sat_generation_by_tech.csv"
        wide.reset_index().to_csv(out_csv, index=False)

        out_png = results_dir / f"plot_window{idx}_mid{MID_HOURS}_SatGen_2ndprice.png"
        plot_stacked_bar_with_2nd_price(
            wide=wide,
            price_2nd=s_price_2nd,
            title=f"window{idx} (middle {MID_HOURS}h: t={mid_t0}..{mid_t1})",
            out_png=out_png,
        )

    (results_dir / "run_info.txt").write_text(
        "\n".join(
            [
                f"BASE_DIR={base_dir}",
                f"GENMETA={path_genmeta}",
                f"GEN_TS={path_gen_ts}",
                f"ELPRICE={path_elprice}",
                f"ELPRICE_SHEET={ELPRICE_SHEET}",
                f"ELPRICE_COL={ELPRICE_COL}",
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