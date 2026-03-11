"""
Plot hourly marginal generators for the middle 48h of each analysis window.

Input:
- CentIv_2050/mappings/Data_generators.csv
- Results/OpCost_MarGen_fast_v1/stage1/marginals_per_hour_multi.csv
- Results/OpCost_MarGen_fast_v1/stage2/marginals_per_hour_multi.csv
- Final_Generation_cap_check_v3.py (TIME_WINDOWS)

Output:
- Results/Post_analysis_marginal_gen_<VERSION>/
  stage1_window1_EffectiveMC_stackedLevels_by_Technology.png
  stage1_window2_EffectiveMC_stackedLevels_by_Technology.png
  stage2_window1_EffectiveMC_stackedLevels_by_Technology.png
  stage2_window2_EffectiveMC_stackedLevels_by_Technology.png
"""

from __future__ import annotations

from pathlib import Path
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =========================
# Config
# =========================
VERSION = "v1"
COUNTRY_CODE = "CH"

# outer folder name (same as the outer directory name)
RUN_NAME = "run_nuclear_all_eu_evflex_8k_pathway_2050_DE_WRT_NF_SN"

# opcost marginal csv location under INNER/Results
OPCOST_FOLDER = "OpCost_MarGen_fast_v1"

# plot config
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

# =========================
# I/O + parsing
# =========================
def load_data_generators_ch_with_tech(path: Path, country_code: str = COUNTRY_CODE) -> pd.DataFrame:
    """
    Read Data_generators.csv and keep GenID + Technology for CH.
    Assumption: Technology column name is exactly "Technology".
    """
    df = pd.read_csv(path)

    genid_col = df.columns[0]
    if "Country" not in df.columns:
        raise ValueError("Data_generators.csv missing required column: Country")
    if "Technology" not in df.columns:
        raise ValueError("Data_generators.csv missing required column: Technology")

    out = df[[genid_col, "Country", "Technology"]].copy()
    out = out.rename(columns={genid_col: "GenID"})

    out["GenID"] = pd.to_numeric(out["GenID"], errors="coerce")
    out["Country"] = out["Country"].astype(str)
    out["Technology"] = out["Technology"].astype(str)

    out = out.dropna(subset=["GenID", "Country", "Technology"]).copy()
    out["GenID"] = out["GenID"].astype(int)

    out = out.loc[out["Country"] == country_code].drop_duplicates(subset=["GenID"]).reset_index(drop=True)
    return out


def load_marginals_csv(path: Path) -> pd.DataFrame:
    """
    Read marginals_per_hour_multi.csv (marginal generators, NOT saturation).
    - first column: GenID
    - second column: time
    - must contain EffectiveMC (variable cost + opportunity cost)
    time can repeat (multiple marginal generators per hour).
    """
    df = pd.read_csv(path)

    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError(f"CSV has <2 columns: {path}")

    if "EffectiveMC" not in df.columns:
        raise ValueError(f"Missing column EffectiveMC in {path}. Available: {list(df.columns)}")

    df = df.rename(columns={cols[0]: "GenID", cols[1]: "time"})

    df["GenID"] = pd.to_numeric(df["GenID"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["EffectiveMC"] = pd.to_numeric(df["EffectiveMC"], errors="coerce")

    df = df.dropna(subset=["GenID", "time", "EffectiveMC"]).copy()
    df["GenID"] = df["GenID"].astype(int)
    df["time"] = df["time"].astype(int)
    return df


def add_technology(df: pd.DataFrame, df_genmeta: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(df_genmeta[["GenID", "Technology"]], on="GenID", how="left")
    out["Technology"] = out["Technology"].fillna("UNKNOWN")
    return out


def extract_time_windows_from_py(py_path: Path) -> list[tuple[int, int]]:
    
    if not py_path.exists():
        raise FileNotFoundError(py_path)

    txt = py_path.read_text(encoding="utf-8", errors="ignore")

    m = re.search(r"TIME_WINDOWS\s*=\s*(\[[\s\S]*?\])", txt)
    if not m:
        raise ValueError(f"Cannot find TIME_WINDOWS in {py_path}")

    raw = m.group(1)
    try:
        val = ast.literal_eval(raw)
    except Exception as e:
        raise ValueError(f"Failed to parse TIME_WINDOWS literal in {py_path}: {e}")

    out: list[tuple[int, int]] = []
    for item in val:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((int(item[0]), int(item[1])))

    if len(out) == 0:
        raise ValueError(f"Parsed TIME_WINDOWS but got empty: {val}")

    return out


def middle_48_hours(window: tuple[int, int]) -> tuple[int, int]:
    """
    Take the middle 48 hours of a window (inclusive).
    start = mid-24, end = mid+23
    """
    t0, t1 = window
    mid = (t0 + t1) // 2
    return mid - 48, mid + 47


# =========================
# Plotting
# =========================
def plot_stage_window_stacked_levels(
    df_stage: pd.DataFrame,
    stage_name: str,
    win_name: str,
    t_start: int,
    t_end: int,
    out_dir: Path,
) -> None:
    """
    Plot one stacked bar per hour using EffectiveMC levels of marginal generators.
    Segment colors are fixed by Technology.
    """
    d = df_stage.loc[(df_stage["time"] >= t_start) & (df_stage["time"] <= t_end)].copy()
    if d.empty:
        raise ValueError(f"No rows in {stage_name}-{win_name} within t={t_start}..{t_end}")

    d["Technology"] = d["Technology"].fillna("UNKNOWN").astype(str)

    tech_present = d["Technology"].unique().tolist()
    ordered_tech = [t for t in TECH_ORDER if t in tech_present] + [t for t in tech_present if t not in TECH_ORDER]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    for t in range(t_start, t_end + 1):
        dt = d.loc[d["time"] == t, ["EffectiveMC", "Technology"]].copy()
        if dt.empty:
            continue

        dt["Technology"] = dt["Technology"].fillna("UNKNOWN").astype(str)
        dt["EffectiveMC"] = pd.to_numeric(dt["EffectiveMC"], errors="coerce")
        dt = dt.dropna(subset=["EffectiveMC"]).sort_values("EffectiveMC", ascending=False)
        if dt.empty:
            continue

        mcs = dt["EffectiveMC"].tolist()
        techs = dt["Technology"].tolist()

        bottom = 0.0
        for i in range(len(mcs)):
            top = float(mcs[i])
            next_mc = float(mcs[i + 1]) if i + 1 < len(mcs) else 0.0
            seg_h = top - next_mc
            if seg_h <= 0:
                continue

            color = TECH_COLOR.get(techs[i], "gray")
            ax.bar(t, seg_h, bottom=bottom, width=0.9, color=color)
            bottom += seg_h

    ax.set_xlabel("time")
    ax.set_ylabel("EffectiveMC (stacked to max per hour)")
    ax.set_title(f"{stage_name} - {win_name} (middle 48h: t={t_start}..{t_end})")
    ax.set_xlim(t_start - 0.5, t_end + 0.5)

    handles = [Patch(facecolor=TECH_COLOR.get(t, "gray"), label=t) for t in ordered_tech]
    ax.legend(handles=handles, title="Technology", loc="upper left", ncol=2, fontsize=8)

    fig.tight_layout()
    out_path = out_dir / f"{stage_name}_{win_name}_EffectiveMC_stackedLevels_by_Technology.png"
    fig.savefig(out_path)
    plt.close(fig)


# =========================
# Main
# =========================
def main() -> None:
    script_dir = Path(__file__).resolve().parent

    # outer = where this script is
    outer_dir = script_dir

    # inner = outer/<same name> (contains CentIv_2050, InvestmentRun_2050, Results)
    inner_dir = outer_dir / outer_dir.name
    if not inner_dir.exists():
        raise FileNotFoundError(f"Inner run dir not found: {inner_dir}")

    # output goes to OUTER/Results/<script>_<VERSION>
    results_root = inner_dir / "Results"
    results_root.mkdir(parents=True, exist_ok=True)
    out_dir = results_root / f"{Path(__file__).stem}_{VERSION}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # inputs from INNER
    path_genmeta = inner_dir / "CentIv_2050" / "mappings" / "Data_generators.csv"
    if not path_genmeta.exists():
        raise FileNotFoundError(f"Missing: {path_genmeta}")

    path_timewin_py = outer_dir / "Final_Generation_cap_check_1st.py"
    if not path_timewin_py.exists():
        raise FileNotFoundError(f"Missing: {path_timewin_py}")

    opcost_root = inner_dir / "Results" / OPCOST_FOLDER
    path_stage1 = opcost_root / "stage1" / "marginals_per_hour_multi.csv"
    path_stage2 = opcost_root / "stage2" / "marginals_per_hour_multi.csv"

    if not path_stage1.exists():
        raise FileNotFoundError(f"Missing: {path_stage1}")
    if not path_stage2.exists():
        raise FileNotFoundError(f"Missing: {path_stage2}")

    # load meta + time windows
    df_genmeta = load_data_generators_ch_with_tech(path_genmeta, country_code=COUNTRY_CODE)

    time_windows = extract_time_windows_from_py(path_timewin_py)
    if len(time_windows) < 2:
        raise ValueError(f"Need at least 2 TIME_WINDOWS, got: {time_windows}")

    w1s, w1e = middle_48_hours(time_windows[0])
    w2s, w2e = middle_48_hours(time_windows[1])

    # load marginal tables + add technology
    df1 = add_technology(load_marginals_csv(path_stage1), df_genmeta)
    df2 = add_technology(load_marginals_csv(path_stage2), df_genmeta)

    # plots (4 figures)
    plot_stage_window_stacked_levels(df1, "stage1", "window1", w1s, w1e, out_dir)
    plot_stage_window_stacked_levels(df1, "stage1", "window2", w2s, w2e, out_dir)
    plot_stage_window_stacked_levels(df2, "stage2", "window1", w1s, w1e, out_dir)
    plot_stage_window_stacked_levels(df2, "stage2", "window2", w2s, w2e, out_dir)

    # run log
    (out_dir / "run_info.txt").write_text(
        "\n".join(
            [
                f"OUTER_DIR={outer_dir}",
                f"INNER_DIR={inner_dir}",
                f"OUTPUT_DIR={out_dir}",
                f"DATA_GENERATORS={path_genmeta}",
                f"TIME_WINDOWS_SOURCE={path_timewin_py}",
                f"TIME_WINDOWS_PARSED={time_windows}",
                f"MIDDLE48_WINDOW1={w1s}..{w1e}",
                f"MIDDLE48_WINDOW2={w2s}..{w2e}",
                f"STAGE1_CSV={path_stage1}",
                f"STAGE2_CSV={path_stage2}",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()