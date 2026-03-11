"""
Compare hourly electricity prices between the 1st-stage and 2nd-stage markets
and evaluate their differences over the full year.

Input data:
- ElPrice_hourly_CH.xlsx: 2nd-stage electricity price (CH00 column).
- NodalConstraint_one_CH_dual.csv: 1st-stage electricity price.

Processing:
- Read hourly prices for both stages (0–8759).
- Compute hourly difference: 2nd_Price − 1st_Price.
- Calculate comparison metrics: MAE, RMSE, Bias, and correlation.

Output:
- CSV file with hourly 1st and 2nd stage prices and their difference.
- Text file with comparison metrics.
- Two plots:
  1) 1st vs 2nd stage price time series.
  2) Price difference over time.

Output folder:
Results/<script_name>_<VERSION>/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "v1_SN"

N_HOURS = 8760
SCENARIO = "run_nuclear_all_eu_evflex_8k_pathway_2050_DE_WRT_NF_SN"

# SN: 1st stage price NO /100
SCALE_1ST = 1.0

# 2nd stage file
ELPRICE_SHEET = "CHF_per_MWh"
ELPRICE_COL = "CH00"  # single-node aggregated column name in your screenshot


# -------- paths (same style as marginal generators script) --------
BASE_DIR = Path(__file__).resolve().parent / SCENARIO
RESULTS_ROOT = BASE_DIR / "Results"
SCRIPT_NAME = Path(__file__).stem
RESULTS_DIR = RESULTS_ROOT / f"{SCRIPT_NAME}_{VERSION}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ELPRICE_CH = BASE_DIR / "CentIv_2050" / "ElPrice_hourly_CH.xlsx"
NODAL_DUAL_1ST = BASE_DIR / "InvestmentRun_2050" / "NodalConstraint_one_CH_dual.csv"


def read_2nd_price_ch00(path_xlsx: Path) -> pd.Series:
    df = pd.read_excel(path_xlsx, sheet_name=ELPRICE_SHEET, header=0, engine="openpyxl")

    if ELPRICE_COL not in df.columns:
        raise ValueError(f"[{path_xlsx.name}] Column '{ELPRICE_COL}' not found. Columns: {df.columns.tolist()}")

    # first column is time (0..8759)
    time_col = df.columns[0]

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[ELPRICE_COL] = pd.to_numeric(df[ELPRICE_COL], errors="coerce")
    df = df.dropna(subset=[time_col, ELPRICE_COL]).copy()

    df[time_col] = df[time_col].astype(int)

    s = df.set_index(time_col)[ELPRICE_COL].sort_index()
    s = s.reindex(range(N_HOURS))

    if s.isna().any():
        miss = int(s.isna().sum())
        raise ValueError(f"[{path_xlsx.name}] Missing hours after reindex: {miss} NaN hours.")
    return s


def read_1st_price_one_ch(path_csv: Path) -> pd.Series:
    df = pd.read_csv(path_csv)

    time_col = "NodalConstraint_one_CH"
    val_col = "value"
    if time_col not in df.columns or val_col not in df.columns:
        raise ValueError(f"[{path_csv.name}] Unexpected columns: {df.columns.tolist()}")

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[time_col, val_col]).copy()

    df[time_col] = df[time_col].astype(int)

    s = df.groupby(time_col)[val_col].mean().sort_index()
    s = s.reindex(range(N_HOURS))

    if s.isna().any():
        miss = int(s.isna().sum())
        raise ValueError(f"[{path_csv.name}] Missing hours after reindex: {miss} NaN hours.")
    return s * SCALE_1ST


def main():
    # file existence checks (same style)
    if not ELPRICE_CH.exists():
        raise FileNotFoundError(f"Missing: {ELPRICE_CH}")
    if not NODAL_DUAL_1ST.exists():
        raise FileNotFoundError(f"Missing: {NODAL_DUAL_1ST}")

    p2 = read_2nd_price_ch00(ELPRICE_CH)
    p1 = read_1st_price_one_ch(NODAL_DUAL_1ST)

    out = pd.DataFrame(
        {
            "Time": range(N_HOURS),
            "1st_Price": p1.values,
            "2nd_Price": p2.values,
        }
    )
    out["diff"] = out["2nd_Price"] - out["1st_Price"]

    out_csv = RESULTS_DIR / f"{SCRIPT_NAME}_{VERSION}.csv"
    out.to_csv(out_csv, index=False)

    mae = float(out["diff"].abs().mean())
    rmse = float(np.sqrt((out["diff"] ** 2).mean()))
    bias = float(out["diff"].mean())
    corr = float(out["1st_Price"].corr(out["2nd_Price"]))

    metrics_path = RESULTS_DIR / f"{SCRIPT_NAME}_{VERSION}_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Scenario : {SCENARIO}\n")
        f.write(f"SCALE_1ST: {SCALE_1ST}\n")
        f.write(f"MAE      : {mae:.6f}\n")
        f.write(f"RMSE     : {rmse:.6f}\n")
        f.write(f"Bias     : {bias:.6f}\n")
        f.write(f"Corr     : {corr:.6f}\n")

    fig1 = RESULTS_DIR / f"{SCRIPT_NAME}_{VERSION}.png"
    plt.figure(figsize=(20, 5), dpi=400)
    plt.plot(out["Time"], out["1st_Price"], label="1st")
    plt.plot(out["Time"], out["2nd_Price"], label="2nd")
    plt.xticks(range(0, N_HOURS + 1, 500))
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1, dpi=400)
    plt.close()

    fig2 = RESULTS_DIR / f"{SCRIPT_NAME}_{VERSION}_diff.png"
    plt.figure(figsize=(20, 5), dpi=400)
    plt.plot(out["Time"], out["diff"], label="2nd - 1st")
    plt.axhline(y=0, linestyle="--", linewidth=1)
    plt.xticks(range(0, N_HOURS + 1, 500))

    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=400)
    plt.close()

    print(f"Saved: {out_csv}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {fig1}")
    print(f"Saved: {fig2}")


if __name__ == "__main__":
    main()