"""
Final_Generation_cap_1st.py

Purpose:
Identify generators that operate close to their capacity (saturated) in the
1st-stage market and analyze their maximum-generation periods.

Input data:
- Data_generators.csv: generator metadata (GenID, Country, Pmax, InvCost, TotVarCost).
- PowerGenerated.csv: 1st-stage generation (GenID, time, generation).

Processing:
- Keep only CH generators.
- For each TIME_WINDOW, detect hours where generation ≥ 0.99 * Pmax and generation > 1e-3.
- Calculate:
  maximum generation, hours in band, and longest continuous saturated period.

Output:
- CSV table summarizing saturation statistics for each generator:
  GenMaxBandPeriods_sorted_<VERSION>.csv
  saved under Results/<script_name>/.
"""

from pathlib import Path
import pandas as pd

VERSION = "v1"

TIME_WINDOWS = [
    (1500, 1700),
    (6000, 6300),
]


def continuous_periods(times: pd.Series) -> list[tuple[int, int, int]]:
    """Return continuous (start, end, length) segments where adjacent time differs by 1."""
    if times.empty:
        return []
    t = times.astype(int).tolist()
    out = []
    start = prev = t[0]
    for x in t[1:]:
        if x == prev + 1:
            prev = x
        else:
            out.append((start, prev, prev - start + 1))
            start = prev = x
    out.append((start, prev, prev - start + 1))
    return out


def load_ch_meta(path_genmeta: Path) -> tuple[pd.DataFrame, set[int]]:
    df_meta = pd.read_csv(path_genmeta)
    first_col = df_meta.columns[0]

    need_cols = {"Country", "Pmax", "InvCost", "TotVarCost", first_col}

    miss = need_cols - set(df_meta.columns)
    if miss:
        raise ValueError(f"Data_generators.csv missing columns: {sorted(miss)}")

    df_meta_ch = (
        df_meta.loc[
            df_meta["Country"].astype(str).eq("CH"),
            [first_col, "Country", "Pmax", "InvCost", "TotVarCost"],
        ]
        .copy()
        .rename(columns={first_col: "GenID"})
    )

    df_meta_ch["GenID"] = pd.to_numeric(df_meta_ch["GenID"], errors="coerce")
    df_meta_ch["Pmax"] = pd.to_numeric(df_meta_ch["Pmax"], errors="coerce")
    df_meta_ch["InvCost"] = pd.to_numeric(df_meta_ch["InvCost"], errors="coerce")
    df_meta_ch["TotVarCost"] = pd.to_numeric(df_meta_ch["TotVarCost"], errors="coerce")

    df_meta_ch = df_meta_ch.dropna(subset=["GenID", "Pmax"])
    df_meta_ch["GenID"] = df_meta_ch["GenID"].astype(int)
    df_meta_ch["Country"] = df_meta_ch["Country"].astype(str)

    # (optional but recommended) ensure uniqueness
    dup = df_meta_ch["GenID"].duplicated(keep=False)
    if dup.any():
        bad = df_meta_ch.loc[dup].sort_values("GenID")
        raise ValueError(
            "Duplicated GenID rows for CH in Data_generators.csv (expected unique per generator).\n"
            f"{bad.head(30).to_string(index=False)}"
        )

    ch_gen_set = set(df_meta_ch["GenID"].tolist())
    if not ch_gen_set:
        raise ValueError("No CH generators found in Data_generators.csv (Country==CH).")

    return df_meta_ch, ch_gen_set


def load_generation_long_3cols_1st(path_power_generated: Path, ch_gen_set: set[int]) -> pd.DataFrame:
    """
    Read InvestmentRun_2050/PowerGenerated.csv (header=None), expected 3 cols:
      col0 = GenID, col1 = time (0..8759), col2 = generation
    Return long table: time, GenID, generation (CH filtered)
    """
    df = pd.read_csv(path_power_generated, header=None, low_memory=False)
    if df.shape[1] < 3:
        raise ValueError(f"{path_power_generated.name}: expected >=3 columns (GenID,time,generation)")

    df = df.iloc[:, :3].copy()
    df.columns = ["GenID", "time", "generation"]

    df["GenID"] = pd.to_numeric(df["GenID"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce")

    df = df.dropna(subset=["GenID", "time", "generation"]).copy()
    df["GenID"] = df["GenID"].astype(int)
    df["time"] = df["time"].astype(int)
    df["generation"] = df["generation"].astype(float)

    # hard filter to CH gens
    df = df[df["GenID"].isin(ch_gen_set)].copy()

    return df


def build_max_band_periods(df_long: pd.DataFrame, windows: list[tuple[int, int]]) -> pd.DataFrame:
    # static columns per GenID
    static_cols = [
        c
        for c in ["GenID", "Country", "Country_from_xlsx", "Tech_type", "Pmax", "InvCost", "TotVarCost"]
        if c in df_long.columns
    ]
    out = (
        df_long[static_cols]
        .drop_duplicates(subset=["GenID"])
        .sort_values("GenID")
        .reset_index(drop=True)
    )

    for w_idx, (t0, t1) in enumerate(windows, start=1):
        w = df_long[(df_long["time"] >= t0) & (df_long["time"] <= t1)].copy()
        if w.empty:
            out[f"window{w_idx}_max_generation"] = pd.NA
            out[f"window{w_idx}_hours_in_band"] = 0
            out[f"window{w_idx}_longest_period_hours"] = 0
            continue

        w["time"] = pd.to_numeric(w["time"], errors="coerce").astype("Int64")
        w["generation"] = pd.to_numeric(w["generation"], errors="coerce")
        w = w.dropna(subset=["time", "generation", "GenID"])
        w["time"] = w["time"].astype(int)
        w["GenID"] = w["GenID"].astype(int)

        max_by_gen = w.groupby("GenID", as_index=False)["generation"].max().rename(
            columns={"generation": f"window{w_idx}_max_generation"}
        )
        w = w.merge(max_by_gen, on="GenID", how="left")

        # saturation / in-band rule (unchanged)
        lower = w["Pmax"] * 0.99
        w["in_band"] = (w["generation"] >= lower) & (w["generation"] > 1e-3)

        periods_map = {}
        hours_in_band_map = {}
        longest_period_map = {}

        for gid, gdf in w[w["in_band"]].groupby("GenID"):
            times_in_band = gdf["time"].sort_values().drop_duplicates()
            periods = continuous_periods(times_in_band)
            gid = int(gid)
            periods_map[gid] = periods
            hours_in_band_map[gid] = int(times_in_band.shape[0])  # 1hr resolution
            longest_period_map[gid] = max((p[2] for p in periods), default=0)

        max_n = max((len(v) for v in periods_map.values()), default=0)

        rows = []
        for gid in out["GenID"].astype(int).tolist():
            periods = periods_map.get(gid, [])
            row = {"GenID": gid}
            row[f"window{w_idx}_hours_in_band"] = hours_in_band_map.get(gid, 0)
            row[f"window{w_idx}_longest_period_hours"] = longest_period_map.get(gid, 0)
            for k in range(1, max_n + 1):
                s_col = f"window{w_idx}_per_start{k}"
                e_col = f"window{w_idx}_per_end{k}"
                if k <= len(periods):
                    row[s_col] = periods[k - 1][0]
                    row[e_col] = periods[k - 1][1]
                else:
                    row[s_col] = pd.NA
                    row[e_col] = pd.NA
            rows.append(row)

        df_periods_wide = pd.DataFrame(rows)

        out = out.merge(max_by_gen, on="GenID", how="left").merge(df_periods_wide, on="GenID", how="left")

    # sort key: longest continuous in-band hours across windows
    w1 = "window1_longest_period_hours"
    w2 = "window2_longest_period_hours"
    if w1 not in out.columns:
        out[w1] = 0
    if w2 not in out.columns:
        out[w2] = 0
    out["longest_at_max_hours"] = out[[w1, w2]].max(axis=1)

    # Put Country next to GenID
    cols = out.columns.tolist()
    if "Country" in cols:
        cols.remove("Country")
        cols.insert(cols.index("GenID") + 1, "Country")
        out = out[cols]

    out = out.sort_values(
        by=["longest_at_max_hours", "window1_max_generation", "GenID"]
        if "window1_max_generation" in out.columns
        else ["longest_at_max_hours", "GenID"],
        ascending=[False, False, True] if "window1_max_generation" in out.columns else [False, True],
        na_position="last",
    )

    return out


def main():
    base_dir = Path(__file__).resolve().parent / "run_nuclear_all_eu_evflex_8k_pathway_2050_DE_WRT_NF_SN"
    results_root = base_dir / "Results"
    script_name = Path(__file__).stem
    results_dir = results_root / script_name
    results_dir.mkdir(parents=True, exist_ok=True)

    path_genmeta = base_dir / "CentIv_2050" / "mappings" / "Data_generators.csv"

    # 1st-stage generation (updated)
    path_gen_ts_1st = base_dir / "InvestmentRun_2050" / "PowerGenerated.csv"

    if not path_genmeta.exists():
        raise FileNotFoundError(f"Missing: {path_genmeta}")
    if not path_gen_ts_1st.exists():
        raise FileNotFoundError(f"Missing: {path_gen_ts_1st}")

    df_meta_ch, ch_gen_set = load_ch_meta(path_genmeta)

    # generation long from 1st stage
    df_long = load_generation_long_3cols_1st(path_gen_ts_1st, ch_gen_set)

    # Attach meta (hard CH constraint via inner)
    df_long = df_long.merge(df_meta_ch, on="GenID", how="inner")

    df_out = build_max_band_periods(df_long, TIME_WINDOWS)

    out_path = results_dir / f"GenMaxBandPeriods_sorted_{VERSION}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()