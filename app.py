import io, os, re, zipfile, tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest, GradientBoostingRegressor


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="BMS Processor", layout="wide")
st.title("BMS CSV Processor")
st.caption("Auto-detect + fallback → tables + plots + AI anomaly detection + downloads")


# -----------------------------
# Helpers
# -----------------------------
def _norm(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = s.replace("°", "").replace("(", " ").replace(")", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def find_col(df: pd.DataFrame, includes, excludes=None):
    includes = [_norm(x) for x in includes]
    excludes = [_norm(x) for x in (excludes or [])]
    for c in df.columns:
        nc = _norm(c)
        if all(k in nc for k in includes) and not any(k in nc for k in excludes):
            return c
    return None

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def safe_div(n, d):
    """
    Safe divide that works for BOTH scalars (int/float) and pandas Series.
    Returns 0 when denominator is 0 or NaN.
    """
    if np.isscalar(n) and np.isscalar(d):
        if d in [0, 0.0] or pd.isna(d) or pd.isna(n):
            return 0.0
        return float(n) / float(d)

    n = pd.to_numeric(n, errors="coerce")
    d = pd.to_numeric(d, errors="coerce")
    out = n / d.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def compute_load_rt_and_kw(flow_ls, dT, rho=997, cp=4.186):
    flow_ls = pd.to_numeric(flow_ls, errors="coerce")
    dT = pd.to_numeric(dT, errors="coerce")

    flow_m3s = flow_ls / 1000        # Convert L/s → m³/s
    m_dot = rho * flow_m3s           # kg/s
    q_kw = m_dot * cp * dT           # kW (since cp is kJ/kg·K)
    rt = q_kw / 3.517                # Convert kW → RT

    return rt, q_kw

AM_PM = ["12 AM","1 AM","2 AM","3 AM","4 AM","5 AM","6 AM","7 AM","8 AM","9 AM","10 AM","11 AM",
         "12 PM","1 PM","2 PM","3 PM","4 PM","5 PM","6 PM","7 PM","8 PM","9 PM","10 PM","11 PM"]

def hourly_mean(df, dt_col, y_col):
    tmp = df[[dt_col, y_col]].copy()
    tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors="coerce")
    tmp[y_col] = to_num(tmp[y_col])
    tmp = tmp.dropna()
    tmp["hour"] = tmp[dt_col].dt.hour
    return tmp.groupby("hour")[y_col].mean().reindex(range(24))

def hourly_mean_2(df, dt_col, c1, c2):
    tmp = df[[dt_col, c1, c2]].copy()
    tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors="coerce")
    tmp[c1] = to_num(tmp[c1])
    tmp[c2] = to_num(tmp[c2])
    tmp = tmp.dropna()
    tmp["hour"] = tmp[dt_col].dt.hour
    out = tmp.groupby("hour")[[c1, c2]].mean().reindex(range(24))
    return out

def plot_hourly_line(series24, title, ylabel):
    fig = plt.figure(figsize=(12, 5))
    plt.plot(range(24), series24.values, linewidth=2.5, marker="o", markersize=4)
    plt.title(title, fontweight="bold")
    plt.xticks(range(24), AM_PM, rotation=45, ha="right")
    plt.xlabel("Time of Day")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig

def plot_hourly_overlay(df24_2col, title, ylabel, l1, l2):
    fig = plt.figure(figsize=(12, 5))
    plt.plot(range(24), df24_2col.iloc[:, 0].values, linewidth=2.2, marker="o", markersize=4, label=l1)
    plt.plot(range(24), df24_2col.iloc[:, 1].values, linewidth=2.2, marker="o", markersize=4, label=l2)
    plt.fill_between(range(24), df24_2col.iloc[:, 0].values, df24_2col.iloc[:, 1].values, alpha=0.12)
    plt.title(title, fontweight="bold")
    plt.xticks(range(24), AM_PM, rotation=45, ha="right")
    plt.xlabel("Time of Day")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    return fig

def scatter_plot(x, y, title, xlabel="Cooling Load (RT)", ylabel="Efficiency (kW/RT)"):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(x[m], y[m], s=10, alpha=0.6)
    plt.title(title, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    return fig

def scatter_flagged(x, y, flag, title, ylabel):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    flag = flag.fillna(0).astype(int)
    m = x.notna() & y.notna()
    x, y, flag = x[m], y[m], flag[m]

    fig = plt.figure(figsize=(8, 5))
    normal = flag == 0
    abn = flag == 1
    plt.scatter(x[normal], y[normal], alpha=0.35, label="Normal")
    plt.scatter(x[abn], y[abn], marker="x", s=80, label="AI Flagged")
    plt.title(title, fontweight="bold")
    plt.xlabel("Cooling Load (RT)")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    return fig

def zip_folder(folder_path: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder_path):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, folder_path)
                z.write(full, rel)
    buf.seek(0)
    return buf.getvalue()

def save_fig(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
# -----------------------------
# AI Waste Helper Plots
# -----------------------------
def plot_pareto_cause(cause_summary: pd.DataFrame, title="Pareto: Waste by Cause"):
    cs = cause_summary.copy()
    cs = cs[cs["waste_kWh"] > 0].reset_index(drop=True)
    if cs.empty:
        return None

    cs["cum_kWh"] = cs["waste_kWh"].cumsum()
    total = cs["waste_kWh"].sum()
    cs["cum_pct"] = (cs["cum_kWh"] / total) * 100.0 if total > 0 else 0.0

    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(cs))
    ax1.bar(x, cs["waste_kWh"].values, edgecolor="white")
    ax1.set_ylabel("Waste (kWh)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cs["AI_cause"].astype(str).values, rotation=30, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, cs["cum_pct"].values, marker="D", linewidth=2)
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 105)

    ax1.set_title(title, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_daily_waste(df_out: pd.DataFrame, dt_col: str):
    tmp = df_out.copy()
    tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors="coerce")
    tmp = tmp.dropna(subset=[dt_col])

    tmp["date"] = tmp[dt_col].dt.date
    daily = (tmp.groupby("date")
             .agg(waste_kWh=("AI_waste_kWh", "sum"),
                  cost_SGD=("AI_waste_cost_SGD", "sum"))
             .reset_index())

    fig1, ax = plt.subplots(figsize=(12, 4))
    ax.bar(daily["date"].astype(str), daily["waste_kWh"].values)
    ax.set_title("Daily Waste (kWh/day)", fontweight="bold")
    plt.xticks(rotation=45)
    plt.tight_layout()

    fig2, ax = plt.subplots(figsize=(12, 4))
    ax.bar(daily["date"].astype(str), daily["cost_SGD"].values)
    ax.set_title("Daily Waste Cost (SGD/day)", fontweight="bold")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return daily, fig1, fig2

# -----------------------------
# Upload
# -----------------------------
uploaded = st.file_uploader("Upload BMS CSV", type=["csv"])
if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)
df.columns = df.columns.str.strip()

st.subheader("Preview")
st.dataframe(df.head(10), use_container_width=True)
st.caption(f"Rows: {len(df):,} | Columns: {len(df.columns)}")


# -----------------------------
# Auto-detect mapping
# -----------------------------
auto = {}
auto["dt"] = find_col(df, ["date", "time"]) or find_col(df, ["datetime"]) or find_col(df, ["timestamp"])

auto["chwst"] = find_col(df, ["chilled", "water", "supply", "temp"]) or find_col(df, ["chw", "supply", "temp"]) or find_col(df, ["chwst"])
auto["chwrt"] = find_col(df, ["chilled", "water", "return", "temp"]) or find_col(df, ["chw", "return", "temp"]) or find_col(df, ["chwrt"])
auto["chw_dt"] = find_col(df, ["chilled", "water", "delta", "t"]) or find_col(df, ["chilled", "water", "dt"]) or find_col(df, ["chilled", "water", "Δt"]) or find_col(df, ["chilled", "water", "Δ", "t"])
auto["chw_flow"] = find_col(df, ["chilled", "water", "flow", "rate"]) or find_col(df, ["chw", "flow"])
auto["chw_load_rt_raw"] = find_col(df, ["chilled", "water", "load", "rt"]) or find_col(df, ["chw", "load", "rt"])

auto["cwst"] = find_col(df, ["condenser", "water", "supply", "temp"]) or find_col(df, ["cw", "supply", "temp"]) or find_col(df, ["cwst"])
auto["cwrt"] = find_col(df, ["condenser", "water", "return", "temp"]) or find_col(df, ["cw", "return", "temp"]) or find_col(df, ["cwrt"])
auto["cw_dt"] = find_col(df, ["condenser", "water", "delta", "t"]) or find_col(df, ["condenser", "water", "dt"]) or find_col(df, ["condenser", "water", "Δt"]) or find_col(df, ["condenser", "water", "Δ", "t"])
auto["cw_flow"] = find_col(df, ["condenser", "water", "flow", "rate"]) or find_col(df, ["cw", "flow"])
auto["cw_load_rt_raw"] = find_col(df, ["condenser", "water", "load", "rt"]) or find_col(df, ["cw", "load", "rt"])

auto["chiller_kw"] = find_col(df, ["total", "chiller", "power"]) or find_col(df, ["chiller", "power", "kw"])
auto["chwp_kw"] = find_col(df, ["total", "chwp", "power"]) or find_col(df, ["chw", "pump", "power"])
auto["cwp_kw"] = find_col(df, ["total", "cwp", "power"]) or find_col(df, ["cw", "pump", "power"])
auto["ct_kw"] = find_col(df, ["total", "ct", "power"]) or find_col(df, ["cooling", "tower", "power"])

auto["oat"] = find_col(df, ["oat"]) or find_col(df, ["dry", "bulb"]) or find_col(df, ["outside", "air", "temp"])
auto["rh"] = find_col(df, ["oarh"]) or find_col(df, ["relative", "humidity"]) or find_col(df, ["rh"])

all_cols = ["(none)"] + list(df.columns)

def pick(label, key, required=False):
    guess = auto.get(key)
    idx = all_cols.index(guess) if guess in all_cols else 0
    sel = st.selectbox(label, all_cols, index=idx)
    if required and sel == "(none)":
        st.warning(f"Missing required mapping: {label}")
    return sel

st.subheader("Header Selection")
c1, c2, c3 = st.columns(3)

with c1:
    dt_col = pick("Datetime", "dt", required=True)
    chwst_col = pick("CHW Supply Temp", "chwst", required=True)
    chwrt_col = pick("CHW Return Temp", "chwrt", required=True)
    chw_dt_col = pick("CHW ΔT (optional)", "chw_dt", required=False)
    chw_flow_col = pick("CHW Flow (L/s)", "chw_flow", required=True)
    chw_load_rt_raw_col = pick("CHW Load RT raw (optional)", "chw_load_rt_raw", required=False)

with c2:
    cwst_col = pick("CW Supply Temp", "cwst", required=False)
    cwrt_col = pick("CW Return Temp", "cwrt", required=False)
    cw_dt_col = pick("CW ΔT (optional)", "cw_dt", required=False)
    cw_flow_col = pick("CW Flow (L/s) (optional)", "cw_flow", required=False)
    cw_load_rt_raw_col = pick("CW Load RT raw (optional)", "cw_load_rt_raw", required=False)

with c3:
    chiller_kw_col = pick("TOTAL CHILLER POWER (kW)", "chiller_kw", required=True)
    chwp_kw_col = pick("TOTAL CHWP POWER (kW)", "chwp_kw", required=True)
    cwp_kw_col = pick("TOTAL CWP POWER (kW)", "cwp_kw", required=True)
    ct_kw_col = pick("TOTAL CT POWER (kW)", "ct_kw", required=True)
    oat_col = pick("Dry bulb / OAT (for AI)", "oat", required=False)
    rh_col = pick("RH % (for AI)", "rh", required=False)

sampling_min = st.number_input("Sampling interval (minutes) for occurrence/mins (default 5)", 1, 60, 5)
enable_ai = st.toggle("Enable AI anomaly detection", value=True)

run = st.button("Run full analysis (tables + plots + AI)", type="primary")
if not run:
    st.stop()


# -----------------------------
# Compute df_calc
# -----------------------------
d = df.copy()

d[dt_col] = pd.to_datetime(d[dt_col].astype(str).str.strip(), errors="coerce", dayfirst=True)
d = d.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)

for col in [chwst_col, chwrt_col, chw_dt_col, chw_flow_col, chw_load_rt_raw_col,
            cwst_col, cwrt_col, cw_dt_col, cw_flow_col, cw_load_rt_raw_col,
            chiller_kw_col, chwp_kw_col, cwp_kw_col, ct_kw_col, oat_col, rh_col]:
    if col != "(none)" and col in d.columns:
        d[col] = to_num(d[col])

if chw_dt_col != "(none)" and chw_dt_col in d.columns and d[chw_dt_col].notna().any():
    d["CHW dT"] = d[chw_dt_col]
else:
    d["CHW dT"] = d[chwrt_col] - d[chwst_col]

if cw_dt_col != "(none)" and cw_dt_col in d.columns and d[cw_dt_col].notna().any():
    d["CW dT"] = d[cw_dt_col]
elif (cwst_col != "(none)" and cwrt_col != "(none)" and cwst_col in d.columns and cwrt_col in d.columns):
    d["CW dT"] = d[cwrt_col] - d[cwst_col]
else:
    d["CW dT"] = np.nan

if chw_load_rt_raw_col != "(none)" and chw_load_rt_raw_col in d.columns and d[chw_load_rt_raw_col].notna().any():
    d["CHW load (RT)"] = d[chw_load_rt_raw_col]
    d["CHW load (kW)"] = d["CHW load (RT)"] * 3.517
else:
    rt, kw = compute_load_rt_and_kw(d[chw_flow_col], d["CHW dT"])
    d["CHW load (RT)"] = rt
    d["CHW load (kW)"] = kw

d.loc[d["CHW load (RT)"] <= 0, "CHW load (RT)"] = np.nan

if cw_load_rt_raw_col != "(none)" and cw_load_rt_raw_col in d.columns and d[cw_load_rt_raw_col].notna().any():
    d["CW load (RT)"] = d[cw_load_rt_raw_col]
    d["CW load (kW)"] = d["CW load (RT)"] * 3.517
elif (cw_flow_col != "(none)" and cw_flow_col in d.columns and d[cw_flow_col].notna().any() and d["CW dT"].notna().any()):
    rt, kw = compute_load_rt_and_kw(d[cw_flow_col], d["CW dT"])
    d["CW load (RT)"] = rt
    d["CW load (kW)"] = kw
else:
    d["CW load (RT)"] = np.nan
    d["CW load (kW)"] = np.nan

d["Total plant power (kW)"] = (
    d[chiller_kw_col].fillna(0)
    + d[chwp_kw_col].fillna(0)
    + d[cwp_kw_col].fillna(0)
    + d[ct_kw_col].fillna(0)
)

d["Heat in (kW)"] = d[chiller_kw_col].fillna(0) + d["CHW load (kW)"].fillna(0)
d["Heat out (kW)"] = d["CW load (kW)"]
d["Heat balance (%)"] = safe_div(d["Heat out (kW)"] - d["Heat in (kW)"], d["Heat out (kW)"]) * 100.0

d["Chiller efficiency (kW/RT)"] = safe_div(d[chiller_kw_col].fillna(0), d["CHW load (RT)"])
d["CHWP efficiency (kW/RT)"] = safe_div(d[chwp_kw_col].fillna(0), d["CHW load (RT)"])
d["CWP efficiency (kW/RT)"] = safe_div(d[cwp_kw_col].fillna(0), d["CHW load (RT)"])
d["CT efficiency (kW/RT)"] = safe_div(d[ct_kw_col].fillna(0), d["CHW load (RT)"])
d["Overall efficiency (kW/RT)"] = safe_div(d["Total plant power (kW)"], d["CHW load (RT)"])

df_calc = d


# -----------------------------
# Month filter
# -----------------------------
months = sorted(df_calc[dt_col].dt.month.dropna().unique().astype(int).tolist())
if not months:
    st.error("No valid datetimes found after parsing. Check your Datetime column mapping / format.")
    st.stop()

default_month = 7 if 7 in months else months[0]

import calendar

# Get unique months from data
months = sorted(df_calc[dt_col].dt.month.dropna().unique().astype(int))

# Create mapping {1: "Jan", 2: "Feb", ...}
month_map = {m: calendar.month_abbr[m] for m in months}

# Default month (July if exists)
default_month = 7 if 7 in months else months[0]

st.divider()
st.subheader("Select Month for Profile Plots")

# Show month names instead of numbers
sel_month_name = st.selectbox(
    "Month",
    [month_map[m] for m in months],
    index=months.index(default_month)
)

# Convert back to numeric month
sel_month = [k for k, v in month_map.items() if v == sel_month_name][0]

df_m = df_calc[df_calc[dt_col].dt.month == sel_month].copy()


# -----------------------------
# Tabs
# -----------------------------
st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["✅ Tables", "Graphs", "Graphs(extd)", "AI anomoly & detection"])


# -----------------------------
# Tables
# -----------------------------
with tab1:
    st.subheader("Efficiency Summary")

    chw_rt = df_calc["CHW load (RT)"]
    den = float(chw_rt[chw_rt > 0].sum())

    def sumif_pos(series: pd.Series) -> float:
        s = series.fillna(0)
        return float(s[s > 0].sum())

    def eff_sumif(power_series: pd.Series) -> float:
        num = sumif_pos(power_series)
        return float(num / den) if den > 0 and np.isfinite(den) else 0.0

    eff_vals = [
        eff_sumif(df_calc[chiller_kw_col]),
        eff_sumif(df_calc[chwp_kw_col]),
        eff_sumif(df_calc[cwp_kw_col]),
        eff_sumif(df_calc[ct_kw_col]),
        eff_sumif(df_calc["Total plant power (kW)"]),
    ]

    eff_summary = pd.DataFrame({
        "Metric": [
            "Chiller efficiency\n(kW/RT)",
            "CHWP efficiency (kW/RT)",
            "CWP efficiency (kW/RT)",
            "CT efficiency (kW/RT)",
            "Overall efficiency\n(kW/RT)",
        ],
        "Value": eff_vals
    })
    eff_summary["Value"] = eff_summary["Value"].astype(float).round(4)

    st.dataframe(eff_summary, use_container_width=True, hide_index=True)

    st.subheader("Heat Balance Summary")

    hb = df_calc["Heat balance (%)"]
    total = int(hb.notna().sum())

    gt5 = int((hb >= 5).sum())
    ltneg5 = int((hb < -5).sum())

    out = int((hb.abs() > 5).sum())
    within = int((hb.abs() <= 5).sum())

    pct_within = (within / total * 100) if total > 0 else 0.0
    heat_balance_summary = pd.DataFrame({
        "Metric": [
            "Heat Balance data count",
            "Data > 5%",
            "Data < -5%",
            "Total count out",
            "Total count",
            "Total count within",
            "% within",
        ],
        "Value": [total, gt5, ltneg5, out, total, within, pct_within]
    })

    st.dataframe(heat_balance_summary, use_container_width=True)

        # -----------------------------
    # Heat Balance: show where counts OUT are
    # -----------------------------
    st.subheader("Heat Balance Out-of-Tolerance Events")

    hb_col = "Heat balance (%)"
    hb_view = df_calc.copy()
    hb_view[hb_col] = pd.to_numeric(hb_view[hb_col], errors="coerce")

    out_mask = hb_view[hb_col].notna() & (hb_view[hb_col].abs() > 5)
    hb_out = hb_view.loc[out_mask].copy()

    hb_out["HB_flag"] = np.where(hb_out[hb_col] > 5, "> +5%", "< -5%")
    hb_out["HB_abs"] = hb_out[hb_col].abs()

    preferred_cols = [
        dt_col, hb_col, "HB_flag",
        "CHW load (RT)", "CHW load (kW)",
        "Heat in (kW)", "Heat out (kW)",
        chiller_kw_col, chwp_kw_col, cwp_kw_col, ct_kw_col,
        "Total plant power (kW)",
        "CHW dT", chw_flow_col,
        "CW dT", cw_flow_col,
    ]
    show_cols = [c for c in preferred_cols if (c != "(none)") and (c in hb_out.columns)]

    hb_out = hb_out.sort_values(dt_col, ascending=True)

    # Round some columns for readability
    for c in [hb_col, "CHW load (RT)", "CHW load (kW)", "Heat in (kW)", "Heat out (kW)",
              "Total plant power (kW)", "CHW dT", "CW dT"]:
        if c in hb_out.columns:
            hb_out[c] = hb_out[c].astype(float).round(3)

    st.caption(f"Out-of-tolerance rows: {len(hb_out):,}")
    st.dataframe(hb_out[show_cols], use_container_width=True)

    st.download_button(
        "Download heat balance out-of-tolerance rows (CSV)",
        data=hb_out[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="heat_balance_out_of_tolerance.csv",
        mime="text/csv",
    )    

    st.subheader("Tonnage Bin Summary")
    BIN_SIZE_RT = 100
    MAX_RT_EDGE = 1600

    df_bin = df_calc.copy()
    if (chw_load_rt_raw_col != "(none)") and (chw_load_rt_raw_col in df_bin.columns) and df_bin[chw_load_rt_raw_col].notna().any():
        df_bin["RT_for_bin"] = pd.to_numeric(df_bin[chw_load_rt_raw_col], errors="coerce")
        rt_source = chw_load_rt_raw_col
    else:
        df_bin["RT_for_bin"] = pd.to_numeric(df_bin["CHW load (RT)"], errors="coerce")
        rt_source = "CHW load (RT)"

    st.caption(f"RT used for binning: **{rt_source}** | Sampling interval used: **{sampling_min} min**")

    labels_bins = ["<100"] + [f"{i} to <{i+100}" for i in range(100, MAX_RT_EDGE, 100)] + [f">={MAX_RT_EDGE}"]

    df_bin["Tonnage"] = None
    df_bin.loc[df_bin["RT_for_bin"] < 100, "Tonnage"] = "<100"
    for i in range(100, MAX_RT_EDGE, 100):
        df_bin.loc[(df_bin["RT_for_bin"] >= i) & (df_bin["RT_for_bin"] < i + 100), "Tonnage"] = f"{i} to <{i+100}"
    df_bin.loc[df_bin["RT_for_bin"] >= MAX_RT_EDGE, "Tonnage"] = f">={MAX_RT_EDGE}"

    grp = df_bin.groupby("Tonnage", observed=True)
    occ_rows = grp.size().reindex(labels_bins).fillna(0).astype(int)
    occ_mins = (occ_rows * float(sampling_min)).astype(float)

    def band_eff_ratio(g):
        rt = pd.to_numeric(g["CHW load (RT)"], errors="coerce")
        kw = pd.to_numeric(g["Total plant power (kW)"], errors="coerce")
        rt_sum = rt[rt > 0].sum()
        kw_sum = kw[kw > 0].sum()
        return float(kw_sum / rt_sum) if rt_sum > 0 else 0.0

    avg_eff = grp.apply(band_eff_ratio).reindex(labels_bins).fillna(0.0)

    tonnage_table = pd.DataFrame({
        "Tonnage": labels_bins,
        "Avg Chiller Plant Eff": avg_eff.values,
        "Occurance": occ_rows.values,
        "Occurance (mins)": occ_mins.values,
        "Cumulative Frequency": occ_rows.cumsum().values
    })

    total_occ = int(occ_rows.sum())
    tonnage_table["Percentage"] = np.where(total_occ > 0, occ_rows.values / total_occ * 100.0, 0.0)
    tonnage_table["Cumulative %"] = np.where(total_occ > 0, tonnage_table["Cumulative Frequency"] / total_occ * 100.0, 0.0)

    tonnage_table["Avg Chiller Plant Eff"] = tonnage_table["Avg Chiller Plant Eff"].round(6)
    tonnage_table["Percentage"] = tonnage_table["Percentage"].round(1)
    tonnage_table["Cumulative %"] = tonnage_table["Cumulative %"].round(1)

    st.dataframe(tonnage_table, use_container_width=True)

    st.subheader("Calculated Data")
    st.dataframe(df_calc.head(50), use_container_width=True)


# -----------------------------
# Graphs
# -----------------------------
with tab2:
    st.subheader(f"Graphs (Month = {sel_month})")

    fig = plot_hourly_line(
        hourly_mean(df_m, dt_col, "CHW load (RT)"),
        f"{sel_month}: Typical 24-Hour Cooling Load Profile (CHW load)",
        "Avg Cooling Load (RT)"
    )
    st.pyplot(fig)

    KW_PER_RT = 3.517
    cooling_rt = (to_num(df_m["CHW load (RT)"]) + to_num(df_m[chiller_kw_col]) / KW_PER_RT)
    cooling_rt = cooling_rt[(cooling_rt.notna()) & (cooling_rt > 0)]

    edges = [-np.inf] + list(range(100, 1700, 100)) + [np.inf]
    labels_occ = ["<100"] + [f"{i} to <{i+100}" for i in range(100, 1600, 100)] + [">1600"]
    bins = pd.cut(cooling_rt, bins=edges, right=False, labels=labels_occ, include_lowest=True)
    occ = bins.value_counts(sort=False)
    cum = occ.cumsum()
    cum_pct = (cum / occ.sum() * 100) if occ.sum() > 0 else cum * 0

    fig, ax1 = plt.subplots(figsize=(14, 6))
    x = np.arange(len(occ.index))
    ax1.bar(x, occ.values, width=0.85, edgecolor="white", label="Occurrence")
    ax2 = ax1.twinx()
    ax2.plot(x, cum_pct.values, marker="D", markersize=4, linewidth=2, label="Cumulative %")

    ax1.set_title(f"{sel_month}: Cooling Load Occurrence Distribution", fontweight="bold", pad=12)
    ax1.set_xlabel("Tonnage Range (RT)")
    ax1.set_ylabel("Occurrence (No. of points)")
    ax2.set_ylabel("Cumulative Percentage (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(occ.index.astype(str), rotation=45, ha="right")

    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    fig = plot_hourly_overlay(
        hourly_mean_2(df_m, dt_col, chwst_col, chwrt_col),
        f"{sel_month}: Average CHW Temperatures",
        "Temperature (°C)",
        "CHW Supply", "CHW Return"
    )
    st.pyplot(fig)

    fig = plot_hourly_line(hourly_mean(df_m, dt_col, "CHW dT"), f"{sel_month}: Average CHW ΔT", "ΔT (°C)")
    st.pyplot(fig)

    fig = plot_hourly_line(hourly_mean(df_m, dt_col, chw_flow_col), f"{sel_month}: Average CHW Flow Rate", "Flow (L/s)")
    st.pyplot(fig)

    if cwst_col != "(none)" and cwrt_col != "(none)" and cwst_col in df_m.columns and cwrt_col in df_m.columns:
        fig = plot_hourly_overlay(
            hourly_mean_2(df_m, dt_col, cwst_col, cwrt_col),
            f"{sel_month}: Typical 24-Hour Condenser Water Temp Profile",
            "Temperature (°C)",
            "CW Supply (From Tower)", "CW Return (To Tower)"
        )
        st.pyplot(fig)
    else:
        st.info("CW Supply/Return not mapped → skipping CW temp overlay plot.")

    if df_m["CW dT"].notna().any():
        fig = plot_hourly_line(hourly_mean(df_m, dt_col, "CW dT"), f"{sel_month}: Average CW Delta T Profile", "ΔT (°C)")
        st.pyplot(fig)
    else:
        st.info("CW ΔT not available → skipping CW ΔT plot.")

    if cw_flow_col != "(none)" and cw_flow_col in df_m.columns and df_m[cw_flow_col].notna().any():
        fig = plot_hourly_line(hourly_mean(df_m, dt_col, cw_flow_col), f"{sel_month}: Average CW Flow Rate Profile", "Flow (L/s)")
        st.pyplot(fig)
    else:
        st.info("CW Flow not mapped → skipping CW flow plot.")


# -----------------------------
# Graphs (extd)
# -----------------------------
with tab3:
    st.subheader("Graphs (Efficiency plots)")

    d_plot = df_calc.copy()
    for col in ["Chiller efficiency (kW/RT)", "CHWP efficiency (kW/RT)", "CWP efficiency (kW/RT)",
                "CT efficiency (kW/RT)", "Overall efficiency (kW/RT)"]:
        d_plot.loc[(d_plot[col] < 0) | (d_plot[col] > 5), col] = np.nan

    st.pyplot(plot_hourly_line(hourly_mean(d_plot, dt_col, "Chiller efficiency (kW/RT)"),
                               "Chiller efficiency vs time (Daily Profile)", "kW/RT"))
    st.pyplot(plot_hourly_line(hourly_mean(d_plot, dt_col, "CHWP efficiency (kW/RT)"),
                               "CHWP efficiency vs time (Daily Profile)", "kW/RT"))
    st.pyplot(plot_hourly_line(hourly_mean(d_plot, dt_col, "CWP efficiency (kW/RT)"),
                               "CWP efficiency vs time (Daily Profile)", "kW/RT"))
    st.pyplot(plot_hourly_line(hourly_mean(d_plot, dt_col, "CT efficiency (kW/RT)"),
                               "CT efficiency vs time (Daily Profile)", "kW/RT"))
    st.pyplot(plot_hourly_line(hourly_mean(d_plot, dt_col, "Overall efficiency (kW/RT)"),
                               "Chiller plant efficiency vs time (Daily Profile)", "kW/RT"))

    st.pyplot(scatter_plot(d_plot["CHW load (RT)"], d_plot["Chiller efficiency (kW/RT)"], "Chiller efficiency vs load"))
    st.pyplot(scatter_plot(d_plot["CHW load (RT)"], d_plot["CHWP efficiency (kW/RT)"], "CHWP efficiency vs load"))
    st.pyplot(scatter_plot(d_plot["CHW load (RT)"], d_plot["CWP efficiency (kW/RT)"], "CWP efficiency vs load"))
    st.pyplot(scatter_plot(d_plot["CHW load (RT)"], d_plot["CT efficiency (kW/RT)"], "CT efficiency vs load"))
    st.pyplot(scatter_plot(d_plot["CHW load (RT)"], d_plot["Overall efficiency (kW/RT)"], "Chiller plant efficiency vs load", ylabel="Plant kW/RT"))


# -----------------------------
# AI
# -----------------------------
with tab4:
    st.subheader("AI Anomaly Detection")

    if not enable_ai:
        st.info("AI is disabled.")
    else:
        if oat_col == "(none)" or rh_col == "(none)":
            st.error("AI requires OAT (dry bulb) and RH columns. Map them on the main page.")
            st.stop()

        ai = df_calc.copy()

        ai["_db"] = to_num(ai[oat_col])
        ai["_rh"] = to_num(ai[rh_col])
        ai["_heat_index_proxy"] = ai["_db"] + 0.1 * ai["_rh"]
        ai["_rh_frac"] = ai["_rh"] / 100.0

        ai["_load_rt"] = to_num(ai["CHW load (RT)"]).replace(0, np.nan)

        ai["_chiller_kw"] = to_num(ai[chiller_kw_col])
        ai["_chwp_kw"] = to_num(ai[chwp_kw_col])
        ai["_cwp_kw"] = to_num(ai[cwp_kw_col])
        ai["_ct_kw"] = to_num(ai[ct_kw_col])

        ai["Chiller_kWRT"] = ai["_chiller_kw"] / ai["_load_rt"]
        ai["CHWP_kWRT"] = ai["_chwp_kw"] / ai["_load_rt"]
        ai["CWP_kWRT"] = ai["_cwp_kw"] / ai["_load_rt"]
        ai["CT_kWRT"] = ai["_ct_kw"] / ai["_load_rt"]

        ai["_plant_kw"] = ai["_chiller_kw"] + ai["_chwp_kw"] + ai["_cwp_kw"] + ai["_ct_kw"]
        ai["Plant_kWRT"] = ai["_plant_kw"] / ai["_load_rt"]

        AI_FEATURES = [
            "_load_rt",
            "_db", "_rh", "_heat_index_proxy",
            "Plant_kWRT",
            "Chiller_kWRT", "CHWP_kWRT", "CWP_kWRT", "CT_kWRT",
            "_chiller_kw", "_chwp_kw", "_cwp_kw", "_ct_kw",
        ]

        X = ai[AI_FEATURES].replace([np.inf, -np.inf], np.nan).dropna()

        cont = 0.01 if len(X) >= 2000 else 0.02
        iso = IsolationForest(n_estimators=500, contamination=cont, random_state=42)

        ai["AI_iforest_flag"] = 0
        ai.loc[X.index, "AI_iforest_flag"] = (iso.fit_predict(X) == -1).astype(int)

        ai["AI_iforest_score"] = np.nan
        ai.loc[X.index, "AI_iforest_score"] = iso.decision_function(X)

        BASE_FEATS = ["_load_rt", "_db", "_rh", "_heat_index_proxy"]
        M = ai[BASE_FEATS].replace([np.inf, -np.inf], np.nan)
        Y = ai["Plant_kWRT"].replace([np.inf, -np.inf], np.nan)

        mask = M.notna().all(axis=1) & Y.notna()
        M2, Y2 = M[mask], Y[mask]

        split = int(len(M2) * 0.7) if len(M2) > 20 else max(1, int(len(M2) * 0.5))
        M_train, Y_train = M2.iloc[:split], Y2.iloc[:split]

        gbr = GradientBoostingRegressor(random_state=42)
        gbr.fit(M_train, Y_train)

        ai["AI_expected_Plant_kWRT"] = np.nan
        ai.loc[mask, "AI_expected_Plant_kWRT"] = gbr.predict(M2)

        ai["AI_residual_Plant_kWRT"] = ai["Plant_kWRT"] - ai["AI_expected_Plant_kWRT"]

        res = ai.loc[mask, "AI_residual_Plant_kWRT"].dropna()
        robust_sigma = 1.4826 * (res - res.median()).abs().median() if len(res) else np.nan
        thr = max(0.05, 3 * robust_sigma) if np.isfinite(robust_sigma) else 0.08

        ai["AI_baseline_flag"] = ((ai["AI_residual_Plant_kWRT"] > thr) & ai["AI_expected_Plant_kWRT"].notna()).astype(int)

        def cause_tag(row):
            parts = {
                "Chiller": row.get("Chiller_kWRT", np.nan),
                "CHWP": row.get("CHWP_kWRT", np.nan),
                "CWP": row.get("CWP_kWRT", np.nan),
                "CT": row.get("CT_kWRT", np.nan),
            }
            parts = {k: v for k, v in parts.items() if pd.notna(v)}
            if not parts:
                return "Unknown"
            if pd.notna(row.get("_load_rt")) and row["_load_rt"] < 80:
                return "Low-load / staging"
            dom = max(parts, key=parts.get)
            return f"{dom} driven"

        ai["AI_cause"] = ai.apply(cause_tag, axis=1)

        ai["AI_flag"] = ((ai["AI_iforest_flag"] == 1) | (ai["AI_baseline_flag"] == 1)).astype(int)

        ranked = ai[ai["AI_flag"] == 1].copy()
        ranked["severity"] = ranked["AI_residual_Plant_kWRT"].fillna(0) + (-ranked["AI_iforest_score"].fillna(0))

        st.caption(
            f"Rows used for IForest: {len(X):,} | IForest flagged: {int(ai['AI_iforest_flag'].sum()):,} | "
            f"Baseline threshold (kW/RT): {thr:.4f} | Baseline flagged: {int(ai['AI_baseline_flag'].sum()):,} | "
            f"Total flagged: {int(ai['AI_flag'].sum()):,}"
        )

        show_cols = [
            dt_col, "CHW load (RT)", oat_col, rh_col,
            "Plant_kWRT", "AI_expected_Plant_kWRT", "AI_residual_Plant_kWRT",
            "AI_iforest_flag", "AI_baseline_flag", "AI_cause", "severity"
        ]
        ranked_sorted = ranked.sort_values("severity", ascending=False)[show_cols]
        st.dataframe(ranked_sorted.head(200), use_container_width=True)

        st.download_button(
            "Download AI flagged events CSV",
            data=ranked_sorted.to_csv(index=False).encode("utf-8"),
            file_name="AI_flagged_events.csv",
            mime="text/csv",
        )

        flag = ai["AI_flag"].fillna(0).astype(int)
        st.pyplot(scatter_flagged(ai["CHW load (RT)"], ai["Chiller_kWRT"], flag, "Chiller kW/RT vs Load (AI flagged)", "Chiller kW/RT"))
        st.pyplot(scatter_flagged(ai["CHW load (RT)"], ai["CHWP_kWRT"], flag, "CHWP kW/RT vs Load (AI flagged)", "CHWP kW/RT"))
        st.pyplot(scatter_flagged(ai["CHW load (RT)"], ai["CWP_kWRT"], flag, "CWP kW/RT vs Load (AI flagged)", "CWP kW/RT"))
        st.pyplot(scatter_flagged(ai["CHW load (RT)"], ai["CT_kWRT"], flag, "CT kW/RT vs Load (AI flagged)", "CT kW/RT"))
        st.pyplot(scatter_flagged(ai["CHW load (RT)"], ai["Plant_kWRT"], flag, "Plant kW/RT vs Load (AI flagged)", "Plant kW/RT"))

# ==========================================================
# ENERGY WASTE ESTIMATION
# ==========================================================
st.divider()
st.subheader("💸 Energy Waste Estimation")

TARIFF = st.number_input("Tariff (SGD/kWh)", value=0.25)
MIN_RT = st.number_input("Minimum RT to count waste", value=200.0)

df_out = ai.copy()

df_out["AI_flag_any"] = ((df_out["AI_iforest_flag"] == 1) | 
                         (df_out["AI_baseline_flag"] == 1)).astype(int)

df_out["AI_actual_kW"] = df_out["Plant_kWRT"] * df_out["CHW load (RT)"]
df_out["AI_expected_kW"] = df_out["AI_expected_Plant_kWRT"] * df_out["CHW load (RT)"]

df_out["AI_waste_kW"] = (df_out["AI_actual_kW"] - df_out["AI_expected_kW"]).clip(lower=0)

# ignore tiny loads
df_out.loc[df_out["CHW load (RT)"] < MIN_RT, "AI_waste_kW"] = 0
df_out.loc[df_out["AI_flag_any"] != 1, "AI_waste_kW"] = 0

dt_hours = float(sampling_min) / 60.0

df_out["AI_waste_kWh"] = df_out["AI_waste_kW"] * dt_hours
df_out["AI_waste_cost_SGD"] = df_out["AI_waste_kWh"] * TARIFF

total_kwh = df_out["AI_waste_kWh"].sum()
total_cost = df_out["AI_waste_cost_SGD"].sum()

c1, c2 = st.columns(2)
c1.metric("Avoidable Waste (kWh)", f"{total_kwh:,.1f}")
c2.metric("Avoidable Cost (SGD)", f"{total_cost:,.2f}")

# Top events
top_events = df_out[df_out["AI_waste_kWh"] > 0] \
                .sort_values("AI_waste_kWh", ascending=False) \
                .head(20)

st.dataframe(top_events[[dt_col,
                         "CHW load (RT)",
                         "AI_waste_kWh",
                         "AI_waste_cost_SGD",
                         "AI_cause"]])

# Cause summary
cause_summary = df_out[df_out["AI_waste_kWh"] > 0] \
    .groupby("AI_cause") \
    .agg(events=("AI_waste_kWh", "size"),
         waste_kWh=("AI_waste_kWh", "sum"),
         cost_SGD=("AI_waste_cost_SGD", "sum")) \
    .sort_values("waste_kWh", ascending=False) \
    .reset_index()

st.subheader("Pareto by Cause")
st.dataframe(cause_summary)

pareto_fig = plot_pareto_cause(cause_summary)
if pareto_fig:
    st.pyplot(pareto_fig)

# Daily trend
daily_tbl, daily_fig1, daily_fig2 = plot_daily_waste(df_out, dt_col)

st.subheader("Daily Waste Trend")
st.dataframe(daily_tbl)
st.pyplot(daily_fig1)
st.pyplot(daily_fig2)

# store for ZIP
st.session_state["df_out"] = df_out
st.session_state["top_events"] = top_events
st.session_state["cause_summary"] = cause_summary
st.session_state["daily_tbl"] = daily_tbl
st.session_state["pareto_fig"] = pareto_fig
st.session_state["daily_fig1"] = daily_fig1
st.session_state["daily_fig2"] = daily_fig2

# Download enriched AI dataframe
st.download_button(
    "Download AI + waste columns CSV",
    data=df_out.to_csv(index=False).encode("utf-8"),
    file_name="AI_with_waste.csv",
    mime="text/csv",
)
    
# -----------------------------
# Downloads
# -----------------------------
st.divider()
st.subheader("Downloads")

# 1) df_calc download
st.download_button(
    "Download df_calc (processed CSV)",
    data=df_calc.to_csv(index=False).encode("utf-8"),
    file_name=f"df_calc_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)

# 2) ZIP report (plots + key tables + AI waste outputs)
with tempfile.TemporaryDirectory() as tmpdir:
    outdir = os.path.join(tmpdir, "bms_report")
    os.makedirs(outdir, exist_ok=True)

    # --- Save key tables ---
    df_calc.head(5000).to_csv(os.path.join(outdir, "df_calc_head.csv"), index=False)

    # --- Build plots FIRST (so figs exists) ---
    figs = []
    figs.append(
        plot_hourly_line(
            hourly_mean(df_m, dt_col, "CHW load (RT)"),
            f"{sel_month}: Typical 24-Hour Cooling Load Profile",
            "RT",
        )
    )
    figs.append(
        plot_hourly_line(
            hourly_mean(df_calc, dt_col, "Overall efficiency (kW/RT)"),
            "Plant Efficiency vs time (Daily Profile)",
            "kW/RT",
        )
    )
    figs.append(
        scatter_plot(
            df_calc["CHW load (RT)"],
            df_calc["Overall efficiency (kW/RT)"],
            "Plant Efficiency vs Load",
            ylabel="kW/RT",
        )
    )

    # --- Save plots ---
    for i, fig in enumerate(figs, start=1):
        save_fig(fig, os.path.join(outdir, f"plot_{i:02d}.png"))

    # --- Save AI waste outputs into ZIP (only if they exist in session_state) ---
    df_out_ss = st.session_state.get("df_out", None)
    if isinstance(df_out_ss, pd.DataFrame) and len(df_out_ss) > 0:
        df_out_ss.to_csv(os.path.join(outdir, "AI_with_waste.csv"), index=False)

        top_events_ss = st.session_state.get("top_events", None)
        if isinstance(top_events_ss, pd.DataFrame) and len(top_events_ss) > 0:
            top_events_ss.to_csv(os.path.join(outdir, "AI_top_events.csv"), index=False)

        cause_summary_ss = st.session_state.get("cause_summary", None)
        if isinstance(cause_summary_ss, pd.DataFrame) and len(cause_summary_ss) > 0:
            cause_summary_ss.to_csv(os.path.join(outdir, "AI_cause_summary.csv"), index=False)

        daily_tbl_ss = st.session_state.get("daily_tbl", None)
        if isinstance(daily_tbl_ss, pd.DataFrame) and len(daily_tbl_ss) > 0:
            daily_tbl_ss.to_csv(os.path.join(outdir, "AI_daily_waste.csv"), index=False)

        pareto_fig_ss = st.session_state.get("pareto_fig", None)
        if pareto_fig_ss is not None:
            save_fig(pareto_fig_ss, os.path.join(outdir, "AI_pareto.png"))

        daily_fig1_ss = st.session_state.get("daily_fig1", None)
        if daily_fig1_ss is not None:
            save_fig(daily_fig1_ss, os.path.join(outdir, "AI_daily_kWh.png"))

        daily_fig2_ss = st.session_state.get("daily_fig2", None)
        if daily_fig2_ss is not None:
            save_fig(daily_fig2_ss, os.path.join(outdir, "AI_daily_cost.png"))

    # --- ZIP download button MUST be last, still inside tempdir ---
    st.download_button(
        "Download report ZIP (tables + key plots + AI waste outputs)",
        data=zip_folder(outdir),
        file_name=f"bms_report_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
        mime="application/zip",
    )
