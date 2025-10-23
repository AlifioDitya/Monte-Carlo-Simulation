# app.py — Goal Projection Calculator (Streamlit)
# Run:  streamlit run app.py
# Deps: pip install streamlit numpy pandas matplotlib

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timezone, timedelta

# ---------- Page setup ----------
st.set_page_config(
    page_title="Goal Projection Calculator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal theming
st.markdown("""
<style>
/* Subtle card-like look for sections */
.block-container { padding-top: 1.5rem; }
div[data-testid="stMetric"] { background: #f7f8fa; border-radius: 14px; padding: 12px; }
hr { margin: 0.8rem 0 1.2rem 0; }
.small-note { color: #6b7280; font-size: 0.9rem; }
.caption { color: #6b7280; font-size: 0.85rem; }
h1, h2, h3 { letter-spacing: .2px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* --- KPI cards (light mode) --- */
div[data-testid="stMetric"] {
  background-color: #f1f5f9 !important;   /* slate-100 */
  border: 1px solid #e2e8f0;              /* slate-200 */
  border-radius: 12px;
  padding: 12px 16px;
  box-shadow: 0 1px 2px rgba(0,0,0,.03);
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: #0f172a !important;              /* slate-900 */
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  color: #475569 !important;              /* slate-600 */
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
  color: #2563eb;                          /* keep delta readable */
}

/* --- KPI cards (dark mode) --- */
@media (prefers-color-scheme: dark) {
  div[data-testid="stMetric"] {
    background-color: #1f2937 !important; /* slate-800 */
    border-color: #334155;                /* slate-700 */
  }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;            /* slate-200 */
  }
  div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #94a3b8 !important;            /* slate-400 */
  }
}
</style>
""", unsafe_allow_html=True)


# ---------- Data classes ----------
@dataclass
class GoalParams:
    W0: float
    target_FV: float
    horizon_months: int
    contributions: float | Sequence[float]  # start-of-month; scalar or array length n

@dataclass
class MarketParams:
    # Either monthly lognormal params...
    m_month: Optional[float] = None
    s_month: Optional[float] = None
    # ...or annual arithmetic assumptions (converted internally)
    annual_return: Optional[float] = None
    annual_vol: Optional[float] = None
    # Optional: if user inputs CAGR instead of arithmetic, we convert internally
    use_cagr: bool = False
    cagr: Optional[float] = None

@dataclass
class SimulationConfig:
    paths: int = 5000
    seed: Optional[int] = 12345
    store_paths: bool = True             # needed for projection band
    max_paths_for_storage: int = 20000   # memory guardrail

# ---------- Helpers ----------
def _to_monthly_params(market: MarketParams) -> Tuple[float, float, float]:
    """Return (m, s, a) where:
       m = monthly log-mean; s = monthly log-std; a = E[G]_month (expected monthly gross).
    """
    # Direct monthly params
    if market.m_month is not None and market.s_month is not None:
        m = float(market.m_month)
        s = float(market.s_month)
        a = float(np.exp(m + 0.5 * s * s))
        return m, s, a

    # Annual to monthly conversion
    if market.use_cagr and market.cagr is not None and market.annual_vol is not None:
        # Convert CAGR (geometric) to arithmetic using lognormal relationship
        r_arith = (1.0 + float(market.cagr)) * math.exp(0.5 * float(market.annual_vol) ** 2) - 1.0
        annual_return = r_arith
        annual_vol = float(market.annual_vol)
    else:
        if market.annual_return is None or market.annual_vol is None:
            raise ValueError("Provide (m_month,s_month) OR (annual_return,annual_vol) OR (cagr,annual_vol + toggle).")
        annual_return = float(market.annual_return)
        annual_vol = float(market.annual_vol)

    E_G_month = float((1.0 + annual_return) ** (1.0 / 12.0))
    s_month = annual_vol / math.sqrt(12.0)
    m_month = float(np.log(E_G_month) - 0.5 * (s_month ** 2))
    return m_month, s_month, E_G_month

def _normalize_contributions(contribs: float | Sequence[float], n: int) -> np.ndarray:
    if isinstance(contribs, (int, float)):
        return np.full(n, float(contribs), dtype=float)
    arr = np.asarray(contribs, dtype=float).ravel()
    if arr.size != n:
        raise ValueError(f"contributions length must be {n}, got {arr.size}")
    return arr

def now_jakarta_iso() -> str:
    # Asia/Jakarta is UTC+7 (no DST)
    return (datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S WIB")

# ---------- Engine ----------
def simulate_goal_mc(goal: GoalParams, market: MarketParams, cfg: SimulationConfig) -> Dict[str, Any]:
    """Monte Carlo goal simulation with start-of-month contributions.
       Wealth update (t=1..n): W_t = (W_{t-1} + c_t) * G_t,  G_t = exp(m + s * Z_t), Z~N(0,1)
    """
    m, s, E_G_month = _to_monthly_params(market)
    n = int(goal.horizon_months)
    contribs = _normalize_contributions(goal.contributions, n)
    rng = np.random.default_rng(cfg.seed)

    paths = int(cfg.paths)
    store_paths = bool(cfg.store_paths) and (paths <= cfg.max_paths_for_storage)

    if store_paths:
        W = np.zeros((n + 1, paths), dtype=float)
        W[0, :] = goal.W0
        for t in range(1, n + 1):
            Z = rng.standard_normal(paths)
            G = np.exp(m + s * Z)
            W[t, :] = (W[t - 1, :] + contribs[t - 1]) * G
        end_vals = W[-1, :].copy()
    else:
        # Chunked simulation (still retains full matrix to compute quantiles)
        W_accum = []
        for start in range(0, paths, 2000):
            take = min(2000, paths - start)
            W_chunk = np.zeros((n + 1, take), dtype=float)
            W_chunk[0, :] = goal.W0
            for t in range(1, n + 1):
                Z = rng.standard_normal(take)
                G = np.exp(m + s * Z)
                W_chunk[t, :] = (W_chunk[t - 1, :] + contribs[t - 1]) * G
            W_accum.append(W_chunk)
        W = np.concatenate(W_accum, axis=1)
        end_vals = W[-1, :]

    prob = float(np.mean(end_vals >= goal.target_FV))
    p10, p50, p90 = np.percentile(end_vals, [10, 50, 90])
    q10 = np.percentile(W, 10, axis=1)
    q50 = np.percentile(W, 50, axis=1)
    q90 = np.percentile(W, 90, axis=1)

    # Sample up to 200 paths for plotting
    paths_sample = W[:, : min(200, W.shape[1])].copy()

    return {
        "as_of": now_jakarta_iso(),
        "as_of_params": {
            "W0": float(goal.W0),
            "target_FV": float(goal.target_FV),
            "horizon_months": n,
            "contributions_sum": float(contribs.sum()),
            "m_month": float(m),
            "s_month": float(s),
            "E_G_month": float(E_G_month),
            "paths": paths,
            "seed": cfg.seed,
        },
        "end_values": end_vals,
        "prob_completion": prob,
        "percentiles": {"p10": float(p10), "p50": float(p50), "p90": float(p90)},
        "time_grid": np.arange(n + 1, dtype=int),
        "projection_band": {"q10": q10, "q50": q50, "q90": q90},
        "paths_sample": paths_sample,
    }

def fenton_wilkinson_probability(goal: GoalParams, market: MarketParams) -> Dict[str, float]:
    """Analytic probability via Fenton–Wilkinson moment matching on end wealth W_n."""
    m, s, _ = _to_monthly_params(market)
    a = float(np.exp(m + 0.5 * s * s))
    b = float(np.exp(2.0 * m + 2.0 * s * s))

    n = int(goal.horizon_months)
    contribs = _normalize_contributions(goal.contributions, n)

    # Mean
    mu1 = float(goal.W0 * (a ** n) + sum(contribs[t] * (a ** (n - t)) for t in range(n)) * a)

    # Second moment
    mu2 = float((goal.W0 ** 2) * (b ** n))
    cross = 0.0
    for k in range(1, n + 1):
        cross += (b ** n) * (a ** (k - 1)) * contribs[k - 1]
    mu2 += 2.0 * goal.W0 * cross

    ssum = 0.0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            ssum += contribs[i - 1] * contribs[j - 1] * (b ** (n - max(i, j) + 1)) * (a ** abs(i - j))
    mu2 += ssum

    var = max(mu2 - mu1 * mu1, 0.0)
    if mu1 <= 0.0:
        return {"mu1": mu1, "var": var, "p": 0.0}
    tilde_sigma2 = float(np.log(1.0 + var / (mu1 * mu1))) if var > 0 else 0.0
    tilde_sigma = float(np.sqrt(max(tilde_sigma2, 0.0)))
    tilde_mu = float(np.log(mu1) - 0.5 * tilde_sigma2) if tilde_sigma2 > 0 else float(np.log(mu1))
    if tilde_sigma > 0:
        z = (np.log(goal.target_FV) - tilde_mu) / tilde_sigma
        p = 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    else:
        p = 1.0 if mu1 >= goal.target_FV else 0.0
    return {"mu1": mu1, "var": var, "p": p}

def deterministic_end_wealth(goal: GoalParams, market: MarketParams) -> float:
    """Deterministic end wealth using expected monthly gross a (no volatility)."""
    m, s, _ = _to_monthly_params(market)
    a = float(np.exp(m + 0.5 * s * s))
    n = int(goal.horizon_months)
    contribs = _normalize_contributions(goal.contributions, n)
    Wn = goal.W0 * (a ** n)
    for t in range(n):
        Wn += contribs[t] * (a ** (n - t)) * a  # start-of-month contribution
    return float(Wn)

# ---------- Plotly figures (interactive) ----------
def fig_projection_band(res, target_fv):
    t   = res["time_grid"]
    q10 = res["projection_band"]["q10"]
    q50 = res["projection_band"]["q50"]
    q90 = res["projection_band"]["q90"]

    fig = go.Figure()

    # P10–P90 fill
    fig.add_trace(go.Scatter(x=t, y=q10, mode="lines",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=t, y=q90, mode="lines",
                             line=dict(width=0), fill="tonexty",
                             name="P10–P90 band",
                             hovertemplate="Month %{x}<br>Wealth %{y:,.0f}<extra>P10–P90</extra>"))

    # Median
    fig.add_trace(go.Scatter(x=t, y=q50, mode="lines",
                             name="Median (P50)", line=dict(width=2)))

    # >>> Accented Target line
    fig.add_hline(
        y=target_fv, line_dash="dash", line_color="#e11d48", line_width=3,  # crimson-ish
        annotation_text="Target at horizon", annotation_position="top left",
        annotation=dict(font_color="#ff0e42", bgcolor="#182231", yshift=12)
    )

    fig.update_layout(
        title="Projection Band (Monthly)",
        xaxis_title="Month", yaxis_title="Wealth (IDR)",
        hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10)
    )
    fig.update_xaxes(rangeslider=dict(visible=True))
    return fig


def fig_simulated_paths(res, target_fv):
    t = res["time_grid"]
    W = res["paths_sample"]                # (n+1, <=200)
    q10 = res["projection_band"]["q10"]
    q50 = res["projection_band"]["q50"]
    q90 = res["projection_band"]["q90"]

    fig = go.Figure()

    # Faint individual paths (no hover so cursor "follows" median only)
    for k in range(W.shape[1]):
        fig.add_trace(go.Scattergl(
            x=t, y=W[:, k], mode="lines", showlegend=False,
            line=dict(width=1, color="#94a3b8"),  # slate-400
            opacity=0.18, hoverinfo="skip"
        ))

    # Highlighted median path with custom tooltip containing P90/P50/P10 at that month
    custom = np.column_stack([q90, q50, q10])  # High, Median, Low
    fig.add_trace(go.Scatter(
        x=t, y=q50, mode="lines+markers", name="Median (P50)",
        line=dict(width=3, color="#2563eb"), marker=dict(size=3, color="#2563eb"),
        customdata=custom,
        hovertemplate=(
            "Month %{x}<br>"
            "High (P90) %{customdata[0]:,.0f}<br>"
            "<b>Median (P50) %{y:,.0f}</b><br>"
            "Low (P10) %{customdata[2]:,.0f}<extra></extra>"
        )
    ))

    # Accented Target
    fig.add_hline(
        y=target_fv, line_dash="dash", line_color="#e11d48", line_width=3,
        annotation_text="Target", annotation_position="top left",
        annotation=dict(font_color="#ff0e42", bgcolor="#182231", yshift=12)
    )

    fig.update_layout(
        title=f"Simulated Wealth Paths (sample {W.shape[1]})",
        xaxis_title="Month", yaxis_title="Wealth (IDR)",
        hovermode="x", margin=dict(l=10, r=10, t=40, b=10)
    )
    fig.update_xaxes(rangeslider=dict(visible=True))
    return fig



def fig_end_wealth_hist(res, target_fv):
    vals = res["end_values"]
    p    = res["prob_completion"]
    p10, p50, p90 = res["percentiles"]["p10"], res["percentiles"]["p50"], res["percentiles"]["p90"]

    fig = px.histogram(x=vals, nbins=40, labels={"x": "End Wealth (IDR)", "y": "Count"})
    fig.update_traces(hovertemplate="End wealth %{x:,.0f}<br>Count %{y}<extra></extra>")

    # >>> Accented Target (left) with offset so it doesn't clash with P50 label
    fig.add_vline(
      x=target_fv, line_dash="dash", line_color="#e11d48", line_width=3,
      annotation=dict(
        text="Target",
        x=target_fv, xref="x", xanchor="center",
        y=1.0, yref="paper", yanchor="bottom",
        font_color="#ff0e42", bgcolor="#182231", yshift=-16
      )
    )

    # P50/P10/P90 guides (more subdued + positioned away)
    fig.add_vline(
      x=p50,
      line=dict(color="#ffff12", dash="dot"),  # slate-500
      annotation=dict(
        text="P50",
        x=p50, xref="x", xanchor="center",
        y=1.0, yref="paper", yanchor="bottom",
        font_color="#ffff12", bgcolor="#182231", yshift=-36
      )
    )
    fig.add_vline(x=p10, line=dict(color="#94a3b8", dash="dot"),
                  annotation_text="P10", annotation_position="bottom right",
                  annotation=dict(font_color="#FFFFFF", bgcolor="#182231", yshift=-12))
    fig.add_vline(x=p90, line=dict(color="#94a3b8", dash="dot"),
                  annotation_text="P90", annotation_position="bottom left",
                  annotation=dict(font_color="#FFFFFF", bgcolor="#182231", yshift=-12))

    fig.update_layout(
        title=f"End Wealth Distribution — P(≥Target) = {p*100:.1f}%",
        bargap=0.05, margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

# ---------- Sidebar inputs ----------
st.sidebar.header("Configuration")

st.sidebar.subheader("Goal")
W0 = st.sidebar.number_input("Starting NAV (IDR)", min_value=0.0, value=125_000_000.0, step=1_000_000.0, format="%.0f")
target = st.sidebar.number_input("Target at horizon (IDR)", min_value=0.0, value=305_000_000.0, step=1_000_000.0, format="%.0f")
horizon = st.sidebar.slider("Horizon (months)", min_value=6, max_value=360, value=36, step=1)
contrib = st.sidebar.number_input("Monthly contribution (start-of-month, IDR)", min_value=0.0, value=3_000_000.0, step=500_000.0, format="%.0f")

st.sidebar.subheader("Market assumptions")
mode = st.sidebar.radio("Return convention", ["Arithmetic annual return", "CAGR (geometric)"], index=0, horizontal=False)
if mode == "Arithmetic annual return":
    annual_return = st.sidebar.number_input("Arithmetic annual return", min_value=-0.8, max_value=1.5, value=0.094, step=0.005, format="%.3f")
    cagr = None
    use_cagr = False
else:
    cagr = st.sidebar.number_input("CAGR (geometric)", min_value=-0.8, max_value=1.5, value=0.090, step=0.005, format="%.3f")
    annual_return = None
    use_cagr = True
annual_vol = st.sidebar.number_input("Annual volatility (stdev)", min_value=0.0, max_value=3.0, value=0.18, step=0.01, format="%.3f")

st.sidebar.subheader("Simulation")
paths = st.sidebar.slider("Monte Carlo paths", min_value=1000, max_value=20000, value=5000, step=1000)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=2**32-1, value=12345, step=1)
store_paths = st.sidebar.checkbox("Store all paths (needed for projection band)", value=True)

st.sidebar.markdown("---")
preset = st.sidebar.selectbox("Quick presets", ["None", "Higher contribution (3.7M)", "Longer horizon (+6m)"])
if preset == "Higher contribution (3.7M)":
    contrib = 3_700_000.0
elif preset == "Longer horizon (+6m)":
    horizon = horizon + 6

# Sidebar toggle
st.sidebar.markdown("---")
st.sidebar.subheader("Plot options")
log_y = st.sidebar.checkbox("Log scale (Y axis for line charts)", value=False)
log_y_hist = st.sidebar.checkbox("Log scale (Y axis for histogram counts)", value=False)


# ---------- Compute ----------
goal = GoalParams(W0=W0, target_FV=target, horizon_months=horizon, contributions=contrib)
market = MarketParams(
    annual_return=annual_return, annual_vol=annual_vol, use_cagr=use_cagr, cagr=cagr
)
cfg = SimulationConfig(paths=paths, seed=seed, store_paths=store_paths)

res = simulate_goal_mc(goal, market, cfg)
fw = fenton_wilkinson_probability(goal, market)
W_det = deterministic_end_wealth(goal, market)

# ---------- Header ----------
st.title("📈 Goal Projection Calculator")
st.caption("Monte Carlo projection with start-of-month contributions • Streamlit demo")

# ---------- KPI grid (2 rows, readable labels) ----------
row1 = st.columns(2)
with row1[0]:
    st.metric("Probability (Monte Carlo)", f"{res['prob_completion']*100:.1f} %")
with row1[1]:
    st.metric("Probability (Analytic — Fenton-Wilkinson)", f"{fw['p']*100:.1f} %")

row2 = st.columns(2)
with row2[0]:
    st.metric(
        "Deterministic End Wealth (mean-return, no vol)",
        f"Rp {res['end_values'].mean():,.0f}".replace(",", ".")
    )
with row2[1]:
    st.metric("Median (P50) End Wealth", f"Rp {res['percentiles']['p50']:,.0f}".replace(",", "."))

row3 = st.columns(2)
with row3[0]:
    st.metric("Starting NAV (W0)", f"Rp {W0:,.0f}".replace(",", "."))
with row3[1]:
    st.metric("Target @ Horizon", f"Rp {target:,.0f}".replace(",", "."))


# Format "As of" as "MonthName Year" (e.g. "October 2025")
as_of_raw = res.get("as_of", "")
as_of_fmt = as_of_raw
try:
  ts = as_of_raw.replace(" WIB", "")  # strip timezone token
  dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
  as_of_fmt = dt.strftime("%B %Y")
except Exception:
  pass

st.markdown(
  f"<div class='small-note'>As of {as_of_fmt} • Using {'CAGR→Arithmetic' if use_cagr else 'Arithmetic annual return'} "
  f"with σ={annual_vol:.1%} • m={res['as_of_params']['m_month']:.5f}, s={res['as_of_params']['s_month']:.5f}</div>",
  unsafe_allow_html=True
)
st.markdown("---")

# --- Projection band (interactive) ---
st.subheader("Projection band (P10-P50-P90) to horizon")
fig1 = fig_projection_band(res, target)
if log_y:
    fig1.update_yaxes(type="log")
st.plotly_chart(fig1, use_container_width=True)
st.markdown("""
**Interpretation:** The band shows the distribution of possible wealth by month. 
P50 is the base case; P10 pessimistic; P90 optimistic.
""")

# --- Simulated wealth paths (interactive) ---
st.subheader("Simulated wealth paths (sample)")
fig2 = fig_simulated_paths(res, target)
if log_y:
    fig2.update_yaxes(type="log")
st.plotly_chart(fig2, use_container_width=True)
st.markdown("""
**Interpretation:** Each line is one simulated future. Early drawdowns hurt compounding (sequence-of-returns risk).
""")

# --- End-wealth distribution (interactive) ---
st.subheader("End-wealth distribution at horizon")
fig3 = fig_end_wealth_hist(res, target)
if log_y_hist:
    fig3.update_yaxes(type="log")  # affects the histogram's count axis
st.plotly_chart(fig3, use_container_width=True)
st.markdown("""
**Interpretation:** The fraction to the right of the target line equals the **probability of completion**.
""")


# ---------- Data outputs & downloads ----------
st.markdown("---")
st.subheader("Outputs & downloads")

# Summary table
summary = pd.DataFrame({
    "Metric": [
        "As-of", "Paths", "Horizon (mo)", "W0", "Sum(contrib)", "Target",
        "E[G]_month", "m_month", "s_month",
        "P10 end", "P50 end", "P90 end",
        "Prob Completion (MC)", "Prob (FW Analytic)", "Deterministic End"
    ],
    "Value": [
        res["as_of"], res["as_of_params"]["paths"], res["as_of_params"]["horizon_months"],
        f"Rp {res['as_of_params']['W0']:,.0f}".replace(",", "."),
        f"Rp {res['as_of_params']['contributions_sum']:,.0f}".replace(",", "."),
        f"Rp {res['as_of_params']['target_FV']:,.0f}".replace(",", "."),
        f"{(res['as_of_params']['E_G_month'] - 1) * 100:.3f} %",
        f"{res['as_of_params']['m_month']:.5f}",
        f"{res['as_of_params']['s_month']:.5f}",
        f"Rp {res['percentiles']['p10']:,.0f}".replace(",", "."),
        f"Rp {res['percentiles']['p50']:,.0f}".replace(",", "."),
        f"Rp {res['percentiles']['p90']:,.0f}".replace(",", "."),
        f"{res['prob_completion']*100:.1f} %",
        f"{fw['p']*100:.1f} %",
        f"Rp {W_det:,.0f}".replace(",", "."),
    ]
})
st.dataframe(summary, use_container_width=True)

# CSVs
band_df = pd.DataFrame({
  "month": res["time_grid"].tolist(),
  "q10": res["projection_band"]["q10"].tolist(),
  "q50": res["projection_band"]["q50"].tolist(),
  "q90": res["projection_band"]["q90"].tolist(),
})
end_df = pd.DataFrame({"end_wealth": res["end_values"]})

colA, colB = st.columns(2)
with colA:
    st.download_button(
        "⬇️ Download projection band (CSV)",
        data=band_df.to_csv(index=False).encode("utf-8"),
        file_name="projection_band.csv",
        mime="text/csv",
    )
with colB:
    st.download_button(
        "⬇️ Download end-wealth distribution (CSV)",
        data=end_df.to_csv(index=False).encode("utf-8"),
        file_name="end_wealth_distribution.csv",
        mime="text/csv",
    )

st.markdown(
    "<div class='caption'>Educational only; not investment advice. Results depend on inputs and assumptions.</div>",
    unsafe_allow_html=True
)
