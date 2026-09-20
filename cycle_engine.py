import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, hilbert
import plotly.graph_objects as go

st.set_page_config(page_title="Cycle Engine Terminal", layout="wide")

st.markdown("""
<style>
.stApp { background-color:#070A0F; color:#EAEAEA; }
section[data-testid="stSidebar"] { background-color:#11131C; }
div[data-testid="stMetric"] {
    background-color:#111827; padding:18px; border-radius:12px; border:1px solid #1f2937;
}
.main-title { text-align:center; font-size:42px; font-weight:800; margin-bottom:0; }
.sub-title { text-align:center; color:#8b949e; font-size:13px; margin-bottom:25px; }
.cycle-card { background-color:#0B111A; border:1px solid #263241; border-radius:12px; padding:18px; margin-bottom:12px; }
.cycle-label { color:#F6C85F; font-size:12px; font-weight:700; letter-spacing:1.5px; }
.cycle-number { font-size:30px; font-weight:800; color:white; }
.cycle-small { color:#9CA3AF; font-size:12px; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Cycle Terminal")
st.sidebar.caption("Advanced Universal Cycle Engine")

asset = st.sidebar.text_input("Asset", "BTC-USD").upper().strip()

cycle_mode = st.sidebar.radio(
    "Cycle View",
    ["Long Term", "Mid Term", "Short Term", "Composite Mix"]
)

period = st.sidebar.selectbox("Historie", ["1y", "2y", "5y", "max"], index=2)

start_date = st.sidebar.date_input("In-Sample Start", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("In-Sample End", pd.to_datetime("today"))

y_axis_mode = st.sidebar.selectbox("Y-Achse", ["Linear", "Logarithmic"], index=1)

show_tops = st.sidebar.checkbox("Cycle Tops anzeigen", True)
show_bottoms = st.sidebar.checkbox("Cycle Bottoms anzeigen", True)
show_projection = st.sidebar.checkbox("Future Projection anzeigen", True)
show_spectrum = st.sidebar.checkbox("Cycle Spectrum anzeigen", True)

st.markdown('<div class="main-title">Cycle Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Bandpass + Hilbert phase + stability + composite cycle model</div>',
    unsafe_allow_html=True
)

@st.cache_data(ttl=3600)
def load_data(symbol, period_value):
    df = yf.download(symbol, period=period_value, interval="1d", progress=False, auto_adjust=True)

    if df.empty and "-" not in symbol and symbol in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE"]:
        df = yf.download(f"{symbol}-USD", period=period_value, interval="1d", progress=False, auto_adjust=True)

    return df

data = load_data(asset, period)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

if data.empty:
    st.error("Keine Daten gefunden. Beispiele: BTC-USD, SOL-USD, ETH-USD, AAPL, NVDA, TSLA, SPY")
    st.stop()

data = data.reset_index()
data["Date"] = pd.to_datetime(data["Date"])
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
data = data.dropna(subset=["Close"])

data = data[
    (data["Date"] >= pd.to_datetime(start_date)) &
    (data["Date"] <= pd.to_datetime(end_date))
].copy()

if len(data) < 180:
    st.error("Zu wenig Daten. Für gute Cycle-Erkennung bitte mindestens 180 Tageskerzen wählen.")
    st.stop()

close = data["Close"].astype(float).values
n = len(close)

def classify_cycle(period_days):
    if period_days >= 150:
        return "Long"
    elif period_days >= 55:
        return "Medium"
    return "Short"

def safe_log_price(values):
    s = pd.Series(values).replace(0, np.nan).interpolate().bfill().ffill()
    return np.log(s.values)

def detrend_signal(values):
    log_price = safe_log_price(values)
    trend_window = max(40, min(160, len(log_price) // 4))
    trend = pd.Series(log_price).rolling(trend_window, min_periods=10).mean().bfill().ffill().values
    signal = log_price - trend
    signal = signal - np.nanmean(signal)
    return signal

def bandpass_filter(signal, period_days):
    fs = 1.0
    low_period = period_days * 1.45
    high_period = max(8, period_days * 0.65)

    low_freq = 1 / low_period
    high_freq = 1 / high_period

    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq

    low = max(low, 0.001)
    high = min(high, 0.99)

    if low >= high:
        return signal

    try:
        b, a = butter(2, [low, high], btype="band")
        return filtfilt(b, a, signal)
    except Exception:
        return signal

def cycle_fit(signal, period_days):
    t = np.arange(len(signal))
    weights = np.linspace(0.35, 1.0, len(signal))

    s = np.sin(2 * np.pi * t / period_days)
    c = np.cos(2 * np.pi * t / period_days)

    X = np.column_stack([s, c])
    W = np.sqrt(weights)

    beta, _, _, _ = np.linalg.lstsq(X * W[:, None], signal * W, rcond=None)
    fitted = X @ beta

    corr = np.corrcoef(signal, fitted)[0, 1]
    if np.isnan(corr):
        corr = 0

    return fitted, beta, abs(corr)

def stability_score(signal, period_days):
    if len(signal) < 240:
        windows = 3
    else:
        windows = 5

    chunks = np.array_split(signal, windows)
    scores = []

    for chunk in chunks:
        if len(chunk) < max(40, period_days * 0.5):
            continue
        _, _, corr = cycle_fit(chunk, period_days)
        scores.append(corr)

    if len(scores) < 2:
        return 0

    return float(np.mean(scores) * (1 - np.std(scores)))

def phase_quality(filtered_signal):
    try:
        analytic = hilbert(filtered_signal)
        phase = np.unwrap(np.angle(analytic))
        inst_period = 2 * np.pi / np.gradient(phase)
        inst_period = inst_period[np.isfinite(inst_period)]
        if len(inst_period) == 0:
            return 0
        return float(1 / (1 + np.nanstd(inst_period) / (abs(np.nanmean(inst_period)) + 1e-9)))
    except Exception:
        return 0

def calculate_cycle_score(base_signal, close_values, period_days):
    filtered = bandpass_filter(base_signal, period_days)
    fitted, beta, corr_quality = cycle_fit(filtered, period_days)

    power = np.std(filtered) / (np.std(base_signal) + 1e-9)
    phase_q = phase_quality(filtered)
    stability = stability_score(base_signal, period_days)

    quality = (
        0.45 * corr_quality +
        0.25 * min(power, 1.0) +
        0.20 * stability +
        0.10 * phase_q
    )

    amplitude = np.std(close_values) * quality
    impact = amplitude * quality * np.sqrt(period_days)

    return quality, amplitude, impact, beta, filtered, fitted, stability, phase_q

base_signal = detrend_signal(close)

cycle_rows = []
max_cycle = min(420, max(80, len(data) // 2))

for p in np.arange(16, max_cycle, 0.5):
    q, amp, impact, beta, filtered, fitted, stability, phase_q = calculate_cycle_score(base_signal, close, p)

    cycle_rows.append({
        "Period (d)": round(float(p), 1),
        "Band": classify_cycle(p),
        "Amplitude": round(float(amp), 2),
        "Cycle Impact": round(float(impact), 2),
        "Cycle Quality": round(float(q), 3),
        "Stability": round(float(stability), 3),
        "Phase Quality": round(float(phase_q), 3),
        "Score": round(float(impact * q), 3),
    })

spectrum_raw = pd.DataFrame(cycle_rows)
spectrum_raw = spectrum_raw.sort_values("Score", ascending=False).reset_index(drop=True)

selected_cycles = []
min_gap_days = 18

for _, row in spectrum_raw.iterrows():
    period_value = row["Period (d)"]
    if not any(abs(period_value - selected["Period (d)"]) < min_gap_days for selected in selected_cycles):
        selected_cycles.append(row)

spectrum = pd.DataFrame(selected_cycles).reset_index(drop=True)

best_long = spectrum[spectrum["Band"] == "Long"].sort_values("Cycle Quality", ascending=False).head(1)
best_medium = spectrum[spectrum["Band"] == "Medium"].sort_values("Cycle Quality", ascending=False).head(1)
best_short = spectrum[spectrum["Band"] == "Short"].sort_values("Cycle Quality", ascending=False).head(1)

if cycle_mode == "Long Term":
    sidebar_band = "Long"
elif cycle_mode == "Mid Term":
    sidebar_band = "Medium"
elif cycle_mode == "Short Term":
    sidebar_band = "Short"
else:
    sidebar_band = "Long"

top_cycles = (
    spectrum[spectrum["Band"] == sidebar_band]
    .sort_values("Cycle Quality", ascending=False)
    .head(12)
    .copy()
)

cycle_options = [
    f'{row["Period (d)"]:.1f}d · {row["Band"]} · quality {row["Cycle Quality"]:.2f} · stability {row["Stability"]:.2f}'
    for _, row in top_cycles.iterrows()
]

manual_cycle_choice = st.sidebar.selectbox("Cycle manuell auswählen", ["Auto"] + cycle_options)

manual_selected_cycle = None
if manual_cycle_choice != "Auto":
    manual_selected_cycle = float(manual_cycle_choice.split("d")[0])

def selected_from_df(df, fallback=80):
    if df.empty:
        return fallback, 0
    row = df.iloc[0]
    return float(row["Period (d)"]), float(row["Cycle Quality"])

if manual_selected_cycle:
    selected_cycle = manual_selected_cycle
    match = spectrum[spectrum["Period (d)"] == selected_cycle]
    selected_quality = float(match.iloc[0]["Cycle Quality"]) if not match.empty else 0
elif cycle_mode == "Long Term":
    selected_cycle, selected_quality = selected_from_df(best_long, 240)
elif cycle_mode == "Mid Term":
    selected_cycle, selected_quality = selected_from_df(best_medium, 90)
elif cycle_mode == "Short Term":
    selected_cycle, selected_quality = selected_from_df(best_short, 30)
else:
    selected_cycle, selected_quality = selected_from_df(best_long, 240)

def build_cycle_line(period_days):
    q, amp, impact, beta, filtered, fitted, stability, phase_q = calculate_cycle_score(base_signal, close, period_days)

    trend_window = max(5, int(period_days / 4))
    trend = pd.Series(close).rolling(trend_window, min_periods=5).mean().bfill().ffill().values

    cycle_line = trend * np.exp(fitted)

    future_cycles = 3 if period_days >= 150 else 5 if period_days >= 55 else 8
    future_days = int(period_days * future_cycles)

    t_future = np.arange(n, n + future_days)
    future_raw = (
        beta[0] * np.sin(2 * np.pi * t_future / period_days)
        + beta[1] * np.cos(2 * np.pi * t_future / period_days)
    )

    last_trend = trend[-1]
    future_line = last_trend * np.exp(future_raw)

    return cycle_line, future_line, future_days, q, stability, phase_q

cycle_line, future_line, future_days, q_final, stability_final, phase_final = build_cycle_line(selected_cycle)

composite_line = None
composite_future = None

if not best_long.empty and not best_medium.empty and not best_short.empty:
    cycles_for_mix = [
        float(best_long.iloc[0]["Period (d)"]),
        float(best_medium.iloc[0]["Period (d)"]),
        float(best_short.iloc[0]["Period (d)"]),
    ]

    weights = np.array([
        float(best_long.iloc[0]["Cycle Quality"]),
        float(best_medium.iloc[0]["Cycle Quality"]),
        float(best_short.iloc[0]["Cycle Quality"]),
    ])

    weights = weights / (weights.sum() + 1e-9)

    lines = []
    futures = []
    max_future = 0

    for cyc in cycles_for_mix:
        line, fut, fdays, _, _, _ = build_cycle_line(cyc)
        lines.append(line)
        futures.append(fut)
        max_future = max(max_future, len(fut))

    composite_line = np.average(np.vstack(lines), axis=0, weights=weights)

    padded_futures = []
    for fut in futures:
        if len(fut) < max_future:
            fut = np.pad(fut, (0, max_future - len(fut)), mode="edge")
        padded_futures.append(fut)

    composite_future = np.average(np.vstack(padded_futures), axis=0, weights=weights)

if cycle_mode == "Composite Mix" and composite_line is not None:
    plot_cycle = composite_line
    plot_future = composite_future
    future_days = len(plot_future)
    selected_label = "Composite Mix"
else:
    plot_cycle = cycle_line
    plot_future = future_line
    selected_label = f"Cycle {selected_cycle:.1f}d"

data["CycleLine"] = plot_cycle

clean = data.copy()
distance_window = max(8, int(selected_cycle / 2))

peaks, _ = find_peaks(clean["CycleLine"], distance=distance_window)
bottoms, _ = find_peaks(-clean["CycleLine"], distance=distance_window)

clean["Top"] = np.nan
clean["Bottom"] = np.nan
clean.iloc[peaks, clean.columns.get_loc("Top")] = clean.iloc[peaks]["CycleLine"]
clean.iloc[bottoms, clean.columns.get_loc("Bottom")] = clean.iloc[bottoms]["CycleLine"]

current_price = float(data["Close"].iloc[-1])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(asset, f"${current_price:,.2f}")
col2.metric("In-Sample Bars", len(data))
col3.metric("Dominant Period", selected_label if cycle_mode == "Composite Mix" else f"{selected_cycle:.1f}d")
col4.metric("Cycle Quality", f"{selected_quality:.2f}")
col5.metric("Stability", f"{stability_final:.2f}")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data["Date"],
    y=data["Close"],
    mode="lines",
    name=f"{asset} Price",
    line=dict(width=1.3, color="#D9DEE7")
))

fig.add_trace(go.Scatter(
    x=data["Date"],
    y=data["CycleLine"],
    mode="lines",
    name=selected_label,
    line=dict(width=3, color="#3CC6B7")
))

if show_tops:
    fig.add_trace(go.Scatter(
        x=clean["Date"],
        y=clean["Top"],
        mode="markers",
        name="Cycle Tops",
        marker=dict(size=10, symbol="triangle-down", color="#FFCC66")
    ))

if show_bottoms:
    fig.add_trace(go.Scatter(
        x=clean["Date"],
        y=clean["Bottom"],
        mode="markers",
        name="Cycle Bottoms",
        marker=dict(size=10, symbol="triangle-up", color="#8B5CF6")
    ))

if show_projection:
    future_dates = pd.date_range(
        start=data["Date"].iloc[-1],
        periods=len(plot_future),
        freq="D"
    )

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=plot_future,
        mode="lines",
        name="Projected Cycle Path",
        line=dict(width=2, dash="dash", color="#3CC6B7")
    ))

    projection_cycles = 3 if selected_cycle >= 150 else 5 if selected_cycle >= 55 else 8

    for i in range(1, projection_cycles + 1):
        ft = data["Date"].iloc[-1] + pd.Timedelta(days=int(selected_cycle * i))
        fig.add_shape(
            type="line",
            x0=ft,
            x1=ft,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="#8B949E", dash="dot", width=1)
        )
        fig.add_annotation(
            x=ft,
            y=1,
            xref="x",
            yref="paper",
            text="Projected Top",
            showarrow=False,
            yshift=12,
            font=dict(size=10, color="#EAEAEA")
        )

fig.update_layout(
    height=720,
    template="plotly_dark",
    paper_bgcolor="#070A0F",
    plot_bgcolor="#0B111A",
    xaxis_title="Date",
    yaxis_title=f"{asset} Price",
    hovermode="x unified",
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
    margin=dict(l=30, r=30, t=40, b=30)
)

fig.update_xaxes(gridcolor="#1F2937")
fig.update_yaxes(gridcolor="#1F2937")

if y_axis_mode == "Logarithmic":
    fig.update_yaxes(type="log")

st.plotly_chart(fig, use_container_width=True)

if show_spectrum:
    st.markdown("### Cycle Spectrum")

    spec_fig = go.Figure()

    spec_fig.add_trace(go.Scatter(
        x=spectrum_raw["Period (d)"],
        y=spectrum_raw["Cycle Quality"],
        mode="lines",
        name="Cycle Quality",
        line=dict(width=2, color="#3CC6B7"),
        fill="tozeroy"
    ))

    spec_fig.add_vline(
        x=selected_cycle,
        line_dash="dash",
        line_color="#F6C85F",
        annotation_text=f"{selected_cycle:.1f}d",
        annotation_position="top"
    )

    spec_fig.update_layout(
        height=260,
        template="plotly_dark",
        paper_bgcolor="#070A0F",
        plot_bgcolor="#0B111A",
        xaxis_title="Cycle Period (days)",
        yaxis_title="Cycle Quality",
        margin=dict(l=30, r=30, t=20, b=30)
    )

    spec_fig.update_xaxes(gridcolor="#1F2937")
    spec_fig.update_yaxes(gridcolor="#1F2937")

    st.plotly_chart(spec_fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    def render_card(column, title, df, color):
        if df.empty:
            column.info(f"Kein {title} gefunden")
            return

        row = df.iloc[0]
        column.markdown(f"""
        <div class="cycle-card" style="border-color:{color};">
            <div class="cycle-label">{title.upper()} · TOP CYCLE</div>
            <div class="cycle-number">{row["Period (d)"]:.1f}d</div>
            <div class="cycle-small">
                quality {row["Cycle Quality"]:.2f} ·
                stability {row["Stability"]:.2f} ·
                phase {row["Phase Quality"]:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_card(c1, "Long", best_long, "#B89F4D")
    render_card(c2, "Medium", best_medium, "#3CC6B7")
    render_card(c3, "Short", best_short, "#C47A4A")

    table_band = sidebar_band if cycle_mode != "Composite Mix" else "Long"
    shown_table = spectrum[spectrum["Band"] == table_band].sort_values("Cycle Quality", ascending=False).head(25)

    st.dataframe(
        shown_table,
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "Hinweis: Auch diese Version ist keine sichere Vorhersage. Sie erkennt historische Cycle-Strukturen, Stabilität und Phasenqualität und projiziert daraus mögliche Zeitfenster."
)