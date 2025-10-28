# app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import io, json
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

# ----------------------------------------
# PAGE CONFIG + DARK THEME CSS
# ----------------------------------------
st.set_page_config(page_title="💓 FitPulse Health Dashboard", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; color: #FFFFFF !important; }
    [data-testid="stSidebar"] { background-color: #0b0b0b !important; color: #FFFFFF !important; }
    h1, h2, h3, h4, h5, h6, p, label { color: #00ffcc !important; }
    .stFileUploader { background-color: #111111; border-radius: 8px; padding: 8px; }
    .stButton>button, .stDownloadButton>button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #888 !important;
        border-radius: 6px !important;
        font-weight: bold !important;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #222222 !important;
        color: #00ffcc !important;
        border: 1px solid #00ffcc !important;
    }
    .js-plotly-plot .plotly { background-color: #000000 !important; }
    hr { border: 1px solid #00ffcc !important; }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: #00ffcc; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------
# Header
# ----------------------------------------
st.markdown("""
    <div style="text-align:center;">
        <img src="https://img.icons8.com/fluency/96/heart-with-pulse.png" width="90">
        <h1>FitPulse</h1>
        <p style="font-size:16px;color:#bfeee0;">
            Real-time insights from your heartbeat & health metrics 
        </p>
    </div>
""", unsafe_allow_html=True)

# ----------------------------------------
# Sidebar navigation
# ----------------------------------------
page = st.sidebar.radio(
    "🔎 Select Milestone",
    [
        "🧩 Milestone 1 – Data Collection & Preprocessing",
        "⚙️ Milestone 2 – Feature Extraction & Modelling",
        "🚨 Milestone 3 – Anomaly Detection & Visualization",
    ],
)

u_file = st.sidebar.file_uploader("Upload CSV/JSON", type=["csv", "json"])

@st.cache_data
def load_data_from_file(uploaded):
    try:
        if uploaded.name.endswith(".csv"):
            return pd.read_csv(uploaded)
        else:
            return pd.read_json(uploaded)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def compute_basic_stats(df):
    return int(df["bpm"].min()), int(df["bpm"].max()), round(df["bpm"].mean(), 1), int(df["bpm"].iloc[-1])

def download_csv_button(df, label, name):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv")

def plot_gauge(bpm, key=None):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(bpm),
            gauge={
                "axis": {"range": [40, 160]},
                "bar": {"color": "#e63946" if bpm > 100 else "#06d6a0"},
                "steps": [
                    {"range": [40, 60], "color": "#00b4d8"},
                    {"range": [60, 100], "color": "#06d6a0"},
                    {"range": [100, 160], "color": "#e63946"},
                ],
            },
            title={"text": "Current BPM"},
        )
    )
    st.plotly_chart(fig, use_container_width=True, key=key or f"gauge_{bpm}")

# ----------------------------------------
# Milestone 1
# ----------------------------------------
if page.startswith("🧩"):
    st.header("🧩 Milestone 1 — Data Collection & Preprocessing")

    anim_speed = st.sidebar.slider("⏱ Animation speed", 0.05, 1.0, 0.3, step=0.05)
    theme_choice = st.sidebar.radio("🎨 Chart theme", ["plotly_dark", "plotly_white"], index=0)
    show_hist = st.sidebar.checkbox("Show BPM Histogram", True)
    show_pie = st.sidebar.checkbox("Show Zone Pie Chart", True)
    show_avg = st.sidebar.checkbox("Show Rolling Average", True)

    if u_file is not None:
        df1 = load_data_from_file(u_file)
        if df1 is None: st.stop()
    else:
        np.random.seed(0)
        timestamps = pd.date_range(start="2025-01-01", periods=200, freq="T")
        df1 = pd.DataFrame({"timestamp": timestamps, "bpm": np.random.normal(80, 12, len(timestamps)).round(0)})

    df1 = df1.sort_values("timestamp").reset_index(drop=True)
    df1["bpm"] = pd.to_numeric(df1["bpm"], errors="coerce").fillna(method="ffill")
    df1["avg"] = df1["bpm"].rolling(5, min_periods=1).mean()

    min_bpm, max_bpm, avg_bpm, latest_bpm = compute_basic_stats(df1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔻 Min BPM", min_bpm)
    c2.metric("📈 Max BPM", max_bpm)
    c3.metric("📊 Average BPM", avg_bpm)
    c4.metric("🕒 Latest BPM", latest_bpm)

    filtered_clean = df1[df1["bpm"] <= 100].copy()
    download_csv_button(filtered_clean, "⬇ Download Clean Data", "heartbeat_clean.csv")

    if show_hist: st.bar_chart(df1["bpm"])

    if show_pie:
        st.subheader("Time Spent in Heart Rate Zones")
        low = (df1["bpm"] < 60).sum()
        normal = ((df1["bpm"] >= 60) & (df1["bpm"] <= 100)).sum()
        high = (df1["bpm"] > 100).sum()
        pie = px.pie(values=[low, normal, high], names=["Low", "Normal", "High"], template=theme_choice)
        st.plotly_chart(pie, use_container_width=True)

    st.subheader("Live BPM Animation & Gauge")
    alert_box, bpm_box, heart_box, chart_box, gauge_box = st.empty(), st.empty(), st.empty(), st.empty(), st.empty()

    if st.button("▶ Start Animation"):
        for i, row in df1.iterrows():
            bpm = float(row["bpm"])
            color = "#00b4d8" if bpm < 60 else "#06d6a0" if bpm <= 100 else "#e63946"
            alert = "⚠ Low heart rate!" if bpm < 60 else "🚨 High heart rate!" if bpm > 100 else ""
            alert_box.markdown(f"<h3 style='text-align:center;color:{color};'>{alert}</h3>", unsafe_allow_html=True)
            bpm_box.markdown(f"<h2 style='text-align:center;color:{color};'>Current BPM: {int(bpm)}</h2>", unsafe_allow_html=True)
            heart_box.markdown(f"<h1 style='text-align:center;color:{color};'>❤</h1>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df1.iloc[:i+1]["timestamp"], y=df1.iloc[:i+1]["bpm"], mode="lines+markers",
                                     line=dict(color=color, width=3)))
            if show_avg and i >= 4:
                fig.add_trace(go.Scatter(x=df1.iloc[:i+1]["timestamp"], y=df1.iloc[:i+1]["avg"], mode="lines",
                                         line=dict(color="#ffa600", width=2, dash="dot")))
            fig.update_layout(template=theme_choice, height=420)
            chart_box.plotly_chart(fig, use_container_width=True)
            time.sleep(anim_speed)

# ----------------------------------------
# Milestone 2
# ----------------------------------------
elif page.startswith("⚙️"):
    st.header("⚙️ Milestone 2 — Feature Extraction & Modelling")

    if u_file is not None:
        df2 = load_data_from_file(u_file)
        if df2 is None: st.stop()
    else:
        np.random.seed(42)
        timestamps = pd.date_range(start="2025-01-01", end="2025-02-28 23:00:00", freq="H")
        df2 = pd.DataFrame({
            "timestamp": timestamps,
            "bpm": np.random.normal(80, 10, len(timestamps)).round(2),
            "step_count": np.random.normal(6000, 2000, len(timestamps)).round(0),
            "sleep_duration": np.random.normal(7.5, 0.8, len(timestamps)).round(2)
        })

    metric = st.sidebar.selectbox("Select Metric", ["bpm", "step_count", "sleep_duration"])
    forecast_period = st.sidebar.slider("Forecast horizon (hours)", 24, 168, 72)
    cluster_choice = st.sidebar.selectbox("Clustering Method", ["K-Means", "DBSCAN"])
    anim_speed2 = st.sidebar.slider("Animation speed", 0.01, 0.2, 0.03)

    # Prophet Forecast Animation
    st.subheader("📈 Prophet Forecasting Animation")
    prophet_df = df2[["timestamp", metric]].rename(columns={"timestamp": "ds", metric: "y"})
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    try:
        m = Prophet()
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=forecast_period, freq="H")
        forecast = m.predict(future)
    except Exception as e:
        st.error(f"Prophet error: {e}")
        forecast = pd.DataFrame()

    ph_placeholder = st.empty()
    if not forecast.empty:
        for i in range(10, len(forecast), max(1, len(forecast)//30)):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], mode="lines", name="Actual", line=dict(color="green")))
            fig.add_trace(go.Scatter(x=forecast["ds"][:i], y=forecast["yhat"][:i], mode="lines", name="Forecast", line=dict(color="orange")))
            fig.update_layout(template="plotly_dark", height=420)
            ph_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(anim_speed2)

    # 🚨 Anomaly Detection Animation
    st.subheader("🚨 Anomaly Detection Animation")
    rolling_mean = prophet_df['y'].rolling(window=24).mean()
    rolling_std = prophet_df['y'].rolling(window=24).std()
    prophet_df['anomaly'] = np.where(
        (prophet_df['y'] > rolling_mean + 2 * rolling_std) | (prophet_df['y'] < rolling_mean - 2 * rolling_std),
        1, 0
    )
    placeholder2 = st.empty()
    for i in range(10, len(prophet_df), 20):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=prophet_df['ds'][:i], y=prophet_df['y'][:i], mode='lines', name='Value', line=dict(color='green', width=3)))
        anomalies = prophet_df[prophet_df['anomaly'] == 1].iloc[:i]
        fig2.add_trace(go.Scatter(x=anomalies['ds'], y=anomalies['y'], mode='markers', name='Anomaly', marker=dict(color='red', size=8)))
        fig2.update_layout(template="plotly_dark", height=400)
        placeholder2.plotly_chart(fig2, use_container_width=True)
        time.sleep(0.03)

    # Behavioral Clustering
    st.subheader("🧠 Behavioral Clustering (3D)")
    if {"bpm", "step_count", "sleep_duration"}.issubset(df2.columns):
        scaled = StandardScaler().fit_transform(df2[["bpm", "step_count", "sleep_duration"]])
        if cluster_choice == "K-Means":
            df2["cluster"] = KMeans(n_clusters=3, n_init=10).fit_predict(scaled)
        else:
            df2["cluster"] = DBSCAN(eps=0.8, min_samples=10).fit_predict(scaled)
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(x=df2["bpm"], y=df2["step_count"], z=df2["sleep_duration"], mode='markers',
                                     marker=dict(size=4, color=df2["cluster"], colorscale='Viridis')))
        fig3d.update_layout(template="plotly_dark", title="3D Clustering View", height=520)
        st.plotly_chart(fig3d, use_container_width=True)

    # Rolling Average Trend
    st.subheader("📉 Rolling Average Trend")
    df2["rolling_bpm_24h"] = df2["bpm"].rolling(window=24, min_periods=1).mean()
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=df2["timestamp"], y=df2["bpm"], mode="lines", name="BPM", line=dict(color="green")))
    fig_roll.add_trace(go.Scatter(x=df2["timestamp"], y=df2["rolling_bpm_24h"], mode="lines", name="24h Rolling Avg", line=dict(color="magenta", width=3)))
    fig_roll.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig_roll, use_container_width=True)

    # Bubble Chart & Pairwise Matrix
    st.subheader("💧 Bubble Chart & Pairwise Matrix")
    if {"bpm", "step_count", "sleep_duration"}.issubset(df2.columns):
        fig_bubble = px.scatter(df2, x="bpm", y="step_count", size="sleep_duration", color=df2["cluster"], template="plotly_dark")
        st.plotly_chart(fig_bubble, use_container_width=True)
        fig_matrix = px.scatter_matrix(df2, dimensions=["bpm", "step_count", "sleep_duration"], color=df2["cluster"], template="plotly_dark")
        st.plotly_chart(fig_matrix, use_container_width=True)

    st.success("✅ Feature Extraction & Modelling Completed.")

# ----------------------------------------
# Milestone 3 — Anomaly Detection & Visualization
# (includes threshold, residual (Prophet), DBSCAN, combined visualizations, downloads)
# ----------------------------------------
elif page.startswith("🚨"):
    st.header("🚨 Milestone 3 — Anomaly Detection & Visualization")

    data_option = st.radio("Select Input Option", ["📘 Use Sample Data", "📤 Upload CSV/JSON"])
    if data_option.startswith("📤"):
        uploaded_m3 = st.file_uploader("Upload health dataset for anomaly detection", type=["csv", "json"])
        if uploaded_m3:
            df3 = load_data_from_file(uploaded_m3)
            if df3 is None:
                st.stop()
        else:
            st.info("Upload a file to run detection.")
            st.stop()
    else:
        # sample daily data
        np.random.seed(42)
        timestamps = pd.date_range(start="2025-01-01", end="2025-02-28", freq="D")
        df3 = pd.DataFrame({
            "timestamp": timestamps,
            "bpm": np.random.normal(80, 10, len(timestamps)).round(2),
            "step_count": np.random.normal(6000, 2000, len(timestamps)).round(0),
            "sleep_duration": np.random.normal(7.5, 0.8, len(timestamps)).round(2)
        })
        df3.loc[np.random.choice(len(df3), 3), "bpm"] = np.random.randint(125, 160, 3)
        df3.loc[np.random.choice(len(df3), 3), "step_count"] = np.random.randint(12000, 16000, 3)
        df3.loc[np.random.choice(len(df3), 3), "sleep_duration"] = np.random.uniform(3, 5, 3)
        st.success("✅ Sample data generated with injected anomalies.")

    required_cols = {"timestamp", "bpm", "step_count", "sleep_duration"}
    if not required_cols.issubset(df3.columns):
        st.error(f"❌ Missing required columns. Required: {required_cols}")
        st.stop()

    # sidebar sensitivity
    sensitivity = st.sidebar.slider("🧭 Threshold Sensitivity (1–5)", 1, 5, 3)
    base_bpm_limit = 120
    base_step_limit = 10000
    base_sleep_limit = 7.0
    bpm_limit = base_bpm_limit - (sensitivity - 3) * 5
    step_limit = base_step_limit - (sensitivity - 3) * 1000
    sleep_limit = base_sleep_limit + (sensitivity - 3) * 0.2
    residual_std_threshold = 2 + (sensitivity - 3) * 0.5

    st.sidebar.markdown("**Effective Limits**")
    st.sidebar.write(f"- BPM > **{bpm_limit:.0f}**\n- Steps > **{step_limit:.0f}**\n- Sleep < **{sleep_limit:.1f} hr**\n- Prophet Std Threshold = **{residual_std_threshold:.1f}**")

    # 1) Threshold-based detection
    df3["bpm_anomaly"] = (df3["bpm"] > bpm_limit).astype(int)
    df3["step_anomaly"] = (df3["step_count"] > step_limit).astype(int)
    df3["sleep_anomaly"] = (df3["sleep_duration"] < sleep_limit).astype(int)

    # 2) Residual-based using Prophet
    prophet_df = df3[["timestamp", "bpm"]].rename(columns={"timestamp": "ds", "bpm": "y"})
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    try:
        model = Prophet(daily_seasonality=True)
        model.fit(prophet_df)
        forecast = model.predict(prophet_df)
        df3["residual"] = prophet_df["y"] - forecast["yhat"]
        std_val = df3["residual"].std()
        df3["bpm_residual_anomaly"] = (abs(df3["residual"]) > residual_std_threshold * std_val).astype(int)
    except Exception as e:
        st.warning(f"Prophet residual detection failed: {e}")
        df3["residual"] = 0.0
        df3["bpm_residual_anomaly"] = 0

    # 3) Cluster-based (DBSCAN)
    features = df3[["bpm", "step_count", "sleep_duration"]]
    scaled = StandardScaler().fit_transform(features)
    db = DBSCAN(eps=1.2, min_samples=4)
    db.fit(scaled)
    df3["cluster_label"] = db.labels_
    df3["cluster_anomaly"] = (db.labels_ == -1).astype(int)

    # combine anomalies
    df3["any_anomaly"] = df3[["bpm_anomaly", "step_anomaly", "sleep_anomaly", "bpm_residual_anomaly", "cluster_anomaly"]].max(axis=1)
    anomalies_df = df3[df3["any_anomaly"] == 1].reset_index(drop=True)

    # Visualizations
    st.subheader("📊 Anomaly Visualizations (All Methods)")
# Threshold visuals
    st.subheader("Threshold-Based Detection (line / bar)")
    thr1, thr2 = st.columns(2)
    with thr1:
        fig_thr_bpm = px.line(df3, x="timestamp", y="bpm", title="BPM Threshold Anomalies", template="plotly_dark")
        fig_thr_bpm.add_scatter(x=df3[df3["bpm_anomaly"] == 1]["timestamp"], y=df3[df3["bpm_anomaly"] == 1]["bpm"], mode="markers", name="BPM Anomaly", marker=dict(color="#ff0066", size=10))
        st.plotly_chart(fig_thr_bpm, use_container_width=True)
    with thr2:
        fig_thr_steps = px.line(df3, x="timestamp", y="step_count", title="Step Count Threshold Anomalies", template="plotly_dark")
        fig_thr_steps.add_scatter(x=df3[df3["step_anomaly"] == 1]["timestamp"], y=df3[df3["step_anomaly"] == 1]["step_count"], mode="markers", name="Step Anomaly", marker=dict(color="#ffcc00", size=10))
        st.plotly_chart(fig_thr_steps, use_container_width=True)

    st.subheader("Sleep Duration Threshold (bar + markers)")
    fig_sleep = px.bar(df3, x="timestamp", y="sleep_duration", title="Sleep Duration (with low-sleep markers)", template="plotly_dark")
    fig_sleep.add_scatter(x=df3[df3["sleep_anomaly"] == 1]["timestamp"], y=df3[df3["sleep_anomaly"] == 1]["sleep_duration"], mode="markers", name="Low Sleep", marker=dict(color="#ff3366", size=10))
    st.plotly_chart(fig_sleep, use_container_width=True)
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Residual-Based BPM Detection (Prophet)**")
        fig_res = px.scatter(df3, x="timestamp", y="bpm",
                             color=df3["bpm_residual_anomaly"].map({0: "Normal", 1: "Anomaly"}),
                             title="Residual-Based BPM Detection",
                             color_discrete_map={"Normal": "#00ccff", "Anomaly": "#ff3366"},
                             template="plotly_dark")
        st.plotly_chart(fig_res, use_container_width=True)
    with colB:
        st.markdown("**Cluster-Based Detection (DBSCAN)**")
        fig_cl = px.scatter_3d(df3, x="bpm", y="step_count", z="sleep_duration",
                               color=df3["cluster_anomaly"].map({0: "Normal", 1: "Anomaly"}),
                               title="Cluster-Based Outliers",
                               color_discrete_map={"Normal": "#00ffcc", "Anomaly": "#ff0066"},
                               template="plotly_dark")
        st.plotly_chart(fig_cl, use_container_width=True)

    

    # Downloads
    st.subheader("📥 Download Anomaly Data")
    if not anomalies_df.empty:
        st.download_button("⬇️ Download Anomalies (CSV)", anomalies_df.to_csv(index=False).encode(), file_name="health_anomalies.csv", mime="text/csv")
        st.download_button("⬇️ Download Anomalies (JSON)", json.dumps(anomalies_df.to_dict(orient="records"), indent=4, default=str).encode(), file_name="health_anomalies.json", mime="application/json")
    else:
        st.info("No anomalies detected with current settings — adjust sensitivity.")

    st.success(f"✅ Detection Completed — Total records: {len(df3)} | Anomalies: {len(anomalies_df)}")

# ----------------------------------------
# Footer
# ----------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Built with ❤ — FitPulse</p>", unsafe_allow_html=True)

