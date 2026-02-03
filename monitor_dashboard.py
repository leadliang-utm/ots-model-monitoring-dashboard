import os
import pandas as pd
import streamlit as st

from log_utils import LOG_PATH

st.set_page_config(page_title="Model Monitoring & Feedback", layout="wide")

st.title("Model Monitoring & Feedback Dashboard")

@st.cache_data
def load_logs():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp")

logs = load_logs()

# Handle "no logs yet"
if logs.empty:
    st.warning(
        "No monitoring logs found yet. "
        "Please run the surgical prediction app, submit feedback at least once, and then refresh this page."
    )
    st.stop()

st.sidebar.header("Filters")
models = ["All"] + sorted(logs["model_version"].unique().tolist())
selected_model = st.sidebar.selectbox("Model version", models)

if selected_model == "All":
    filtered = logs
else:
    filtered = logs[logs["model_version"] == selected_model]

st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", len(filtered))

if filtered["feedback_score"].notna().any():
    col2.metric("Avg Feedback Score", f"{filtered['feedback_score'].mean():.2f}")
else:
    col2.metric("Avg Feedback Score", "N/A")

if filtered["latency_ms"].notna().any():
    col3.metric("Avg Latency (ms)", f"{filtered['latency_ms'].mean():.1f}")
else:
    col3.metric("Avg Latency (ms)", "N/A")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "💬 Feedback Analysis", "📄 Raw Logs"])
with tab1:
    st.subheader("Model Version Comparison (Aggregated)")
    summary = logs.groupby("model_version").agg({
        "feedback_score": "mean",
        "latency_ms": "mean",
    })
    summary = summary.rename(columns={
        "feedback_score": "avg_feedback_score",
        "latency_ms": "avg_latency_ms",
    })
    st.dataframe(summary.style.format({
        "avg_feedback_score": "{:.2f}",
        "avg_latency_ms": "{:.1f}",
    }))

with tab2:
    st.subheader("Average Feedback Score by Model Version")
    fb = logs.groupby("model_version")["feedback_score"].mean().reset_index()
    fb_chart = fb.set_index("model_version")
    st.bar_chart(fb_chart)

    st.subheader("Recent Comments")
    comments = logs.copy()
    comments = comments[comments["feedback_text"].astype(str).str.strip() != ""]
    comments = comments.sort_values("timestamp", ascending=False).head(10)

    if comments.empty:
        st.info("No qualitative comments yet.")
    else:
        for _, row in comments.iterrows():
            st.write(f"**[{row['timestamp']}] {row['model_version']} – Score: {row['feedback_score']}**")
            st.write(row["feedback_text"])
            st.markdown("---")

with tab3:
    st.subheader("Raw Monitoring Logs")
    st.dataframe(filtered)
