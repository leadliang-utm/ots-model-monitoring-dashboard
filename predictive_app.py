import time
import joblib
import pandas as pd
import streamlit as st

from log_utils import log_prediction

st.set_page_config(page_title="Surgical Prediction App with Monitoring",
                   layout="centered")
st.title("Surgical Prediction App with Live Monitoring")

@st.cache_resource
def load_models():
    old_model = joblib.load("baseline_model_v1.pkl")  # trained on ["ProposedProcedure", "DisplayValue"]
    new_model = joblib.load("improved_model_v2.pkl")  # trained on ["ProposedProcedure","DisplayValue","StartHour","TurnAroundTime"]
    return old_model, new_model

old_model, new_model = load_models()

# ---------- Initialise session state ----------
if "pred_ready" not in st.session_state:
    st.session_state["pred_ready"] = False
if "old_pred" not in st.session_state:
    st.session_state["old_pred"] = None
if "new_pred" not in st.session_state:
    st.session_state["new_pred"] = None
if "latency_ms" not in st.session_state:
    st.session_state["latency_ms"] = None
if "input_summary" not in st.session_state:
    st.session_state["input_summary"] = ""

# ---------- INPUT SECTION ----------
st.sidebar.header("Input Parameters")

StartHour = st.sidebar.slider("Operation Start Hour",min_value=0, max_value=23, value=8)
TurnAroundTime = st.sidebar.slider("Turnaround Time (minutes)",min_value=0,max_value=120,value=15)
DisplayValue = st.sidebar.selectbox("DisplayValue", ["OR 1", "OR 2", "OR 3", "OR 4","OR 5", "OR 6", "OR 7", "OR 8","OR 9", "OR 10", "OR 11", "OR 12","Endo 1", "Endo 2", "Endo 3", "Endo 4"])
ProposedProcedure = st.sidebar.selectbox("ProposedProcedure", ["COLOSCP - COLONOSCOPY;", "GASTSCP - GASTROSCOPY;", "GASTCOL - GASTROSCOPY & COLONOSCOPY;", "VIDEO LAPAROSCOPIC CHOLECYSTECTOMY;"])

# Canonical input dataframe
input_df = pd.DataFrame({
    "StartHour": [StartHour],
    "TurnAroundTime": [TurnAroundTime],
    "DisplayValue": [DisplayValue],
    "ProposedProcedure": [ProposedProcedure],
})

st.subheader("Input Summary")
st.write(input_df)

# ---------- BUTTON 1: RUN PREDICTION ----------
if st.button("Run Prediction"):
    start_time = time.time()

    # v1: baseline – only uses units_sold
    input_v1 = input_df[["DisplayValue", "ProposedProcedure"]]
    old_pred = old_model.predict(input_v1)[0]

    # v2: improved – uses all three features
    input_v2 = input_df[["StartHour", "TurnAroundTime", "DisplayValue", "ProposedProcedure"]]
    new_pred = new_model.predict(input_v2)[0]

    latency_ms = (time.time() - start_time) * 1000.0

    # Store in session_state so they survive reruns
    st.session_state["old_pred"] = float(old_pred)
    st.session_state["new_pred"] = float(new_pred)
    st.session_state["latency_ms"] = float(latency_ms)
    st.session_state["input_summary"] = f"StartHour={StartHour}, TurnAroundTime={TurnAroundTime}, DisplayValue={DisplayValue}, ProposedProcedure={ProposedProcedure}"
    st.session_state["pred_ready"] = True

# ---------- SHOW PREDICTIONS IF READY ----------
if st.session_state["pred_ready"]:
    st.subheader("Predictions")
    st.write(f"Old Model (v1 - ProposedProcedure + DisplayValue only): **{st.session_state['old_pred']:,.2f}**")
    st.write(f"New Model (v2 - ProposedProcedure + DisplayValue + StartHour + TurnAroundTime): **{st.session_state['new_pred']:,.2f}**")
    st.write(f"Latency: {st.session_state['latency_ms']:.1f} ms")
else:
    st.info("Click **Run Prediction** to see model outputs before giving feedback.")

# ---------- FEEDBACK SECTION ----------
st.subheader("Your Feedback on These Predictions")

feedback_score = st.slider("How useful were this Model v1 predictions? (1 = Poor, 5 = Excellent)", min_value=1, max_value=5, value=4, key="feedback_score",)
feedback_text = st.text_area("Comments for Model v1 (optional)", key="feedback_text")


feedback_score_v2 = st.slider("How useful were this Model v2 predictions? (1 = Poor, 5 = Excellent)", min_value=1, max_value=5, value=4, key="feedback_score_v2",)
feedback_text_v2 = st.text_area("Comments for Model v2 (optional)", key="feedback_text_v2")

# ---------- BUTTON 2: SUBMIT FEEDBACK ----------
if st.button("Submit Feedback"):
    if not st.session_state["pred_ready"]:
        st.warning("Please run the prediction first, then submit your feedback.")
    else:
        # Log both models using saved predictions and input summary
        log_prediction(
            model_version="v1_old",
            model_type="baseline",
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["old_pred"],
            latency_ms=st.session_state["latency_ms"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )

        log_prediction(
            model_version="v2_new",
            model_type="improved",
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["new_pred"],
            latency_ms=st.session_state["latency_ms"],
            feedback_score=feedback_score_v2,
            feedback_text=feedback_text_v2,
        )

        st.success(
            "Feedback and predictions have been saved to monitoring_logs.csv. "
            "You can now view them in the monitoring dashboard."
        )

