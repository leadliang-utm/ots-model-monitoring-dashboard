import os
from datetime import datetime

import pandas as pd

# Always write logs next to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "monitoring_logs.csv")


def log_prediction(
    model_version,
    model_type,
    input_summary,
    prediction,
    latency_ms,
    feedback_score,
    feedback_text,
):
    """
    Append a single prediction event to monitoring_logs.csv.
    Creates the file with header if it does not exist yet.
    """
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "model_type": model_type,
        "input_summary": input_summary,
        "prediction": float(prediction),
        "latency_ms": float(latency_ms) if latency_ms is not None else None,
        "feedback_score": int(feedback_score) if feedback_score is not None else None,
        "feedback_text": feedback_text or "",
    }

    df_new = pd.DataFrame([row])

    if not os.path.exists(LOG_PATH):
        df_new.to_csv(LOG_PATH, index=False)
    else:
        df_new.to_csv(LOG_PATH, mode="a", header=False, index=False)
