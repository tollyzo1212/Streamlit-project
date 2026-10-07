import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fetal Health Predictor", page_icon="🫀", layout="centered")

@st.cache_resource
def load_bundle():
    return joblib.load("fetal_health_model.joblib")

B = load_bundle()
model, FEATURES, CLASSES = B["model"], B["features"], B["classes"]

LABELS = {
    "abnormal_short_term_variability": "Abnormal short-term variability (%)",
    "percentage_of_time_with_abnormal_long_term_variability": "Time with abnormal long-term variability (%)",
    "histogram_mean": "Histogram mean (bpm)",
    "mean_value_of_short_term_variability": "Mean short-term variability",
    "accelerations": "Accelerations (per second)",
    "prolongued_decelerations": "Prolonged decelerations (per second)",
    "histogram_mode": "Histogram mode (bpm)",
    "mean_value_of_long_term_variability": "Mean long-term variability",
    "baseline_value": "Baseline fetal heart rate (bpm)",
    "histogram_variance": "Histogram variance",
}
STEP = {"accelerations": 0.001, "prolongued_decelerations": 0.0005, "mean_value_of_short_term_variability": 0.1,
        "mean_value_of_long_term_variability": 0.1}
COLORS = {1: "#2a9d8f", 2: "#e9c46a", 3: "#e76f51"}

def predict_proba(values: dict) -> np.ndarray:
    """Apply the same clipping + scaling used in training, then predict class probabilities."""
    x = []
    for f in FEATURES:
        v = float(np.clip(values[f], B["clip_lo"].get(f, -np.inf), B["clip_hi"].get(f, np.inf)))
        x.append((v - B["mean"][f]) / B["scale"][f])
    return model.predict_proba(pd.DataFrame([x], columns=FEATURES))[0]

st.title("🫀 Fetal Health Predictor")
st.caption("Decision-support prototype built from Cardiotocogram (CTG) readings · Random Forest model. "
           "Not a medical device: always confirm with a qualified clinician.")

st.subheader("Enter the CTG measurements")
values = {}
cols = st.columns(2)
for i, f in enumerate(FEATURES):
    lo, hi, med = B["ranges"][f]
    is_int = float(lo).is_integer() and float(hi).is_integer() and f not in STEP
    with cols[i % 2]:
        if is_int:
            values[f] = st.number_input(LABELS[f], min_value=int(lo), max_value=int(hi), value=int(med), step=1)
        else:
            step = STEP.get(f, 0.1)
            values[f] = st.number_input(LABELS[f], min_value=float(lo), max_value=float(hi),
                                        value=float(med), step=float(step), format="%.4f" if step < 0.01 else "%.2f")

if st.button("Predict fetal health", type="primary", use_container_width=True):
    p = predict_proba(values)
    top = int(np.argmax(p)) + 1
    st.markdown("---")
    st.subheader("Result")
    msg = f"The model predicts a **{p[top-1]*100:.2f}% probability** that the fetal health status is **{CLASSES[top]}**."
    {1: st.success, 2: st.warning, 3: st.error}[top](msg)

    st.markdown("**Probability of each outcome**")
    for k in (1, 2, 3):
        st.write(f"{CLASSES[k]}: **{p[k-1]*100:.2f}%**")
        st.progress(float(p[k-1]))
    st.bar_chart(pd.DataFrame({"Probability (%)": [p[k-1]*100 for k in (1, 2, 3)]},
                              index=[CLASSES[k] for k in (1, 2, 3)]))
    if top != 1:
        st.info("Suspect or Pathological readings should be reviewed urgently by a clinician.")

with st.expander("About this model"):
    st.write("Trained on 2,126 CTG recordings (13 duplicates removed). Features were scaled, the training set was "
             "balanced with SMOTE, and 10 clinically relevant features were selected. Test accuracy ≈ 94%, "
             "macro-F1 ≈ 0.90, macro ROC-AUC ≈ 0.985, Pathological recall ≈ 97%.")
