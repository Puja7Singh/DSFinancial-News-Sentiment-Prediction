"""Streamlit dashboard for financial tweet / headline sentiment.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_splits  # noqa: E402
from src.evaluate import load_all_metrics  # noqa: E402
from src.predict import (SentimentPredictor, available_models,  # noqa: E402
                         best_model_name)

COLORS = {"Bearish": "#d64545", "Bullish": "#2e9e5b", "Neutral": "#8a8f98"}
EXAMPLES = [
    "$TSLA shares slump after Goldman Sachs downgrades to Sell",
    "Apple beats on revenue and raises its dividend; stock jumps 5% after hours",
    "Fed minutes due at 2pm ET; markets await guidance on rates",
    "Oil prices slide as OPEC+ signals higher output. Energy names lead the "
    "S&P 500 lower, while tech holds steady.",
]

st.set_page_config(page_title="FinTweet Sentiment", page_icon="📈",
                   layout="wide")


def class_bar(df, value, title, height):
    """Horizontal bar per class with fixed, consistent class colours."""
    scale = alt.Scale(domain=list(COLORS), range=list(COLORS.values()))
    return alt.Chart(df, height=height).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X(f"{value}:Q", title=title),
        y=alt.Y("class:N", sort=config.LABELS, title=None),
        color=alt.Color("class:N", scale=scale, legend=None),
        tooltip=["class", alt.Tooltip(f"{value}:Q", format=".3~f")])


@st.cache_resource
def get_predictor(name):
    return SentimentPredictor(name)


@st.cache_data
def get_validation():
    _, valid_df = load_splits()
    valid_df["sentiment"] = valid_df["label"].map(config.ID2LABEL)
    return valid_df


# --------------------------------------------------------------------------
# Sidebar: model choice and its validation scores
# --------------------------------------------------------------------------
models = available_models()
if not models:
    st.error("No trained model found. Run `python -m src.run_baselines`.")
    st.stop()
default = best_model_name()
st.sidebar.header("Model")
name = st.sidebar.selectbox("Trained model", models,
                            index=models.index(default))
metrics = load_all_metrics()
if name in metrics:
    st.sidebar.metric("Validation accuracy", f"{metrics[name]['accuracy']:.1%}")
    st.sidebar.metric("Validation macro F1", f"{metrics[name]['macro_f1']:.3f}")
st.sidebar.caption(f"Best model by macro F1: **{default}**")

# --------------------------------------------------------------------------
# Main: input and prediction
# --------------------------------------------------------------------------
st.title("📈 Financial News Sentiment")
st.write("Type a finance tweet, headline or paragraph. The model labels it "
         "**Bearish**, **Bullish** or **Neutral**.")

example = st.selectbox("Or start from an example", [""] + EXAMPLES)
text = st.text_area("Text", value=example, height=110,
                    placeholder="e.g. $NFLX surges after subscriber beat")

if st.button("Predict sentiment", type="primary") and text.strip():
    result = get_predictor(name).analyse(text)
    label = result["label"]
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"### Prediction: <span style='color:{COLORS[label]}'>"
                    f"{label}</span>", unsafe_allow_html=True)
        st.write(f"Confidence: {result['probabilities'][label]:.1%}")
    with right:
        probs = pd.DataFrame({"class": list(result["probabilities"]),
                              "probability": list(
                                  result["probabilities"].values())})
        st.altair_chart(class_bar(probs, "probability", "probability", 150),
                        use_container_width=True)
    if len(result["sentences"]) > 1:
        st.subheader("Per-sentence breakdown")
        st.dataframe(pd.DataFrame(result["sentences"]),
                     use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Dataset overview and model comparison
# --------------------------------------------------------------------------
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.subheader("Validation class distribution")
    dist = get_validation()["sentiment"].value_counts().reindex(config.LABELS)
    st.altair_chart(class_bar(dist.rename_axis("class").reset_index(
        name="tweets"), "tweets", "tweets", 200), use_container_width=True)
with col2:
    st.subheader("Model comparison (validation)")
    table = pd.DataFrame([
        {"model": m, "accuracy": v["accuracy"], "macro F1": v["macro_f1"]}
        for m, v in metrics.items() if v.get("family") != "smoke"
    ]).sort_values("macro F1", ascending=False)
    st.dataframe(table.style.format({"accuracy": "{:.3f}",
                                     "macro F1": "{:.3f}"}),
                 use_container_width=True, hide_index=True)
