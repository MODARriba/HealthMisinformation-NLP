"""HealthClaimBERT - Streamlit frontend for health misinformation detection.

Loads the fine-tuned models trained in the project notebooks:
  - models/pubmedbert_fakehealth_healthfact/   (HF save_pretrained folder)
  - models/biobert_fakehealth_healthfact/      (HF save_pretrained folder)
  - models/tfidf_logreg.joblib                 (scikit-learn Pipeline, joblib)

Label mapping (from model.config.id2label): {0: 'misinformation', 1: 'reliable'}
Run locally with:  streamlit run app.py
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="HealthClaimBERT", page_icon="\U0001F9EC", layout="wide")

APP_DIR = Path(__file__).parent
MODEL_ROOT = APP_DIR / "models"
ASSETS_DIR = APP_DIR / "assets"
DATA_DIR = APP_DIR / "data"

MODEL_DIR_MAP = {
    "BioBERT": MODEL_ROOT / "biobert_fakehealth_healthfact",
    "PubMedBERT": MODEL_ROOT / "pubmedbert_fakehealth_healthfact",
}
TFIDF_CANDIDATES = [MODEL_ROOT / "tfidf_logreg.joblib", MODEL_ROOT / "tfidf_logreg.pkl"]
SUMMARY_PATH = DATA_DIR / "comparison_summary.json"

MAX_LENGTH = 256  # must match fine-tuning
ID2LABEL_FALLBACK = {0: "misinformation", 1: "reliable"}
MAX_BATCH_ROWS = 200

# ---------------------------------------------------------------------------
# Real thesis metrics - Chapter 6.2, held-out test set (n=1,311), 5 epochs
# ---------------------------------------------------------------------------
METRICS = pd.DataFrame(
    {
        "Model": ["TF-IDF + LR", "BioBERT", "PubMedBERT"],
        "Accuracy": [68.73, 73.38, 73.84],
        "Precision": [0.7713, 0.7290, 0.7908],
        "Recall": [0.6790, 0.8841, 0.7656],
        "F1": [0.7222, 0.7991, 0.7780],
    }
)

# Chapter 6.4 - eight-bucket error analysis (test set, n=1,311)
EIGHT_BUCKET = pd.DataFrame(
    {
        "bucket": [
            "all_correct", "tfidf_only", "tfidf_pubmed_only", "tfidf_biobert_only",
            "biobert_pubmed_only", "all_wrong", "biobert_only", "pubmed_only",
        ],
        "count": [327, 231, 178, 165, 130, 103, 98, 79],
    }
)

EXAMPLE_CLAIMS = [
    "Drinking lemon water every morning cures cancer.",
    "Vaccines cause autism in young children.",
    "Regular physical activity reduces the risk of heart disease.",
    "5G networks spread the coronavirus.",
    "A balanced diet rich in fruits and vegetables supports a healthy immune system.",
    "Taking high doses of vitamin C prevents the common cold entirely.",
]

# ---------------------------------------------------------------------------
# Cached model loaders (each fails gracefully if files are missing)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading transformer model...")
def load_transformer(name: str):
    """Return (tokenizer, model, id2label) or None if unavailable."""
    model_dir = MODEL_DIR_MAP.get(name)
    if model_dir is None or not model_dir.exists():
        return None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()
        id2label = {int(k): v for k, v in model.config.id2label.items()} or ID2LABEL_FALLBACK
        return tokenizer, model, id2label
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load {name}: {exc}")
        return None


@st.cache_resource(show_spinner="Loading TF-IDF pipeline...")
def load_tfidf():
    """Return the scikit-learn Pipeline or None if unavailable."""
    for path in TFIDF_CANDIDATES:
        if path.exists():
            try:
                import joblib

                return joblib.load(path)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not load TF-IDF pipeline: {exc}")
                return None
    return None


def best_transformer_name() -> str:
    """Mirror notebook 08: pick the best transformer from comparison_summary.json."""
    fallback = "PubMedBERT"
    if SUMMARY_PATH.exists():
        try:
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            best = summary.get("best_model_by_test_f1", fallback)
            if best not in MODEL_DIR_MAP:
                scores = summary.get("test_f1_scores", {})
                candidates = {m: s for m, s in scores.items() if m in MODEL_DIR_MAP}
                if candidates:
                    best = max(candidates, key=candidates.get)
                else:
                    best = fallback
            return best
        except Exception:  # noqa: BLE001
            return fallback
    return fallback


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def predict_transformer(name: str, text: str):
    """Return (label, confidence, probs_dict) or None."""
    bundle = load_transformer(name)
    if bundle is None:
        return None
    tokenizer, model, id2label = bundle
    import torch

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_id = int(np.argmax(probs))
    label = id2label.get(pred_id, ID2LABEL_FALLBACK[pred_id])
    probs_dict = {id2label.get(i, ID2LABEL_FALLBACK[i]): p for i, p in enumerate(probs)}
    return label, float(probs[pred_id]), probs_dict


def predict_tfidf(text: str):
    pipeline = load_tfidf()
    if pipeline is None:
        return None
    try:
        probs = pipeline.predict_proba([text])[0]
        pred_id = int(np.argmax(probs))
        label = ID2LABEL_FALLBACK[pred_id]
        probs_dict = {ID2LABEL_FALLBACK[i]: float(p) for i, p in enumerate(probs)}
        return label, float(probs[pred_id]), probs_dict
    except Exception as exc:  # noqa: BLE001
        st.warning(f"TF-IDF inference failed: {exc}")
        return None


def predict(model_name: str, text: str):
    if model_name == "TF-IDF + LR":
        return predict_tfidf(text)
    return predict_transformer(model_name, text)


def render_verdict(label: str, confidence: float):
    if label == "reliable":
        st.success(f"\u2705 **Reliable** \u2014 confidence {confidence:.1%}")
    else:
        st.warning(f"\u26A0\uFE0F **Misinformation** \u2014 confidence {confidence:.1%}")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("\U0001F9EC HealthClaimBERT")
st.subheader("PubMedBERT \u00b7 BioBERT \u00b7 TF-IDF")
st.caption(
    "Classify health claims as reliable or misinformation using fine-tuned "
    "biomedical language models."
)

BEST_NAME = best_transformer_name()

tab1, tab2, tab3, tab4 = st.tabs(
    ["\U0001F50D Claim Checker", "\u2696\uFE0F Model Comparison", "\U0001F4C2 Batch Inference", "\U0001F4CA Performance Dashboard"]
)

# ---------------------------------------------------------------------------
# Tab 1 - Claim Checker
# ---------------------------------------------------------------------------
with tab1:
    st.markdown(f"Real-time classification using the best transformer: **{BEST_NAME}**.")

    if "claim_text" not in st.session_state:
        st.session_state.claim_text = ""

    with st.expander("\U0001F4A1 Try an example claim"):
        cols = st.columns(2)
        for i, claim in enumerate(EXAMPLE_CLAIMS):
            if cols[i % 2].button(claim, key=f"example_{i}"):
                st.session_state.claim_text = claim

    claim = st.text_area(
        "Enter a health claim:", key="claim_text", height=110,
        placeholder="e.g. Drinking green tea twice a day prevents diabetes.",
    )

    if st.button("Classify", type="primary"):
        if not claim.strip():
            st.info("Please enter a claim first.")
        else:
            result = predict_transformer(BEST_NAME, claim)
            if result is None:
                st.error(
                    f"The {BEST_NAME} model files were not found in "
                    f"`models/`. Copy your fine-tuned folder there (see README)."
                )
            else:
                label, confidence, probs = result
                render_verdict(label, confidence)
                st.markdown("**Raw probabilities**")
                for cls, p in probs.items():
                    st.progress(min(max(p, 0.0), 1.0), text=f"{cls}: {p:.1%}")

# ---------------------------------------------------------------------------
# Tab 2 - Model Comparison
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("Run the **same claim** through all three models side by side.")

    cmp_claim = st.text_area("Health claim to compare:", key="cmp_text", height=110)
    col_run, col_reset = st.columns([1, 1])

    if col_run.button("Compare All Models", type="primary"):
        if not cmp_claim.strip():
            st.info("Please enter a claim first.")
        else:
            rows = []
            for model_name in ["TF-IDF + LR", "BioBERT", "PubMedBERT"]:
                result = predict(model_name, cmp_claim)
                if result is None:
                    rows.append({"Model": model_name, "Prediction": "model not available", "Confidence": "-"})
                else:
                    label, confidence, _ = result
                    rows.append({"Model": model_name, "Prediction": label, "Confidence": f"{confidence:.1%}"})
            st.session_state.cmp_results = pd.DataFrame(rows)

    if col_reset.button("Reset"):
        st.session_state.pop("cmp_results", None)
        st.session_state.cmp_text = ""
        st.rerun()

    if "cmp_results" in st.session_state:
        st.dataframe(st.session_state.cmp_results, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        "**Reference test-set F1 scores** (thesis, n=1,311): "
        "TF-IDF + LR **0.7222** \u00b7 BioBERT **0.7991** \u00b7 PubMedBERT **0.7780**"
    )
    st.info(
        "Precision-recall tradeoff: **BioBERT** has the highest recall (0.8841) "
        "\u2014 it catches the most reliable claims, at the cost of more false positives. "
        "**PubMedBERT** is more balanced, with the highest precision (0.7908) and accuracy (73.84%)."
    )

# ---------------------------------------------------------------------------
# Tab 3 - Batch Inference
# ---------------------------------------------------------------------------
with tab3:
    st.markdown(
        "**Batch inference** lets you classify many claims at once. "
        "Upload a CSV or TXT file of claims \u00b7 one claim per line \u00b7 "
        f"max {MAX_BATCH_ROWS} rows \u00b7 UTF-8 encoded. For CSV files, claims must "
        "be in a column named `claim`. Results are shown in a table and can be "
        "downloaded as CSV."
    )

    uploaded = st.file_uploader("Upload claims file", type=["csv", "txt"])

    claims_list = None
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_in = pd.read_csv(uploaded)
                if "claim" not in df_in.columns:
                    st.error("CSV must contain a column named `claim`.")
                else:
                    claims_list = df_in["claim"].dropna().astype(str).tolist()
            else:
                text = uploaded.read().decode("utf-8")
                claims_list = [line.strip() for line in text.splitlines() if line.strip()]
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read file: {exc}")

    if claims_list is not None:
        if len(claims_list) > MAX_BATCH_ROWS:
            st.error(
                f"File has {len(claims_list)} claims \u2014 the limit is {MAX_BATCH_ROWS} rows "
                "to avoid memory issues on CPU. Please split the file."
            )
        else:
            st.write(f"Loaded **{len(claims_list)}** claims.")
            if st.button("Run Batch Inference", type="primary"):
                if load_transformer(BEST_NAME) is None:
                    st.error(f"The {BEST_NAME} model files were not found in `models/`.")
                else:
                    progress = st.progress(0.0, text="Classifying...")
                    results = []
                    for i, c in enumerate(claims_list):
                        res = predict_transformer(BEST_NAME, c)
                        if res is not None:
                            label, confidence, _ = res
                            results.append({"claim": c, "prediction": label, "confidence": round(confidence, 4)})
                        progress.progress((i + 1) / len(claims_list), text=f"Classifying {i + 1}/{len(claims_list)}")
                    progress.empty()
                    out_df = pd.DataFrame(results)
                    st.dataframe(out_df, use_container_width=True, hide_index=True)

                    csv_buf = io.StringIO()
                    out_df.to_csv(csv_buf, index=False)
                    st.download_button(
                        "\u2B07\uFE0F Download results as CSV", csv_buf.getvalue(),
                        file_name="batch_predictions.csv", mime="text/csv",
                    )

                    # Confidence histogram
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    ax.hist(out_df["confidence"], bins=20, color="#4c78a8", edgecolor="white")
                    ax.set_xlabel("Confidence")
                    ax.set_ylabel("Number of claims")
                    ax.set_title(f"Confidence distribution ({BEST_NAME})")
                    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Tab 4 - Performance Dashboard
# ---------------------------------------------------------------------------
with tab4:
    st.markdown("#### Summary metrics")
    display_metrics = METRICS.copy()
    display_metrics["Accuracy"] = display_metrics["Accuracy"].map(lambda v: f"{v:.2f}%")
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)
    st.caption("Evaluated on held-out test set (n=1,311) \u00b7 5 training epochs \u00b7 results from thesis Chapter 6.2")

    st.markdown("#### Confusion matrices")
    cm_files = {
        "TF-IDF + LR": ASSETS_DIR / "tfidf_cm.png",
        "BioBERT": ASSETS_DIR / "biobert_cm.png",
        "PubMedBERT": ASSETS_DIR / "pubmedbert_cm.png",
    }
    cm_cols = st.columns(3)
    any_cm = False
    for col, (name, path) in zip(cm_cols, cm_files.items()):
        with col:
            if path.exists():
                st.image(str(path), caption=name, use_container_width=True)
                any_cm = True
            else:
                st.info(f"{name}: export `{path.name}` from your notebook into `assets/`.")
    if not any_cm:
        st.caption(
            "Tip: in your notebooks, save matrices with "
            "`fig.savefig('assets/<model>_cm.png', dpi=150, bbox_inches='tight')`."
        )

    st.markdown("#### Multi-metric radar chart")
    try:
        import plotly.graph_objects as go

        categories = ["Accuracy", "Precision", "Recall", "F1"]
        fig = go.Figure()
        for _, row in METRICS.iterrows():
            values = [
                row["Accuracy"],          # already 0-100
                row["Precision"] * 100,
                row["Recall"] * 100,
                row["F1"] * 100,
            ]
            fig.add_trace(go.Scatterpolar(
                r=values + values[:1],
                theta=categories + categories[:1],
                fill="toself",
                name=row["Model"],
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Radar chart unavailable: {exc}")

    st.markdown("#### Eight-bucket error analysis")
    bucket_csv = DATA_DIR / "eight_bucket.csv"
    bucket_png = ASSETS_DIR / "eight_bucket.png"
    if bucket_csv.exists():
        bucket_df = pd.read_csv(bucket_csv)
    else:
        bucket_df = EIGHT_BUCKET
        st.caption("Using embedded thesis values (Chapter 6.4). Drop `data/eight_bucket.csv` to override.")

    import matplotlib.pyplot as plt

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    bars = ax2.bar(bucket_df["bucket"], bucket_df["count"], color="#4c78a8")
    total = bucket_df["count"].sum()
    for bar, count in zip(bars, bucket_df["count"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            f"{count}\n({count / total:.1%})", ha="center", va="bottom", fontsize=8,
        )
    ax2.set_ylabel("Count")
    ax2.set_title("Which models got each test example right (n=1,311)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)

    if bucket_png.exists():
        with st.expander("Original thesis figure"):
            st.image(str(bucket_png), use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "MSc Dissertation \u00b7 Health Misinformation Detection \u00b7 2024-26 \u00b7 "
    "Inference uses real fine-tuned weights."
)
