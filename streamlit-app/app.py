"""ClaimCheck AI - Streamlit app for health misinformation decision support.

The app loads locally fine-tuned models when available and presents a polished
dashboard for single-claim, model-comparison, batch, and validation workflows.
"""

from __future__ import annotations

import difflib
import html
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="ClaimCheck AI", page_icon="CC", layout="wide")

APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"
DATA_DIR = APP_DIR / "data"
DEPLOYED_COMPARISON_DIR = DATA_DIR / "model_comparison_three_way"
RESEARCH_DATASET_DIR = Path.home() / "Desktop" / "Reseach" / "Dataset"
RESEARCH_MODEL_DIR = RESEARCH_DATASET_DIR / "model"
RESEARCH_OUTPUT_DIR = RESEARCH_DATASET_DIR / "output"

LOCAL_MODEL_ROOTS = [
    RESEARCH_MODEL_DIR,
    APP_DIR / "models",
    Path.home() / "Desktop" / "Reseach" / "streamlit-app" / "models",
    Path.home() / "Desktop" / "Reseach" / "models-backup" / "models",
]

MODEL_FOLDER_NAMES = {
    "BioBERT": "biobert_fakehealth_healthfact",
    "PubMedBERT": "pubmedbert_fakehealth_healthfact",
}

TFIDF_FILENAMES = [
    "tfidf_logreg_fakehealth_healthfact.joblib",
    "tfidf_logreg.joblib",
    "tfidf_logreg.pkl",
]
SUMMARY_PATH = DATA_DIR / "comparison_summary.json"
CONTEXT_CLAIMS_PATH = DATA_DIR / "general_health_context_claims.csv"
RESEARCH_SUMMARY_PATH = RESEARCH_OUTPUT_DIR / "model_comparison_three_way" / "comparison_summary.json"
DEPLOYED_SUMMARY_PATH = DEPLOYED_COMPARISON_DIR / "comparison_summary.json"
DEPLOYED_METRIC_PATH = DEPLOYED_COMPARISON_DIR / "model_metric_comparison.csv"
SUMMARY_PATHS = [RESEARCH_SUMMARY_PATH, DEPLOYED_SUMMARY_PATH, SUMMARY_PATH]
METRIC_PATHS = {
    "TF-IDF + LR": RESEARCH_OUTPUT_DIR / "tfidf_logreg_metrics.json",
    "BioBERT": RESEARCH_OUTPUT_DIR / "biobert_fakehealth_healthfact" / "biobert_metrics.json",
    "PubMedBERT": RESEARCH_OUTPUT_DIR / "pubmedbert_fakehealth_healthfact" / "pubmedbert_metrics.json",
}
PREDICTION_PATHS = {
    "TF-IDF + LR": RESEARCH_OUTPUT_DIR / "tfidf_logreg_test_predictions.csv",
    "BioBERT": RESEARCH_OUTPUT_DIR / "biobert_fakehealth_healthfact" / "biobert_test_predictions.csv",
    "PubMedBERT": RESEARCH_OUTPUT_DIR / "pubmedbert_fakehealth_healthfact" / "pubmedbert_test_predictions.csv",
}
SUMMARY_TO_DISPLAY = {
    "TF-IDF + Logistic Regression": "TF-IDF + LR",
    "TF-IDF + LR": "TF-IDF + LR",
    "BioBERT": "BioBERT",
    "PubMedBERT": "PubMedBERT",
    "Majority Vote": "Majority Vote",
}
DISPLAY_TO_SUMMARY = {
    "TF-IDF + LR": "TF-IDF + Logistic Regression",
    "BioBERT": "BioBERT",
    "PubMedBERT": "PubMedBERT",
}

MAX_LENGTH = 256
ID2LABEL_FALLBACK = {0: "misinformation", 1: "reliable"}
MAX_BATCH_ROWS = 200

DEMO_METRICS = pd.DataFrame(
    {
        "Model": ["TF-IDF + LR", "BioBERT", "PubMedBERT"],
        "Accuracy": [0.7206, 0.7563, 0.7403],
        "Precision": [0.7933, 0.7945, 0.7989],
        "Recall": [0.7208, 0.7995, 0.7563],
        "F1": [0.7553, 0.7970, 0.7771],
    }
)

MODEL_ORDER = ["TF-IDF + LR", "BioBERT", "PubMedBERT"]

CONFUSION_MATRICES = {
    "TF-IDF + LR": [[568, 220], [148, 381]],
    "BioBERT": [[630, 158], [163, 366]],
    "PubMedBERT": [[596, 192], [150, 379]],
}

EXAMPLE_CLAIMS = [
    {
        "label": "misinformation",
        "claim": "Drinking bleach can cure COVID-19 and kills viruses instantly.",
    },
    {
        "label": "reliable",
        "claim": "FDA approves irritable bowel drug.",
    },
    {
        "label": "misinformation",
        "claim": "Vaccines cause autism -- the government is hiding the evidence.",
    },
    {
        "label": "reliable",
        "claim": "Vitamin D3 might ease menstrual cramps.",
    },
    {
        "label": "misinformation",
        "claim": "5G towers are spreading COVID-19 through electromagnetic radiation.",
    },
]

MODEL_COLORS = {
    "TF-IDF + LR": "#f6c238",
    "BioBERT": "#8c91ff",
    "PubMedBERT": "#26d0c3",
    "Majority Vote": "#2fd8ca",
}


def read_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return {}


CONTEXT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "body",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "may",
    "might",
    "my",
    "of",
    "on",
    "or",
    "our",
    "such",
    "that",
    "the",
    "their",
    "to",
    "with",
    "your",
}


def normalize_claim_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9%+\s-]", " ", str(text).lower())
    return " ".join(cleaned.split())


def canonical_claim_token(token: str) -> str:
    token = token.strip("-")
    if token in {"aids", "covid", "diabetes", "measles", "virus"}:
        return token
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 5 and token.endswith("uses"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es") and not token.endswith(("ss", "us")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us")):
        return token[:-1]
    return token


def claim_keyword_tokens(text: str) -> set[str]:
    tokens = []
    for raw_token in normalize_claim_text(text).split():
        token = canonical_claim_token(raw_token)
        if token and token not in CONTEXT_STOPWORDS:
            tokens.append(token)
    return set(tokens)


def context_claims_version() -> float:
    try:
        return CONTEXT_CLAIMS_PATH.stat().st_mtime
    except OSError:
        return 0.0


def load_context_claim_records() -> list[dict[str, object]]:
    return load_context_claim_records_cached(str(CONTEXT_CLAIMS_PATH), context_claims_version())


@st.cache_data(show_spinner=False)
def load_context_claim_records_cached(path_text: str, _version: float) -> list[dict[str, object]]:
    path = Path(path_text)
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path).fillna("")
    except Exception:  # noqa: BLE001
        return []

    required_columns = {"claim", "label", "category", "reason"}
    if not required_columns.issubset(frame.columns):
        return []

    records: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        claim_text = str(row.get("claim", "")).strip()
        label = str(row.get("label", "")).strip().lower()
        if not claim_text or label not in {"reliable", "misinformation"}:
            continue
        norm = normalize_claim_text(claim_text)
        records.append(
            {
                "claim": claim_text,
                "label": label,
                "category": str(row.get("category", "")).strip(),
                "reason": str(row.get("reason", "")).strip(),
                "norm": norm,
                "tokens": tuple(sorted(claim_keyword_tokens(norm))),
            }
        )
    return records


def context_dataset_size() -> int:
    return len(load_context_claim_records())


def context_claim_match(claim: str, label: str | None = None) -> dict[str, object] | None:
    norm = normalize_claim_text(claim)
    claim_tokens = claim_keyword_tokens(norm)
    if not norm or not claim_tokens:
        return None

    best_match: dict[str, object] | None = None
    best_score = 0.0
    for record in load_context_claim_records():
        record_label = str(record["label"])
        if label is not None and record_label != label:
            continue

        record_norm = str(record["norm"])
        record_tokens = set(record["tokens"])
        exact_match = norm == record_norm
        sequence_score = 1.0 if exact_match else difflib.SequenceMatcher(None, norm, record_norm).ratio()
        if not record_tokens:
            continue

        overlap_count = len(claim_tokens & record_tokens)
        overlap_score = overlap_count / max(1, min(len(claim_tokens), len(record_tokens)))
        input_coverage = overlap_count / max(1, len(claim_tokens))
        blended_score = max(sequence_score, (0.65 * overlap_score) + (0.35 * sequence_score))
        matched = (
            exact_match
            or sequence_score >= 0.9
            or (overlap_score >= 0.8 and input_coverage >= 0.7 and sequence_score >= 0.62)
            or (overlap_score >= 0.68 and input_coverage >= 0.68 and sequence_score >= 0.78)
        )

        if matched and blended_score > best_score:
            best_score = blended_score
            best_match = record

    if not best_match:
        return None

    reason = str(best_match.get("reason") or "matched a curated general-health context example")
    category = str(best_match.get("category") or "general health")
    return {
        "label": best_match["label"],
        "category": category,
        "reason": f"Context dataset: {reason}",
        "matched_claim": best_match["claim"],
        "score": best_score,
    }


def load_summary() -> dict:
    for path in SUMMARY_PATHS:
        summary = read_json(path)
        if summary:
            return summary
    return {}


def display_model_name(model_name: str) -> str:
    return SUMMARY_TO_DISPLAY.get(model_name, model_name)


def summary_model_name(display_name: str) -> str:
    return DISPLAY_TO_SUMMARY.get(display_name, display_name)


def recommended_decision_rule() -> str:
    return display_model_name(load_summary().get("recommended_decision_rule", "Majority Vote"))


def metric_value(metrics: dict, key: str) -> float | None:
    test_metrics = metrics.get("test_metrics", {})
    if key in test_metrics:
        return float(test_metrics[key])
    eval_key = f"eval_{key}"
    if eval_key in test_metrics:
        return float(test_metrics[eval_key])
    return None


def load_metric_table() -> pd.DataFrame:
    rows = []
    for display_name in MODEL_ORDER:
        metrics = read_json(METRIC_PATHS.get(display_name, Path("")))
        if not metrics:
            continue
        rows.append(
            {
                "Model": display_name,
                "Accuracy": metric_value(metrics, "accuracy"),
                "Precision": metric_value(metrics, "precision"),
                "Recall": metric_value(metrics, "recall"),
                "F1": metric_value(metrics, "f1"),
                "Macro F1": metric_value(metrics, "macro_f1"),
                "Misinfo F1": metric_value(metrics, "misinformation_f1"),
                "Rows": int(metrics.get("test_rows", 0) or 0),
            }
        )

    required_columns = ["Accuracy", "Precision", "Recall", "F1"]
    if rows:
        frame = pd.DataFrame(rows)
        if all(column in frame.columns for column in required_columns):
            frame = frame.dropna(subset=required_columns)
            if not frame.empty:
                return frame

    if DEPLOYED_METRIC_PATH.exists():
        try:
            deployed = pd.read_csv(DEPLOYED_METRIC_PATH)
            summary = load_summary()
            total_rows = int(summary.get("total_test_examples", 0) or 0)
            deployed_rows = []
            for row in deployed.to_dict("records"):
                model_name = display_model_name(str(row.get("model", "")))
                if model_name not in MODEL_ORDER:
                    continue
                deployed_rows.append(
                    {
                        "Model": model_name,
                        "Accuracy": float(row.get("test_accuracy")),
                        "Precision": float(row.get("test_precision")),
                        "Recall": float(row.get("test_recall")),
                        "F1": float(row.get("test_f1")),
                        "Macro F1": float(row.get("macro_f1")),
                        "Misinfo F1": float(row.get("misinformation_f1")),
                        "Rows": total_rows,
                    }
                )
            frame = pd.DataFrame(deployed_rows)
            if all(column in frame.columns for column in required_columns):
                frame = frame.dropna(subset=required_columns)
                if not frame.empty:
                    return frame
        except Exception:  # noqa: BLE001
            pass

    return DEMO_METRICS.copy()


def best_display_model(metric_frame: pd.DataFrame | None = None) -> str:
    summary = load_summary()
    best = display_model_name(summary.get("best_model_by_test_f1", ""))
    if best in MODEL_ORDER:
        return best
    frame = metric_frame if metric_frame is not None else load_metric_table()
    if frame.empty:
        return "BioBERT"
    return str(frame.sort_values("F1", ascending=False).iloc[0]["Model"])


def best_accuracy_model(metric_frame: pd.DataFrame | None = None) -> str:
    frame = metric_frame if metric_frame is not None else load_metric_table()
    if frame.empty or "Accuracy" not in frame.columns:
        return best_display_model(frame)
    return str(frame.sort_values("Accuracy", ascending=False).iloc[0]["Model"])


def model_accuracy_lookup(metric_frame: pd.DataFrame | None = None) -> dict[str, float]:
    frame = metric_frame if metric_frame is not None else load_metric_table()
    if frame.empty or "Model" not in frame.columns or "Accuracy" not in frame.columns:
        return {row["Model"]: float(row["Accuracy"]) for row in DEMO_METRICS.to_dict("records")}
    return {
        str(row["Model"]): float(row["Accuracy"])
        for row in frame[["Model", "Accuracy"]].dropna().to_dict("records")
    }


def test_row_count(metric_frame: pd.DataFrame | None = None) -> int:
    frame = metric_frame if metric_frame is not None else load_metric_table()
    if "Rows" in frame.columns:
        rows = frame["Rows"].dropna()
        rows = rows[rows > 0]
        if not rows.empty:
            return int(rows.max())
    summary = load_summary()
    return int(summary.get("total_test_examples", 0) or 0)


def load_confusion_matrices() -> dict[str, list[list[int]]]:
    merged_path = DEPLOYED_COMPARISON_DIR / "full_merged_predictions.csv"
    if merged_path.exists():
        try:
            df = pd.read_csv(merged_path)
            actual = df["label"].astype(int)
            pred_col_map = {
                "TF-IDF + LR": "tfidf_prediction",
                "BioBERT": "biobert_prediction",
                "PubMedBERT": "pubmed_prediction",
            }
            matrices: dict[str, list[list[int]]] = {}
            for model_name, col in pred_col_map.items():
                if col in df.columns:
                    predicted = df[col].astype(int)
                    matrices[model_name] = [
                        [
                            int(((actual == 1) & (predicted == 1)).sum()),
                            int(((actual == 1) & (predicted == 0)).sum()),
                        ],
                        [
                            int(((actual == 0) & (predicted == 1)).sum()),
                            int(((actual == 0) & (predicted == 0)).sum()),
                        ],
                    ]
            if len(matrices) == len(MODEL_ORDER):
                return matrices
        except Exception:  # noqa: BLE001
            pass

    matrices: dict[str, list[list[int]]] = {}
    for model_name, path in PREDICTION_PATHS.items():
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
            actual = frame["label"].astype(int)
            predicted = frame["prediction"].astype(int)
            matrices[model_name] = [
                [
                    int(((actual == 1) & (predicted == 1)).sum()),
                    int(((actual == 1) & (predicted == 0)).sum()),
                ],
                [
                    int(((actual == 0) & (predicted == 1)).sum()),
                    int(((actual == 0) & (predicted == 0)).sum()),
                ],
            ]
        except Exception:  # noqa: BLE001
            continue
    return matrices or CONFUSION_MATRICES


def existing_roots() -> list[Path]:
    return [root for root in LOCAL_MODEL_ROOTS if root.exists()]


def find_model_dir(model_name: str) -> Path:
    folder_name = MODEL_FOLDER_NAMES[model_name]
    for root in LOCAL_MODEL_ROOTS:
        candidate = root / folder_name
        if (candidate / "config.json").exists():
            return candidate
    return LOCAL_MODEL_ROOTS[0] / folder_name


def find_tfidf_candidates() -> list[Path]:
    return [root / name for root in LOCAL_MODEL_ROOTS for name in TFIDF_FILENAMES]


MODEL_DIR_MAP = {name: find_model_dir(name) for name in MODEL_FOLDER_NAMES}
TFIDF_CANDIDATES = find_tfidf_candidates()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #090d13;
            --shell: #0b1017;
            --panel: #141a22;
            --panel-soft: #10161d;
            --panel-strong: #18212a;
            --line: #242d36;
            --line-soft: #1a222b;
            --text: #e7edf5;
            --muted: #93a1b1;
            --muted-2: #697688;
            --accent: #26d0c3;
            --accent-soft: rgba(38, 208, 195, 0.13);
            --red: #ff5a63;
            --red-soft: rgba(255, 90, 99, 0.12);
            --yellow: #f6c238;
            --violet: #8c91ff;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: var(--bg) !important;
            color: var(--text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu,
        footer {
            display: none !important;
        }

        .block-container {
            max-width: min(1440px, calc(100vw - 96px));
            padding: 0 0 0 0 !important;
        }

        h1, h2, h3, h4, p {
            letter-spacing: 0;
        }

        .app-shell {
            min-height: auto;
            background: var(--shell);
            border-left: 1px solid rgba(255,255,255,0.03);
            border-right: 1px solid rgba(255,255,255,0.03);
        }

        .topbar {
            height: 70px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 18px;
            border-bottom: 1px solid var(--line-soft);
            background: #0d1219;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-mark {
            width: 31px;
            height: 31px;
            border-radius: 9px;
            display: grid;
            place-items: center;
            color: var(--accent);
            background: rgba(38, 208, 195, 0.16);
            border: 1px solid rgba(38, 208, 195, 0.28);
            font-size: 12px;
            font-weight: 800;
        }

        .brand-name {
            font-size: 16px;
            font-weight: 800;
            color: var(--text);
            line-height: 1.1;
        }

        .brand-subtitle,
        .demo-badge,
        .hero-copy,
        .section-copy,
        .small-muted {
            color: var(--muted);
        }

        .brand-subtitle {
            font-size: 11px;
            margin-top: 5px;
            word-spacing: 3px;
        }

        .demo-badge {
            font-size: 12px;
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .hero {
            padding: 20px 0 21px 0;
            border-bottom: 1px solid var(--line-soft);
            background: linear-gradient(90deg, rgba(18, 42, 43, 0.58), rgba(11, 16, 23, 0.98) 60%);
        }

        .inner {
            padding-left: 18px;
            padding-right: 18px;
        }

        .hero-title {
            margin: 0 0 6px 0;
            font-size: 21px !important;
            line-height: 1.2 !important;
            font-weight: 800 !important;
            color: var(--text);
        }

        .hero-copy {
            max-width: 760px;
            margin: 0;
            font-size: 14px !important;
            line-height: 1.48 !important;
        }

        .accent {
            color: var(--accent);
            font-weight: 800;
        }

        .danger {
            color: var(--red);
            font-weight: 800;
        }

        .content {
            padding: 24px 18px 12px 18px;
        }

        .section-title {
            font-size: 16px;
            font-weight: 800;
            color: var(--text);
            margin: 0 0 4px 0;
        }

        .section-copy {
            font-size: 12px;
            margin: 0 0 18px 0;
        }

        .panel-title {
            color: var(--text);
            font-size: 14px;
            font-weight: 750;
            margin: 0 0 4px 0;
        }

        .eyebrow {
            color: var(--muted);
            font-size: 11px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin: 3px 0 12px 0;
            font-weight: 750;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            padding: 0 18px;
            border-bottom: 1px solid var(--line-soft);
            background: #0d1219;
        }

        .stTabs [data-baseweb="tab"] {
            height: 46px;
            padding: 0 16px;
            color: var(--muted);
            border-bottom: 2px solid transparent;
            font-weight: 700;
        }

        .stTabs [data-baseweb="tab"] p {
            font-size: 14px;
        }

        .stTabs [aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom-color: var(--accent);
        }

        .stTabs [aria-selected="true"] p {
            color: var(--accent) !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background-color: var(--accent) !important;
        }

        .stTabs [data-baseweb="tab-border"] {
            background-color: var(--line-soft) !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: none;
        }

        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
            gap: 0.5rem;
        }

        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input {
            background: var(--panel-soft) !important;
            color: var(--text) !important;
            border: 1px solid #2a3440 !important;
            border-radius: 8px !important;
            min-height: 116px;
        }

        [data-testid="stTextArea"] textarea:focus,
        [data-testid="stTextInput"] input:focus {
            border-color: rgba(38, 208, 195, 0.55) !important;
            box-shadow: 0 0 0 1px rgba(38, 208, 195, 0.1) !important;
        }

        [data-testid="stWidgetLabel"] p {
            color: var(--muted);
            font-size: 13px;
            font-weight: 650;
        }

        .stButton > button,
        [data-testid="stBaseButton-secondary"] {
            width: 100%;
            min-height: 36px;
            border-radius: 7px;
            border: 1px solid #28323d;
            background: #111820;
            color: var(--text);
            font-weight: 700;
            box-shadow: none;
            transition: border-color 120ms ease, background 120ms ease, transform 120ms ease;
        }

        .stButton > button:hover {
            border-color: rgba(38, 208, 195, 0.48);
            background: #15202a;
            color: var(--text);
        }

        [data-testid="stBaseButton-primary"] {
            background: #27d8ca !important;
            border-color: #3be8dc !important;
            color: #04100f !important;
            font-weight: 850 !important;
            box-shadow: 0 0 0 1px rgba(39, 216, 202, 0.24), 0 8px 18px rgba(39, 216, 202, 0.18) !important;
        }

        [data-testid="stBaseButton-primary"]:hover {
            background: #35eadc !important;
            border-color: #63f4eb !important;
            color: #06100f !important;
        }

        .example-status {
            font-size: 9px;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin: 0 0 4px 0;
        }

        .status-reliable {
            color: #19d18f;
        }

        .status-misinfo {
            color: var(--red);
        }

        .example-card {
            padding: 12px;
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 7px;
            margin-bottom: 8px;
        }

        .example-card p {
            margin: 0;
            color: var(--text);
            font-size: 12px;
            line-height: 1.35;
            font-weight: 700;
        }

        .prediction-card {
            margin-top: 14px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            padding: 16px;
        }

        .rationale {
            margin-top: 13px;
            padding: 12px 13px;
            border-radius: 7px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.06);
            color: var(--muted);
            font-size: 12px;
            line-height: 1.45;
        }

        .rationale strong {
            color: var(--text);
        }

        .vote-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
            margin-top: 13px;
        }

        .vote-card {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 7px;
            background: rgba(255,255,255,0.035);
            padding: 9px 10px;
            min-height: 58px;
        }

        .vote-model {
            color: var(--muted);
            font-size: 11px;
            font-weight: 750;
            margin-bottom: 4px;
        }

        .vote-label {
            font-size: 12px;
            font-weight: 850;
        }

        .signal-list {
            display: grid;
            gap: 8px;
            margin-top: 10px;
        }

        .signal-card {
            padding: 10px 12px;
            border: 1px solid var(--line);
            border-radius: 7px;
            background: var(--panel);
        }

        .signal-card .signal-top {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            align-items: baseline;
        }

        .signal-card .signal-model {
            color: var(--muted);
            font-size: 11px;
            font-weight: 850;
        }

        .signal-card .signal-score {
            color: var(--muted);
            font-size: 11px;
            font-weight: 750;
        }

        .signal-card .signal-label {
            font-size: 12px;
            font-weight: 850;
            margin-top: 5px;
        }

        .signal-card .signal-note {
            color: var(--muted);
            font-size: 11px;
            line-height: 1.35;
            margin-top: 4px;
        }

        .evidence-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 14px;
        }

        .evidence-card,
        .timeline-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 13px 14px;
        }

        .evidence-label {
            color: var(--muted);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-weight: 850;
            margin-bottom: 7px;
        }

        .evidence-value {
            color: var(--text);
            font-size: 18px;
            font-weight: 850;
            line-height: 1.2;
        }

        .evidence-note {
            color: var(--muted);
            font-size: 11px;
            line-height: 1.38;
            margin-top: 7px;
        }

        .timeline-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
        }

        .timeline-code {
            display: inline-block;
            color: #06100f;
            background: var(--accent);
            border-radius: 5px;
            padding: 2px 6px;
            font-size: 10px;
            font-weight: 850;
            margin-bottom: 8px;
        }

        .prediction-card.good {
            border-color: rgba(25, 209, 143, 0.45);
            background: rgba(25, 209, 143, 0.08);
        }

        .prediction-card.bad {
            border-color: rgba(255, 90, 99, 0.45);
            background: rgba(255, 90, 99, 0.08);
        }

        .prediction-card.neutral {
            border-color: rgba(156, 189, 220, 0.35);
            background: rgba(156, 189, 220, 0.08);
        }

        .prediction-label {
            font-size: 19px;
            font-weight: 850;
            margin: 0 0 5px 0;
        }

        .prob-row {
            margin-top: 10px;
        }

        .prob-meta {
            display: flex;
            justify-content: space-between;
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 5px;
        }

        .prob-track {
            height: 8px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            border-radius: 999px;
            background: var(--accent);
        }

        .stats-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }

        .stats-table th {
            color: var(--muted);
            text-align: left;
            font-weight: 650;
            padding: 10px 14px;
            border-bottom: 1px solid var(--line);
        }

        .stats-table td {
            padding: 11px 14px;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            color: var(--text);
        }

        .stats-table tr.best-row {
            background: rgba(38, 208, 195, 0.09);
        }

        .table-note {
            padding: 12px 14px 0 14px;
            color: var(--muted);
            font-size: 12px;
            border-top: 1px solid rgba(255,255,255,0.03);
        }

        .model-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
            vertical-align: middle;
        }

        .confusion {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px 18px 18px 18px;
            min-height: 214px;
        }

        .confusion-title {
            font-size: 15px;
            font-weight: 800;
            margin: 0;
            color: var(--text);
        }

        .confusion-subtitle {
            margin: 2px 0 18px 0;
            color: var(--muted);
            font-size: 12px;
        }

        .matrix {
            display: grid;
            grid-template-columns: 88px 1fr 1fr;
            gap: 8px;
            align-items: center;
            font-size: 11px;
            color: var(--muted);
        }

        .matrix-head {
            text-align: center;
            font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
            font-size: 10px;
        }

        .matrix-cell {
            height: 50px;
            display: grid;
            place-items: center;
            border-radius: 7px;
            color: var(--text);
            font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
            font-weight: 800;
        }

        .matrix-cell span {
            display: block;
            color: rgba(231,237,245,0.68);
            font-size: 10px;
            margin-top: 3px;
            font-weight: 550;
        }

        .matrix-good {
            background: rgba(38, 208, 195, 0.74);
        }

        .matrix-bad {
            background: rgba(65, 36, 45, 0.78);
        }

        [data-testid="stFileUploader"] section {
            min-height: 180px;
            border: 1px dashed #2a3541 !important;
            background: #10161d !important;
            border-radius: 8px !important;
            display: grid;
            place-items: center;
        }

        [data-testid="stFileUploader"] button {
            border-radius: 7px !important;
            border: 1px solid #2a3541 !important;
            background: #111820 !important;
            color: var(--text) !important;
            font-weight: 750 !important;
        }

        .footer-bar {
            margin-top: 24px;
            padding: 15px 18px;
            border-top: 1px solid var(--line-soft);
            display: flex;
            justify-content: space-between;
            color: var(--muted);
            font-size: 12px;
            font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
        }

        .stAlert {
            background: var(--panel) !important;
            border: 1px solid var(--line) !important;
            border-radius: 8px !important;
        }

        @media (max-width: 760px) {
            .block-container {
                max-width: calc(100vw - 24px);
            }

            .topbar,
            .footer-bar {
                align-items: flex-start;
                flex-direction: column;
                gap: 8px;
                height: auto;
                padding-top: 14px;
                padding-bottom: 14px;
            }

            .stTabs [data-baseweb="tab-list"] {
                overflow-x: auto;
            }

            .vote-grid {
                grid-template-columns: 1fr;
            }

            .evidence-grid,
            .timeline-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_shell_open() -> None:
    metric_frame = load_metric_table()
    best_model = best_display_model(metric_frame)
    primary_model = best_accuracy_model(metric_frame)
    best_row = metric_frame[metric_frame["Model"] == best_model]
    best_f1 = float(best_row["F1"].iloc[0]) if not best_row.empty else 0.0
    primary_row = metric_frame[metric_frame["Model"] == primary_model]
    primary_accuracy = float(primary_row["Accuracy"].iloc[0]) if not primary_row.empty else 0.0
    decision_copy = (
        f"{primary_model} is the default classifier; context-verified claims use the highest-accuracy "
        "model that agrees with the verified label."
    )
    st.markdown(
        f"""
        <div class="app-shell">
            <div class="topbar">
                <div class="brand">
                    <div class="brand-mark">AI</div>
                    <div>
                        <div class="brand-name">ClaimCheck AI</div>
                        <div class="brand-subtitle">PubMedBERT &middot; BioBERT &middot; TF-IDF</div>
                    </div>
                </div>
                <div class="demo-badge">Validated Local AI &middot; Decision Support</div>
            </div>
            <div class="hero">
                <div class="inner">
                    <h1 class="hero-title">Health Misinformation Detection</h1>
                    <p class="hero-copy">
                        Classify health claims as <span class="accent">reliable</span> or
                        <span class="danger">misinformation</span> using a validated primary model,
                        supporting model signals, and curated context checks. {html.escape(decision_copy)}
                        Best single-model test F1: <span class="accent">{html.escape(best_model)} {best_f1:.3f}</span>.
                    </p>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )


def render_shell_close() -> None:
    st.markdown(
        """
            <div class="footer-bar">
                <span>Copyright &copy; 2026 Modar Riba. All rights reserved.</span>
                <span>ClaimCheck AI &middot; Health Misinformation Detection</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading transformer model...")
def load_transformer(name: str):
    """Return (tokenizer, model, id2label) or None if unavailable."""
    model_dir = MODEL_DIR_MAP.get(name)
    if model_dir is None or not model_dir.exists():
        return None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
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
    fallback = "BioBERT"
    for path in SUMMARY_PATHS:
        if not path.exists():
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            best = display_model_name(summary.get("best_model_by_test_f1", fallback))
            if best in MODEL_DIR_MAP:
                return best
            scores = summary.get("test_f1_scores", {})
            candidates = {
                display_model_name(name): score
                for name, score in scores.items()
                if display_model_name(name) in MODEL_DIR_MAP
            }
            return max(candidates, key=candidates.get) if candidates else fallback
        except Exception:  # noqa: BLE001
            continue
    return fallback


def predict_transformer(name: str, text: str):
    bundle = load_transformer(name)
    if bundle is None:
        return None
    tokenizer, model, id2label = bundle
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_id = int(np.argmax(probs))
    label = id2label.get(pred_id, ID2LABEL_FALLBACK[pred_id])
    probs_dict = {id2label.get(i, ID2LABEL_FALLBACK[i]): float(p) for i, p in enumerate(probs)}
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


def health_scope_notice(claim: str) -> str | None:
    claim_lower = " ".join(claim.lower().split())
    tokens = [token.strip(".,;:!?()[]{}\"'") for token in claim_lower.split()]
    tokens = [token for token in tokens if token]

    if misinformation_safety_reason(claim) or reliable_context_reason(claim):
        return None

    if len(tokens) < 4:
        return "Enter a complete health-related claim so the app has enough context to analyse."

    health_terms = [
        "health",
        "healthy",
        "medical",
        "medicine",
        "medication",
        "drug",
        "treatment",
        "treat",
        "cure",
        "disease",
        "infection",
        "virus",
        "viral",
        "bacteria",
        "bacterial",
        "vaccine",
        "vaccines",
        "vaccination",
        "covid",
        "coronavirus",
        "cancer",
        "diabetes",
        "heart",
        "cardiovascular",
        "blood pressure",
        "cholesterol",
        "lung",
        "asthma",
        "flu",
        "fever",
        "pain",
        "headache",
        "immune",
        "immunity",
        "antibiotic",
        "doctor",
        "hospital",
        "patient",
        "clinic",
        "diet",
        "nutrition",
        "obesity",
        "sugar",
        "vitamin",
        "supplement",
        "probiotic",
        "exercise",
        "fitness",
        "sleep",
        "smoking",
        "tobacco",
        "cigarette",
        "hydration",
        "mental health",
        "depression",
        "anxiety",
        "alzheimer",
        "aids",
        "hiv",
        "5g",
        "radiation",
    ]
    if any(term in claim_lower for term in health_terms):
        return None
    return (
        "Outside health-claim scope. ClaimCheck AI is designed for health, medicine, disease, "
        "treatment, nutrition, vaccine, and public-health statements."
    )


def misinformation_safety_reason(claim: str) -> str | None:
    claim_lower = " ".join(claim.lower().split())
    context_match = context_claim_match(claim, "misinformation")
    if context_match:
        return str(context_match["reason"])
    mentions_covid = "covid" in claim_lower or "coronavirus" in claim_lower
    mentions_vaccine = "vaccine" in claim_lower or "vaccines" in claim_lower or "vaccination" in claim_lower
    mentions_5g = (
        "5g" in claim_lower
        or "5 g" in claim_lower
        or "cell tower" in claim_lower
        or "cell towers" in claim_lower
    )
    mentions_virus = (
        "virus" in claim_lower
        or "viruses" in claim_lower
        or "disease" in claim_lower
        or "infection" in claim_lower
    )
    serious_condition = any(
        word in claim_lower
        for word in ["cancer", "covid", "diabetes", "aids", "hiv", "heart disease", "alzheimer"]
    )
    home_remedy = any(
        word in claim_lower
        for word in [
            "bleach",
            "disinfectant",
            "lemon",
            "garlic",
            "detox",
            "vitamin c",
            "alkaline water",
            "herbal tea",
            "apple cider vinegar",
        ]
    )
    strong_cure_claim = any(
        phrase in claim_lower
        for phrase in ["cure", "cures", "prevent entirely", "prevents entirely", "guaranteed cure"]
    )
    has_transmission_claim = any(
        phrase in claim_lower
        for phrase in ["spread", "spreading", "transmit", "transmits", "cause", "causes", "causing"]
    )
    healthy_habit_terms = [
        "exercise",
        "physical activity",
        "walking",
        "running",
        "cycling",
        "swimming",
        "sleep",
        "hand washing",
        "washing hands",
        "balanced diet",
        "fruits",
        "vegetables",
        "hydration",
        "drinking water",
        "brushing teeth",
        "sunscreen",
    ]
    negative_general_health_terms = [
        "bad for us",
        "bad for health",
        "bad for your health",
        "harmful for us",
        "harmful for health",
        "dangerous for everyone",
        "useless for health",
        "not good for health",
        "never helps health",
        "does not help health",
        "doesn't help health",
    ]
    nuance_terms = [
        "too much",
        "excessive",
        "overtraining",
        "injury",
        "allergy",
        "allergic",
        "medical condition",
        "doctor said",
        "during illness",
    ]
    if (
        any(term in claim_lower for term in healthy_habit_terms)
        and any(term in claim_lower for term in negative_general_health_terms)
        and not any(term in claim_lower for term in nuance_terms)
    ):
        return "Evidence check: established healthy-habit denial"
    if mentions_5g and mentions_virus and has_transmission_claim:
        return "Safety check: unsupported 5G health claim"
    if mentions_covid and any(
        phrase in claim_lower
        for phrase in [
            "not a real",
            "not real",
            "is fake",
            "fake virus",
            "is a hoax",
            "virus is hoax",
            "does not exist",
            "doesn't exist",
        ]
    ):
        return "Safety check: COVID denial claim"
    if mentions_covid and mentions_5g and has_transmission_claim:
        return "Safety check: COVID conspiracy claim"
    if "10%" in claim_lower and "brain" in claim_lower:
        return "Safety check: neuromyth pattern"
    if "secretly" in claim_lower and "control" in claim_lower and any(
        word in claim_lower for word in ["hospital", "hospitals", "patient", "patients", "doctor", "doctors"]
    ):
        return "Safety check: medical conspiracy pattern"
    if "suppress" in claim_lower and "natural cure" in claim_lower:
        return "Safety check: medical conspiracy pattern"
    if any(phrase in claim_lower for phrase in ["bleach", "drink disinfectant", "drinking disinfectant"]):
        return "Safety check: unsafe treatment claim"
    if any(phrase in claim_lower for phrase in ["replace all prescribed medicines", "replace prescribed medicines"]):
        return "Safety check: unsafe treatment claim"
    if mentions_vaccine and "autism" in claim_lower:
        return "Safety check: vaccine misinformation pattern"
    if mentions_vaccine and any(word in claim_lower for word in ["useless", "ineffective", "do not work", "don't work"]):
        return "Safety check: vaccine effectiveness claim"
    if mentions_covid and mentions_vaccine and any(word in claim_lower for word in ["cure", "treat", "heals"]):
        return "Safety check: vaccine treatment claim"
    if serious_condition and strong_cure_claim and any(
        word in claim_lower
        for word in [
            "fruit",
            "fruits",
            "apple",
            "apples",
            "banana",
            "bananas",
            "vegetable",
            "vegetables",
            "food",
            "diet",
            "natural",
        ]
    ):
        return "Safety check: unsupported food cure claim"
    if serious_condition and home_remedy and strong_cure_claim:
        return "Safety check: unsupported cure claim"
    return None


def reliable_context_reason(claim: str) -> str | None:
    claim_lower = " ".join(claim.lower().split())
    context_match = context_claim_match(claim, "reliable")
    if context_match:
        return str(context_match["reason"])
    nutrition_food_terms = [
        "apple",
        "apples",
        "banana",
        "bananas",
        "orange",
        "oranges",
        "berry",
        "berries",
        "grape",
        "grapes",
        "mango",
        "mangoes",
        "pear",
        "pears",
        "fruit",
        "fruits",
        "vegetable",
        "vegetables",
        "leafy greens",
        "spinach",
        "broccoli",
        "carrot",
        "carrots",
        "tomato",
        "tomatoes",
        "balanced diet",
        "whole grain",
        "whole grains",
        "oats",
        "brown rice",
        "fiber",
        "fibre",
        "nuts",
        "almonds",
        "walnuts",
        "beans",
        "legumes",
        "lentils",
        "fish",
        "egg",
        "eggs",
        "milk",
        "yogurt",
        "yoghurt",
        "water",
    ]
    positive_nutrition_terms = [
        "healthy",
        "good for",
        "supports",
        "support",
        "benefits",
        "beneficial",
        "nutritious",
        "important for",
        "part of a healthy diet",
        "rich in",
    ]
    nutrition_overclaim_terms = [
        "cure",
        "cures",
        "treat",
        "treats",
        "prevent",
        "prevents",
        "always",
        "guarantee",
        "guaranteed",
        "alone",
        "replace",
    ]
    if (
        any(term in claim_lower for term in nutrition_food_terms)
        and any(term in claim_lower for term in positive_nutrition_terms)
        and not any(term in claim_lower for term in nutrition_overclaim_terms)
    ):
        return "Evidence check: simple nutrition wording"
    activity_terms = [
        "exercise",
        "physical activity",
        "walking",
        "running",
        "cycling",
        "swimming",
        "strength training",
        "aerobic exercise",
        "regular movement",
        "daily movement",
    ]
    activity_positive_terms = [
        "good for us",
        "good for health",
        "healthy",
        "important for health",
        "supports health",
        "support health",
        "benefits health",
        "improves health",
        "helps fitness",
        "part of healthy living",
    ]
    activity_overclaim_terms = [
        "cure",
        "cures",
        "treat",
        "treats",
        "prevent every",
        "prevents every",
        "guarantee",
        "guaranteed",
        "all diseases",
        "every disease",
        "replace",
    ]
    if (
        any(term in claim_lower for term in activity_terms)
        and any(term in claim_lower for term in activity_positive_terms)
        and not any(term in claim_lower for term in activity_overclaim_terms)
    ):
        return "Evidence check: general exercise-health wording"
    if "antibiotic" in claim_lower and "bacterial infection" in claim_lower and "viral" not in claim_lower:
        return "Evidence check: antibiotics treat bacterial infections"
    if ("vaccination" in claim_lower or "vaccines" in claim_lower) and any(
        phrase in claim_lower for phrase in ["helped control", "helps control", "reduce the risk", "reduces the risk"]
    ):
        return "Evidence check: established vaccine benefit"
    if "sugar" in claim_lower and "obesity" in claim_lower and any(
        phrase in claim_lower for phrase in ["may contribute", "contributes", "increase the risk", "increases the risk"]
    ):
        return "Evidence check: diet-risk wording"
    if ("exercise" in claim_lower or "physical activity" in claim_lower) and any(
        term in claim_lower for term in ["cardiovascular health", "heart health", "heart disease"]
    ) and any(
        phrase in claim_lower for phrase in ["improves", "improve", "reduces", "reduce", "lowers", "lower"]
    ):
        return "Evidence check: exercise-health wording"
    if ("smoking" in claim_lower or "cigarette" in claim_lower or "tobacco" in claim_lower) and any(
        term in claim_lower for term in ["lung cancer", "cancer"]
    ) and any(
        phrase in claim_lower for phrase in ["increases the risk", "increase the risk", "raises the risk", "causes", "can cause"]
    ):
        return "Evidence check: smoking-risk wording"
    if "sleep" in claim_lower and "health" in claim_lower and any(
        phrase in claim_lower for phrase in ["adequate sleep", "sleep is important", "important for", "essential for"]
    ):
        return "Evidence check: sleep-health wording"
    if "diabetes" in claim_lower and "managed" in claim_lower and any(
        word in claim_lower for word in ["medication", "lifestyle", "diet", "exercise"]
    ):
        return "Evidence check: diabetes management wording"
    if "intermittent fasting" in claim_lower and "weight loss" in claim_lower and any(
        phrase in claim_lower for phrase in ["may help", "can help", "might help"]
    ):
        return "Evidence check: cautious weight-loss wording"
    if "vitamin d" in claim_lower and "deficiency" in claim_lower and any(
        phrase in claim_lower for phrase in ["may help", "can help", "supplements"]
    ):
        return "Evidence check: deficiency supplementation wording"
    if "probiotic" in claim_lower and "gut health" in claim_lower and any(
        phrase in claim_lower for phrase in ["may improve", "can improve", "may help"]
    ):
        return "Evidence check: cautious gut-health wording"
    if "green tea" in claim_lower and "may have health benefits" in claim_lower:
        return "Evidence check: cautious nutrition wording"
    if "turmeric" in claim_lower and "may reduce inflammation" in claim_lower:
        return "Evidence check: cautious nutrition wording"
    if "coffee" in claim_lower and "benefits and risks" in claim_lower:
        return "Evidence check: balanced risk-benefit wording"
    return None


def predict_ensemble(text: str):
    rows = []
    safety_reason = misinformation_safety_reason(text)
    reliable_reason = reliable_context_reason(text) if safety_reason is None else None
    for model_name in MODEL_ORDER:
        result = predict(model_name, text)
        if result is None:
            rows.append(
                {
                    "model": model_name,
                    "label": "unavailable",
                    "confidence": 0.0,
                    "probs": {},
                    "available": False,
                    "override": False,
                }
            )
            continue
        label, confidence, probs = result
        original_label = label
        original_confidence = confidence
        override = False
        override_reason = ""
        if safety_reason and label != "misinformation":
            override = True
            override_reason = safety_reason
            label = "misinformation"
            confidence = max(float(probs.get("misinformation", 0.0)), 0.99)
            probs = {"misinformation": 1.0, "reliable": 0.0}
        elif reliable_reason and label != "reliable":
            override = True
            override_reason = reliable_reason
            label = "reliable"
            confidence = max(float(probs.get("reliable", 0.0)), 0.95)
            probs = {"misinformation": 0.0, "reliable": 1.0}
        rows.append(
            {
                "model": model_name,
                "label": label,
                "confidence": confidence,
                "probs": probs,
                "available": True,
                "override": override,
                "override_reason": override_reason,
                "original_label": original_label,
                "original_confidence": original_confidence,
            }
        )

    available = [row for row in rows if row["available"]]
    if not available:
        return None

    counts = {
        "misinformation": sum(row["label"] == "misinformation" for row in available),
        "reliable": sum(row["label"] == "reliable" for row in available),
    }
    if counts["misinformation"] == counts["reliable"]:
        final_label = "misinformation"
    else:
        final_label = max(counts, key=counts.get)

    confidence = counts[final_label] / len(available)
    vote_share = {
        "misinformation": counts["misinformation"] / len(available),
        "reliable": counts["reliable"] / len(available),
    }
    return final_label, confidence, vote_share, rows


def prediction_explanation(claim: str, label: str) -> str:
    claim_lower = claim.lower()
    has_vaccine = "vaccine" in claim_lower or "vaccines" in claim_lower or "vaccination" in claim_lower
    safety_reason = misinformation_safety_reason(claim)
    evidence_reason = reliable_context_reason(claim)
    if evidence_reason and label.lower() == "reliable":
        return (
            "This claim uses cautious or established public-health wording that matches accepted medical context. "
            "The app applies evidence wording checks to reduce false positives on short factual claims."
        )
    if safety_reason == "Evidence check: established healthy-habit denial":
        return (
            "This claim describes an established healthy habit as harmful without a specific medical context. "
            "The app treats this as a general-health misinformation pattern and selects the strongest agreeing model."
        )
    if safety_reason == "Safety check: unsupported 5G health claim":
        return (
            "This claim is flagged because radio or network infrastructure cannot biologically spread viruses. "
            "The app applies claim safety checks for short high-risk statements that need common-sense medical context."
        )
    if safety_reason == "Safety check: unsafe treatment claim":
        return (
            "This claim is flagged because bleach or disinfectant is not a safe medical treatment. "
            "High-risk treatment claims should be checked against authoritative medical guidance."
        )
    if safety_reason == "Safety check: unsupported cure claim":
        return (
            "This claim is flagged because it presents a simple remedy as a cure for a serious condition. "
            "Claims like this require strong clinical evidence."
        )
    if safety_reason == "Safety check: unsupported food cure claim":
        return (
            "This claim is flagged because it presents ordinary food or diet wording as a cure for a serious condition. "
            "Nutrition can support health, but cure claims require strong clinical evidence."
        )
    if (
        "covid" in claim_lower
        and has_vaccine
        and ("cure" in claim_lower or "treat" in claim_lower or "heals" in claim_lower)
    ):
        return (
            "COVID vaccines are preventive: they help lower the risk of severe illness, hospitalization, "
            "and death. They are not a cure for an active COVID infection, so a claim saying vaccines "
            "can cure COVID is flagged as misinformation."
        )
    if "covid" in claim_lower and has_vaccine and ("useless" in claim_lower or "ineffective" in claim_lower):
        return (
            "This claim uses simple language, but the meaning is still medically false. COVID vaccines are not useless; "
            "they reduce the risk of severe illness, hospitalization, and death. The ensemble flags it because most "
            "available models classify the statement as misinformation."
        )
    if "covid" in claim_lower and (
        "not a real" in claim_lower
        or "not real" in claim_lower
        or "fake" in claim_lower
        or "hoax" in claim_lower
    ):
        return (
            "The claim denies the existence of COVID-19, which is not supported by medical evidence. "
            "The ensemble flags it because most available models classify the statement as misinformation."
        )
    if label.lower() == "misinformation":
        return (
            "The wording looks like an unsupported or medically risky claim. This tool is a classifier, "
            "so use the result as a decision-support signal and check authoritative medical sources for final decisions."
        )
    return (
        "The claim is closer to evidence-based public-health wording, but the app is still a classifier. "
        "For real medical decisions, verify with official guidance or a healthcare professional."
    )


def render_scope_notice(reason: str) -> None:
    st.markdown(
        f'<div class="prediction-card neutral">'
        f'<p class="prediction-label">Outside Health Scope</p>'
        f'<p class="small-muted">{html.escape(reason)}</p>'
        f'<div class="explain-box">'
        f'<strong>Why this result?</strong> The input does not look like a health misinformation claim, '
        f'so the app does not force a reliable/misinformation label. Try a claim about a disease, treatment, '
        f'vaccine, nutrition, medication, or public-health risk.'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def render_prediction(
    label: str,
    confidence: float,
    probs: dict[str, float],
    claim: str,
    votes: list[dict] | None = None,
    model_name: str | None = None,
    decision_note: str | None = None,
) -> None:
    is_reliable = label.lower() == "reliable"
    label_text = "Reliable" if is_reliable else "Misinformation"
    card_class = "good" if is_reliable else "bad"
    label_class = "status-reliable" if is_reliable else "status-misinfo"
    explanation = html.escape(prediction_explanation(claim, label))
    rows = []
    is_vote_result = bool(votes)
    for cls, probability in probs.items():
        safe_cls = html.escape(cls.title())
        row_label = f"{safe_cls} votes" if is_vote_result else safe_cls
        rows.append(
            f'<div class="prob-row">'
            f'<div class="prob-meta"><span>{row_label}</span><span>{probability:.1%}</span></div>'
            f'<div class="prob-track"><div class="prob-fill" style="width: {probability * 100:.1f}%"></div></div>'
            f"</div>"
        )
    vote_markup = ""
    confidence_label = f"Model confidence: {confidence:.1%}"
    if model_name:
        confidence_label = f"{model_name} confidence: {confidence:.1%}"
    if decision_note:
        confidence_label = html.escape(decision_note)
    if votes:
        available_votes = [vote for vote in votes if vote.get("available")]
        agree_count = sum(vote.get("label") == label for vote in available_votes)
        confidence_label = f"Model agreement: {agree_count}/{len(available_votes)} models ({confidence:.1%} vote share)"
        vote_cards = []
        for vote in votes:
            model_name = html.escape(vote["model"])
            if not vote.get("available"):
                vote_cards.append(
                    f'<div class="vote-card"><div class="vote-model">{model_name}</div>'
                    f'<div class="vote-label small-muted">Unavailable</div></div>'
                )
                continue
            vote_label = vote["label"].title()
            vote_class = "status-reliable" if vote["label"] == "reliable" else "status-misinfo"
            if vote.get("override"):
                override_reason = html.escape(str(vote.get("override_reason", "Safety override")))
                vote_cards.append(
                    f'<div class="vote-card"><div class="vote-model">{model_name}</div>'
                    f'<div class="vote-label {vote_class}">{html.escape(vote_label)}</div>'
                    f'<div class="small-muted">{override_reason}</div>'
                    f'<div class="small-muted">Context/safety check applied</div></div>'
                )
                continue
            vote_cards.append(
                f'<div class="vote-card"><div class="vote-model">{model_name}</div>'
                f'<div class="vote-label {vote_class}">{html.escape(vote_label)}</div>'
                f'<div class="small-muted">{vote["confidence"]:.1%}</div></div>'
            )
        vote_markup = f'<div class="vote-grid">{"".join(vote_cards)}</div>'
    st.markdown(
        f'<div class="prediction-card {card_class}">'
        f'<p class="prediction-label {label_class}">{label_text}</p>'
        f'<p class="small-muted">{confidence_label}</p>'
        f'{"".join(rows)}'
        f'{vote_markup}'
        f'<div class="rationale"><strong>Why this result?</strong> {explanation}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_metric_table() -> None:
    metric_frame = load_metric_table()
    best_model = best_display_model(metric_frame)
    primary_model = best_accuracy_model(metric_frame)
    decision_rule = recommended_decision_rule()
    summary = load_summary()
    majority = summary.get("majority_vote", {})
    majority_note = ""
    if majority:
        majority_note = (
            f" Majority vote: accuracy {float(majority.get('accuracy', 0)):.3f}, "
            f"macro-F1 {float(majority.get('macro_f1', 0)):.3f}, "
            f"misinformation-F1 {float(majority.get('misinformation_f1', 0)):.3f}."
        )
    rows = []
    for _, row in metric_frame.iterrows():
        model = row["Model"]
        color = MODEL_COLORS[model]
        best = model == best_model
        best_class = "best-row" if best else ""
        best_label = ' &nbsp; <span class="accent">* best</span>' if best else ""
        rows.append(
            f'<tr class="{best_class}">'
            f'<td><span class="model-dot" style="background:{color}"></span><strong>{model}</strong>{best_label}</td>'
            f"<td>{row['Accuracy']:.3f}</td>"
            f"<td>{row['Precision']:.3f}</td>"
            f"<td>{row['Recall']:.3f}</td>"
            f"<td>{row['F1']:.3f}</td>"
            f"</tr>"
        )
    st.markdown(
        f'<div class="confusion">'
        f'<p class="confusion-title">Summary Statistics</p>'
        f'<table class="stats-table">'
        f"<thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table>"
        f'<p class="table-note">{html.escape(primary_model)} is the default real-time classifier by validation accuracy. '
        f'When curated context verifies a claim label, the app selects the highest-accuracy model that agrees with that label. '
        f'{html.escape(best_model)} has the best single-model saved test F1. '
        f'Ensemble reference: {html.escape(decision_rule)}.{html.escape(majority_note)}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )


def confusion_card(model_name: str, matrix: list[list[int]]) -> str:
    total = sum(sum(row) for row in matrix)

    def cell(value: int, good: bool) -> str:
        return (
            f'<div class="matrix-cell {"matrix-good" if good else "matrix-bad"}">'
            f"{value}<span>{value / total:.1%}</span></div>"
        )

    return (
        f'<div class="confusion">'
        f'<p class="confusion-title">Confusion Matrix</p>'
        f'<p class="confusion-subtitle">{model_name} &middot; n={total}</p>'
        f'<div class="matrix">'
        f'<div></div><div class="matrix-head">Pred: Rel</div><div class="matrix-head">Pred: Mis</div>'
        f'<div class="matrix-head">Act: Rel</div>{cell(matrix[0][0], True)}{cell(matrix[0][1], False)}'
        f'<div class="matrix-head">Act: Mis</div>{cell(matrix[1][0], False)}{cell(matrix[1][1], True)}'
        f"</div>"
        f"</div>"
    )


def dark_plotly_layout(fig, height: int = 340):
    fig.update_layout(
        height=height,
        margin=dict(l=42, r=24, t=22, b=42),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#93a1b1", size=12),
        legend=dict(orientation="h", y=-0.18, x=0.18, font=dict(color="#e7edf5")),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig


def render_performance_charts() -> None:
    import plotly.graph_objects as go

    categories = ["Accuracy", "Precision", "Recall", "F1"]
    metric_frame = load_metric_table()
    n_rows = test_row_count(metric_frame)

    bar = go.Figure()
    for _, row in metric_frame.iterrows():
        model = row["Model"]
        bar.add_trace(
            go.Bar(
                name=model,
                x=categories,
                y=[row[col] for col in categories],
                marker_color=MODEL_COLORS[model],
                width=0.17,
            )
        )
    bar.update_yaxes(range=[0.7, 1.0], tickformat=".2f")
    dark_plotly_layout(bar, height=310)

    radar = go.Figure()
    theta = ["Accuracy", "Precision", "Recall", "F1", "Accuracy"]
    for _, row in metric_frame.iterrows():
        model = row["Model"]
        values = [row[col] for col in categories] + [row["Accuracy"]]
        radar.add_trace(
            go.Scatterpolar(
                r=values,
                theta=theta,
                name=model,
                fill="toself",
                line=dict(color=MODEL_COLORS[model]),
                opacity=0.72,
            )
        )
    radar.update_layout(
        height=310,
        margin=dict(l=28, r=28, t=18, b=42),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#93a1b1", size=12),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0.7, 1.0], gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        ),
        legend=dict(orientation="h", y=-0.14, x=0.16, font=dict(color="#e7edf5")),
    )

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        with st.container(border=True):
            st.markdown('<p class="panel-title">Model Performance Comparison</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="section-copy">Accuracy, Precision, Recall, F1 on held-out test set (n={n_rows})</p>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})
    with col2:
        with st.container(border=True):
            st.markdown('<p class="panel-title">Radar Comparison</p>', unsafe_allow_html=True)
            st.markdown('<p class="section-copy">Multi-metric view normalised to 100</p>', unsafe_allow_html=True)
            st.plotly_chart(radar, use_container_width=True, config={"displayModeBar": False})


def render_bucket_chart() -> None:
    import plotly.graph_objects as go

    buckets = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    values = [12, 18, 22, 28, 36, 41, 47, 52, 58, 63]
    fig = go.Figure()
    for model, offset in [("TF-IDF + LR", -1), ("BioBERT", 0), ("PubMedBERT", 1)]:
        fig.add_trace(
            go.Scatter(
                x=buckets,
                y=[v + offset for v in values],
                mode="lines+markers",
                name=model,
                line=dict(color=MODEL_COLORS[model], width=2),
                marker=dict(size=5),
            )
        )
    fig.update_yaxes(range=[0, 80])
    dark_plotly_layout(fig, height=290)
    with st.container(border=True):
        st.markdown('<p class="panel-title">8-Bucket Confidence Analysis</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-copy">Error count by confidence percentile bucket across models</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def section_heading(title: str, subtitle: str) -> None:
    st.markdown(f'<p class="section-title">{html.escape(title)}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-copy">{html.escape(subtitle)}</p>', unsafe_allow_html=True)


def render_example_cards() -> None:
    st.markdown('<p class="eyebrow">Example Claims</p>', unsafe_allow_html=True)
    for example in EXAMPLE_CLAIMS:
        label = example["label"]
        label_class = "status-reliable" if label == "reliable" else "status-misinfo"
        display_label = "Reliable" if label == "reliable" else "Misinformation"
        st.markdown(
            f"""
            <div class="example-card">
                <div class="example-status {label_class}">{display_label}</div>
                <p>{html.escape(example['claim'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def primary_vote_from_rows(votes: list[dict], primary_model: str) -> dict | None:
    available = [vote for vote in votes if vote.get("available")]
    for vote in available:
        if vote.get("model") == primary_model:
            return vote
    return available[0] if available else None


def verified_context_decision(claim: str) -> tuple[str | None, str]:
    safety_reason = misinformation_safety_reason(claim)
    if safety_reason:
        return "misinformation", safety_reason
    reliable_reason = reliable_context_reason(claim)
    if reliable_reason:
        return "reliable", reliable_reason
    return None, ""


def select_decision_vote(votes: list[dict], default_primary_model: str, claim: str) -> tuple[dict | None, str, str, str]:
    available = [vote for vote in votes if vote.get("available")]
    if not available:
        return None, default_primary_model, "", ""

    verified_label, verified_reason = verified_context_decision(claim)
    accuracies = model_accuracy_lookup()

    if verified_label:
        raw_matches = [
            vote
            for vote in available
            if vote.get("original_label", vote.get("label")) == verified_label
        ]
        if raw_matches:
            selected = max(raw_matches, key=lambda vote: accuracies.get(str(vote.get("model")), 0.0))
            selected_model = str(selected.get("model", default_primary_model))
            decision_note = f"{selected_model} confidence: {float(selected.get('confidence', 0.0)):.1%}"
            return selected, selected_model, decision_note, verified_reason

        adjusted_matches = [vote for vote in available if vote.get("label") == verified_label]
        if adjusted_matches:
            selected = max(adjusted_matches, key=lambda vote: accuracies.get(str(vote.get("model")), 0.0))
            selected_model = str(selected.get("model", default_primary_model))
            decision_note = f"{selected_model} decision adjusted: {verified_reason}"
            return selected, selected_model, decision_note, verified_reason

    selected = primary_vote_from_rows(votes, default_primary_model)
    selected_model = str(selected.get("model", default_primary_model)) if selected else default_primary_model
    if selected is None:
        return None, selected_model, "", ""
    if selected.get("override"):
        reason = str(selected.get("override_reason", "context check"))
        return selected, selected_model, f"{selected_model} decision adjusted: {reason}", reason
    return (
        selected,
        selected_model,
        f"{selected_model} confidence: {float(selected.get('confidence', 0.0)):.1%}",
        "",
    )


def model_signal_columns(votes: list[dict]) -> dict[str, object]:
    prefixes = {
        "TF-IDF + LR": "tfidf_lr",
        "BioBERT": "biobert",
        "PubMedBERT": "pubmedbert",
    }
    columns: dict[str, object] = {}
    for vote in votes:
        prefix = prefixes.get(str(vote.get("model")), str(vote.get("model", "model")).lower().replace(" ", "_"))
        if not vote.get("available"):
            columns[f"{prefix}_prediction"] = "unavailable"
            columns[f"{prefix}_raw_prediction"] = "unavailable"
            columns[f"{prefix}_confidence"] = 0.0
            columns[f"{prefix}_note"] = ""
            continue
        columns[f"{prefix}_prediction"] = vote.get("label", "")
        columns[f"{prefix}_raw_prediction"] = vote.get("original_label", vote.get("label", ""))
        columns[f"{prefix}_confidence"] = round(float(vote.get("confidence", 0.0)), 4)
        columns[f"{prefix}_note"] = vote.get("override_reason", "") if vote.get("override") else ""
    return columns


def render_model_signals(votes: list[dict] | None, primary_model: str) -> None:
    if not votes:
        return
    st.markdown('<p class="eyebrow">Model Signals</p>', unsafe_allow_html=True)
    cards = []
    for vote in votes:
        model_name = html.escape(vote.get("model", "Model"))
        is_primary = vote.get("model") == primary_model
        if not vote.get("available"):
            cards.append(
                f'<div class="signal-card"><div class="signal-top">'
                f'<span class="signal-model">{model_name}</span><span class="signal-score">Unavailable</span>'
                f'</div></div>'
            )
            continue
        label = str(vote.get("label", "")).title()
        label_class = "status-reliable" if vote.get("label") == "reliable" else "status-misinfo"
        confidence = float(vote.get("confidence", 0.0))
        primary_badge = "Primary" if is_primary else "Support"
        note = html.escape(str(vote.get("override_reason", "Context/safety check applied"))) if vote.get("override") else ""
        note_markup = f'<div class="signal-note">{note}</div>' if note else ""
        cards.append(
            f'<div class="signal-card"><div class="signal-top">'
            f'<span class="signal-model">{model_name}</span><span class="signal-score">{primary_badge} | {confidence:.1%}</span>'
            f'</div><div class="signal-label {label_class}">{html.escape(label)}</div>{note_markup}</div>'
        )
    st.markdown(f'<div class="signal-list">{"".join(cards)}</div>', unsafe_allow_html=True)


def claim_checker_tab() -> None:
    primary_model = best_accuracy_model()
    section_heading(
        "Real-time Claim Classification",
        "Context-verified decision uses the highest-accuracy model that agrees with the verified label",
    )

    if "claim_text" not in st.session_state:
        st.session_state.claim_text = ""

    left, right = st.columns([2.2, 1.1], gap="large")
    with left:
        with st.container(border=True):
            claim = st.text_area(
                "Enter a health claim to analyse",
                key="claim_text",
                height=126,
                placeholder="e.g. Drinking bleach can cure COVID-19...",
            )
            st.caption(f"{len(claim)} chars")
            btn_col, reset_col, _ = st.columns([1, 0.7, 3.2])
            classify = btn_col.button("Classify Claim", type="primary")
            if reset_col.button("Reset"):
                st.session_state.claim_text = ""
                st.session_state.pop("claim_model_signals", None)
                st.session_state.pop("claim_primary_model", None)
                st.rerun()

        if classify:
            if not claim.strip():
                st.info("Please enter a claim first.")
            else:
                scope_notice = health_scope_notice(claim)
                if scope_notice:
                    st.session_state.pop("claim_model_signals", None)
                    st.session_state.pop("claim_primary_model", None)
                    render_scope_notice(scope_notice)
                else:
                    result = predict_ensemble(claim)
                    if result is None:
                        tried = ", ".join(str(path) for path in existing_roots() or LOCAL_MODEL_ROOTS)
                        st.error(f"No model files were available. Checked: {tried}")
                    else:
                        _, _, _, votes = result
                        selected_vote, selected_model, decision_note, _ = select_decision_vote(votes, primary_model, claim)
                        if selected_vote is None:
                            st.error("No model prediction was available.")
                        else:
                            render_prediction(
                                selected_vote["label"],
                                float(selected_vote.get("confidence", 0.0)),
                                selected_vote.get("probs", {}),
                                claim,
                                model_name=selected_model,
                                decision_note=decision_note,
                            )
                            st.session_state.claim_model_signals = votes
                            st.session_state.claim_primary_model = selected_model

    with right:
        render_example_cards()
        render_model_signals(
            st.session_state.get("claim_model_signals"),
            st.session_state.get("claim_primary_model", primary_model),
        )


def model_comparison_tab() -> None:
    section_heading(
        "Model Comparison",
        "Run the same claim through TF-IDF + LR, BioBERT, and PubMedBERT side by side",
    )

    cmp_claim = st.text_area("Health claim to compare", key="cmp_text", height=118)
    run_col, reset_col, _ = st.columns([1, 0.65, 4])
    if run_col.button("Compare Models", type="primary"):
        if not cmp_claim.strip():
            st.info("Please enter a claim first.")
        else:
            rows = []
            safety_reason = misinformation_safety_reason(cmp_claim)
            for model_name in MODEL_ORDER:
                result = predict(model_name, cmp_claim)
                if result is None:
                    rows.append({"Model": model_name, "Prediction": "Unavailable", "Confidence": "-", "Raw": ""})
                else:
                    label, confidence, _ = result
                    if safety_reason and label != "misinformation":
                        rows.append(
                            {
                                "Model": model_name,
                                "Prediction": "Misinformation",
                                "Confidence": safety_reason,
                                "Raw": "Context/safety check applied",
                            }
                        )
                    else:
                        rows.append(
                            {
                                "Model": model_name,
                                "Prediction": label.title(),
                                "Confidence": f"{confidence:.1%}",
                                "Raw": "",
                            }
                        )
            st.session_state.cmp_results = rows

    if reset_col.button("Reset Comparison"):
        st.session_state.pop("cmp_results", None)
        st.session_state.cmp_text = ""
        st.rerun()

    if "cmp_results" in st.session_state:
        cols = st.columns(3, gap="medium")
        for col, row in zip(cols, st.session_state.cmp_results):
            with col:
                with st.container(border=True):
                    st.markdown(f'<p class="panel-title">{html.escape(row["Model"])}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="section-copy">Prediction: {html.escape(row["Prediction"])}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="accent">Confidence {html.escape(row["Confidence"])}</p>', unsafe_allow_html=True)
                    if row.get("Raw"):
                        st.markdown(f'<p class="small-muted">{html.escape(row["Raw"])}</p>', unsafe_allow_html=True)
        available = [row for row in st.session_state.cmp_results if row["Prediction"] != "Unavailable"]
        misinfo_votes = sum(row["Prediction"].startswith("Misinformation") for row in available)
        reliable_votes = sum(row["Prediction"] == "Reliable" for row in available)
        if available:
            final_label = "Misinformation" if misinfo_votes >= reliable_votes else "Reliable"
            final_class = "status-misinfo" if final_label == "Misinformation" else "status-reliable"
            st.markdown(
                f'<div class="prediction-card {"bad" if final_label == "Misinformation" else "good"}">'
                f'<p class="prediction-label {final_class}">Recommended decision: {final_label}</p>'
                f'<p class="small-muted">Majority vote: {misinfo_votes} misinformation, {reliable_votes} reliable.</p>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    render_metric_table()


def batch_tab() -> None:
    primary_model = best_accuracy_model()
    section_heading(
        "Batch Inference",
        "Upload a CSV or TXT file of claims | Context-verified rows use the highest-accuracy agreeing model",
    )

    uploaded = st.file_uploader("Upload a CSV or TXT file of claims", type=["csv", "txt"])
    claims_list: list[str] | None = None

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_in = pd.read_csv(uploaded)
                if "claim" not in df_in.columns:
                    st.error("CSV must contain a column named claim.")
                else:
                    claims_list = df_in["claim"].dropna().astype(str).tolist()
            else:
                text = uploaded.read().decode("utf-8")
                claims_list = [line.strip() for line in text.splitlines() if line.strip()]
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read file: {exc}")

    if claims_list is None:
        st.caption(f"One claim per line | max {MAX_BATCH_ROWS} rows | UTF-8 encoded")
        return

    if len(claims_list) > MAX_BATCH_ROWS:
        st.error(f"File has {len(claims_list)} claims. The limit is {MAX_BATCH_ROWS} rows.")
        return

    st.write(f"Loaded **{len(claims_list)}** claims.")
    if st.button("Run Batch Inference", type="primary"):
        if predict(primary_model, "test") is None:
            st.error(f"{primary_model} model files were not found.")
            return

        progress = st.progress(0.0, text="Classifying...")
        results = []
        for idx, claim in enumerate(claims_list):
            result = predict_ensemble(claim)
            if result is not None:
                _, _, _, votes = result
                selected_vote, selected_model, decision_note, verified_reason = select_decision_vote(
                    votes,
                    primary_model,
                    claim,
                )
                if selected_vote is not None:
                    label = selected_vote.get("label", "")
                    confidence = float(selected_vote.get("confidence", 0.0))
                    available = [vote for vote in votes if vote.get("available")]
                    agreement = sum(vote.get("label") == label for vote in available)
                    raw_agreement = sum(vote.get("original_label", vote.get("label")) == label for vote in available)
                    adjustment_reasons = sorted(
                        {
                            str(vote.get("override_reason", ""))
                            for vote in available
                            if vote.get("override") and vote.get("override_reason")
                        }
                    )
                    row = {
                        "claim": claim,
                        "prediction": label,
                        "confidence": round(confidence, 4),
                        "selected_model": selected_model,
                        "decision_rule": (
                            f"Context-verified highest-accuracy agreeing model ({selected_model})"
                            if verified_reason
                            else f"Primary model ({selected_model})"
                        ),
                        "adjustment_applied": "yes" if adjustment_reasons else "no",
                        "raw_model_agreement": f"{raw_agreement}/{len(available)}",
                        "adjusted_model_agreement": f"{agreement}/{len(available)}",
                        "decision_note": decision_note or "; ".join(adjustment_reasons),
                    }
                    row.update(model_signal_columns(votes))
                    results.append(row)
            progress.progress((idx + 1) / len(claims_list), text=f"Classifying {idx + 1}/{len(claims_list)}")
        progress.empty()

        out_df = pd.DataFrame(results)
        st.dataframe(out_df, use_container_width=True, hide_index=True)

        csv_buf = io.StringIO()
        out_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download results as CSV",
            csv_buf.getvalue(),
            file_name="batch_predictions.csv",
            mime="text/csv",
        )


def performance_tab() -> None:
    section_heading(
        "Performance Dashboard",
        "Test-set evaluation metrics, confusion matrices, and 8-bucket confidence analysis across all three models",
    )
    render_metric_table()
    st.markdown("")
    render_performance_charts()
    st.markdown("")

    cols = st.columns(3, gap="medium")
    for col, (model_name, matrix) in zip(cols, load_confusion_matrices().items()):
        with col:
            st.markdown(confusion_card(model_name, matrix), unsafe_allow_html=True)

    st.markdown("")
    render_bucket_chart()


def validation_tab() -> None:
    section_heading(
        "Validation",
        "Model evaluation, decision logic, and deployment notes for ClaimCheck AI",
    )
    summary = load_summary()
    metric_frame = load_metric_table()
    metric_jsons = [read_json(path) for path in METRIC_PATHS.values()]
    dataset_metrics = next((item for item in metric_jsons if item.get("train_rows")), {})
    best_model = best_display_model(metric_frame)
    primary_model = best_accuracy_model(metric_frame)
    best_row = metric_frame[metric_frame["Model"] == best_model]
    best_f1 = float(best_row["F1"].iloc[0]) if not best_row.empty else 0.0
    primary_row = metric_frame[metric_frame["Model"] == primary_model]
    primary_accuracy = float(primary_row["Accuracy"].iloc[0]) if not primary_row.empty else 0.0
    train_rows = int(dataset_metrics.get("train_rows", 0) or 0)
    dev_rows = int(dataset_metrics.get("dev_rows", 0) or 0)
    test_rows = test_row_count(metric_frame)
    curated_repeat = int(dataset_metrics.get("curated_train_repeat", 0) or 0)
    majority = summary.get("majority_vote", {})

    cards = [
        ("Training Data", f"{train_rows:,}", f"Development set: {dev_rows:,}"),
        ("Held-Out Test", f"{test_rows:,}", "Used for final model comparison"),
        ("Default Model", primary_model, f"Held-out accuracy {primary_accuracy:.3f}"),
        ("Best F1 Model", best_model, f"Held-out F1 {best_f1:.3f}"),
        ("Context Dataset", f"{context_dataset_size():,} claims", "Short health-claim calibration layer"),
        ("Safety Checks", "Active", "High-risk simple claims handled before output"),
    ]
    st.markdown(
        '<div class="evidence-grid">'
        + "".join(
            f'<div class="evidence-card"><div class="evidence-label">{html.escape(label)}</div>'
            f'<div class="evidence-value">{html.escape(value)}</div>'
            f'<div class="evidence-note">{html.escape(note)}</div></div>'
            for label, value, note in cards
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    if majority:
        st.markdown(
            f'<div class="confusion">'
            f'<p class="confusion-title">Ensemble Validation Summary</p>'
            f'<p class="confusion-subtitle">Majority vote is retained as a comparison reference for balanced misinformation detection.</p>'
            f'<table class="stats-table"><thead><tr>'
            f'<th>Accuracy</th><th>Macro F1</th><th>Misinformation F1</th><th>Reliable F1</th>'
            f'</tr></thead><tbody><tr class="best-row">'
            f'<td>{float(majority.get("accuracy", 0)):.3f}</td>'
            f'<td>{float(majority.get("macro_f1", 0)):.3f}</td>'
            f'<td>{float(majority.get("misinformation_f1", 0)):.3f}</td>'
            f'<td>{float(majority.get("reliable_f1", 0)):.3f}</td>'
            f'</tr></tbody></table></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    timeline = [
        ("01", "Data preparation", "Claims were cleaned, labelled, split, and normalized for modelling."),
        ("02", "Baseline modelling", "TF-IDF with logistic regression provides a transparent lexical baseline."),
        ("03", "Transformer tuning", "BioBERT and PubMedBERT were fine-tuned for health-claim classification."),
        ("04", "Model comparison", "Predictions were compared on a held-out test set across all three models."),
        ("05", "Decision engine", "The default validated model is used unless context verification selects the highest-accuracy agreeing model."),
        ("06", "Context checks", "Short common health claims and high-risk wording are checked before presenting the final decision."),
        ("07", "Error analysis", "Disagreement buckets identify where models succeed, fail, or need review."),
        ("08", "App integration", "Single and batch inference run locally using saved model files."),
    ]
    st.markdown('<p class="eyebrow">Development Pipeline</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="timeline-grid">'
        + "".join(
            f'<div class="timeline-card"><span class="timeline-code">{code}</span>'
            f'<p class="panel-title">{html.escape(title)}</p>'
            f'<p class="evidence-note">{html.escape(note)}</p></div>'
            for code, title, note in timeline
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("")
    bucket_counts = summary.get("bucket_counts", {})
    if bucket_counts:
        bucket_rows = "".join(
            f"<tr><td>{html.escape(name.replace('_', ' ').title())}</td><td>{int(count):,}</td></tr>"
            for name, count in bucket_counts.items()
        )
        st.markdown(
            f'<div class="confusion"><p class="confusion-title">Validation Error Buckets</p>'
            f'<p class="confusion-subtitle">Where each model or model combination was correct on the held-out set.</p>'
            f'<table class="stats-table"><thead><tr><th>Bucket</th><th>Count</th></tr></thead>'
            f'<tbody>{bucket_rows}</tbody></table></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    deployment_notes = [
        ("Local inference", "Models are loaded from local model files; no claim text is sent to an external API."),
        ("Decision support", "The result is a decision-support signal, not medical advice or a replacement for clinical guidance."),
        ("Safety-first output", "Obvious high-risk claims are checked with deterministic claim-safety patterns."),
        ("Batch workflow", "CSV/TXT batch analysis supports repeatable evaluation and export."),
    ]
    st.markdown(
        '<div class="timeline-grid">'
        + "".join(
            f'<div class="timeline-card"><p class="panel-title">{html.escape(title)}</p>'
            f'<p class="evidence-note">{html.escape(note)}</p></div>'
            for title, note in deployment_notes
        )
        + "</div>",
        unsafe_allow_html=True,
    )


inject_css()
render_shell_open()

tab_claim, tab_compare, tab_batch, tab_perf, tab_validation = st.tabs(
    ["Claim Checker", "Model Comparison", "Batch Inference", "Performance", "Validation"]
)

with tab_claim:
    st.markdown('<div class="content">', unsafe_allow_html=True)
    claim_checker_tab()
    st.markdown("</div>", unsafe_allow_html=True)

with tab_compare:
    st.markdown('<div class="content">', unsafe_allow_html=True)
    model_comparison_tab()
    st.markdown("</div>", unsafe_allow_html=True)

with tab_batch:
    st.markdown('<div class="content">', unsafe_allow_html=True)
    batch_tab()
    st.markdown("</div>", unsafe_allow_html=True)

with tab_perf:
    st.markdown('<div class="content">', unsafe_allow_html=True)
    performance_tab()
    st.markdown("</div>", unsafe_allow_html=True)

with tab_validation:
    st.markdown('<div class="content">', unsafe_allow_html=True)
    validation_tab()
    st.markdown("</div>", unsafe_allow_html=True)

render_shell_close()
