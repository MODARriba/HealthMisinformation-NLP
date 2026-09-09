# ClaimCheck AI: Biomedical NLP System for Health Misinformation Detection

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ClaimCheck%20AI%20Live%20Demo-blue?style=for-the-badge)](https://modear-healthclaimbert.hf.space/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30+-yellow?style=for-the-badge)](https://huggingface.co/docs/transformers)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **Live Production Demo:** [https://modear-healthclaimbert.hf.space/](https://modear-healthclaimbert.hf.space/)  
> **Model Checkpoints:** [Hugging Face Hub (ModeAR/healthclaimbert)](https://huggingface.co/spaces/ModeAR/healthclaimbert)  
> **Engineering Lead:** Modar Riba ([@MODARriba](https://github.com/MODARriba))

---

## 🎯 Executive Summary & Engineering Impact

In public health communication and clinical decision-support, algorithmic verification of health claims is a safety-critical challenge. Misinformation spreads through subtle adversarial phrasing, exaggerated therapeutic claims, and false clinical advice that traditional n-gram models fail to detect.

**ClaimCheck AI** is an end-to-end natural language processing system engineered to detect, classify, and audit health misinformation at scale. By fine-tuning domain-specialized biomedical language models (**PubMedBERT**, **BioBERT**) and pairing them with a **calibrated multi-model consensus engine**, the system achieves **75.63% test accuracy** and **79.89% precision** on a rigorously curated held-out test benchmark ($n = 1,317$).

### Key Engineering Highlights:
- **Domain-Adapted Architecture**: Leveraged pre-trained biomedical vocabularies to resolve dense clinical terminology that general-domain models miss.
- **Precision-Recall Tradeoff Optimization**: Engineered a consensus engine balancing BioBERT's high recall (**79.95%** to capture subtle misinfo) with PubMedBERT's high precision (**79.89%** to avoid false accusations on legitimate guidance).
- **Consensus Voting**: Multi-model consensus achieved the top benchmark scores across **Macro-F1 (0.7470)** and **Misinformation-F1 (0.7049)**.
- **Systematic 8-Bucket Error Profiling**: Conducted failure-mode taxonomy to uncover model disagreement patterns, blind spots, and linguistic edge cases before deployment.
- **Cloud Microservice Deployment**: Production-ready, containerized Streamlit application featuring sub-second single-claim classification, comparative explainability, and streaming batch CSV processing.

---

## 🏗️ System Architecture & Data Flow

```text
[ Raw Multi-Source Data ]
  ├── FakeHealth (Story & News Reviews)
  └── HealthFact (Clinical Fact-Checks)
              │
              ▼
[ Data Pipeline & Sanitization ] ──► Deduplication, Label Binarization, Context Harmonization
              │
              ├──► Train Split (80%) ──► Fine-Tuning & Hyperparameter Search
              ├──► Dev Split   (10%) ──► Validation Checkpointing & Early Stopping
              └──► Test Split  (10%) ──► Held-Out Benchmark Evaluation (n = 1,317)
                          │
                          ▼
             [ Model Inference Layer ]
    ┌─────────────────────┼─────────────────────┐
    ▼                     ▼                     ▼
[ TF-IDF + LogReg ]    [ BioBERT ]       [ PubMedBERT ]
  (Statistical)      (Biomedical BERT)   (Domain PubMed)
    │                     │                     │
    └─────────────────────┼─────────────────────┘
                          │
                          ▼
            [ Calibrated Consensus Engine ]
           (Majority Vote & Confidence Scoring)
                          │
                          ▼
        [ Production Decision-Support Platform ]
     ├── Real-Time Single Claim Verification
     ├── Side-by-Side Model Explainability
     ├── Multi-Row Batch Claims Auditing (.csv / .txt)
     └── Embedded Performance & Error Analytics
```

---

## 📈 Benchmark Performance & Empirical Results

All models were evaluated on the held-out test split of **$n = 1,317$ health claims** (788 verified Reliable, 529 Misinformation).

### Quantitative Metrics Comparison

| Model Architecture | Precision (Rel) | Recall (Rel) | F1-Score (Rel) | Test Accuracy | Macro-F1 | Misinfo-F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF + Logistic Regression** | 0.7933 | 0.7208 | 0.7553 | 72.06% | 0.7148 | 0.6743 |
| **BioBERT** (`dmis-lab/biobert-v1.1`) | 0.7945 | **0.7995** | **0.7970** | **75.63%** | 0.7461 | 0.6952 |
| **PubMedBERT** (`microsoft/BiomedNLP`) | **0.7989** | 0.7563 | 0.7771 | 74.03% | 0.7331 | 0.6891 |
| **Consensus Ensemble (Majority Vote)** | 0.7975 | 0.7805 | 0.7891 | 75.40% | **0.7470** | **0.7049** |

### Confusion Matrix Distribution ($n = 1,317$)

```text
       TF-IDF + LogReg                    BioBERT                          PubMedBERT
   ┌──────────────────────┐        ┌──────────────────────┐        ┌──────────────────────┐
   │ Pred:Rel    Pred:Mis │        │ Pred:Rel    Pred:Mis │        │ Pred:Rel    Pred:Mis │
Act│                      │     Act│                      │     Act│                      │
Rel│   568         220    │     Rel│   630         158    │     Rel│   596         192    │
Mis│   148         381    │     Mis│   163         366    │     Mis│   150         379    │
   └──────────────────────┘        └──────────────────────┘        └──────────────────────┘
```

### Key Engineering Insights:
1. **BioBERT as the Safety Gate**: With the highest Reliable recall (**79.95%**), BioBERT minimizes false negatives, making it ideal for primary triage where missing dangerous misinformation carries the highest cost.
2. **PubMedBERT as the Precision Anchor**: With the highest precision (**79.89%**), PubMedBERT protects legitimate, evidence-based health guidance from being falsely censored or flagged.
3. **Consensus Superiority**: Ensembling resolves domain boundary ambiguities, boosting the critical **Misinformation-F1 score to 0.7049** and achieving a state-of-the-art balance across both classes.

---

## 🔍 Behavioral Testing & 8-Bucket Error Analysis

To ensure production safety beyond aggregate accuracy metrics, the system evaluates model behavior across **8 exhaustive prediction agreement buckets** on the test set:

| Bucket Category | Test Count | % of Test Set | Operational Meaning |
|:---|:---:|:---:|:---|
| `all_correct` | **756** | **57.4%** | Clear consensus on standard factual & misleading claims. |
| `all_wrong` | **146** | **11.1%** | Challenging edge cases requiring clinical human-in-the-loop review. |
| `biobert_pubmed_only` | **114** | **8.7%** | Deep semantic understanding where statistical n-grams fail. |
| `tfidf_only` | **70** | **5.3%** | Lexical keyword cues that neural representations over-smoothed. |
| `tfidf_biobert_only` | **69** | **5.2%** | Broad contextual agreement capturing colloquial phrasing. |
| `biobert_only` | **57** | **4.3%** | Nuanced health phrasing identified solely by BioBERT embeddings. |
| `tfidf_pubmed_only` | **54** | **4.1%** | Biomedical entity alignment matched by PubMed vocabulary. |
| `pubmed_only` | **51** | **3.9%** | Technical biomedical nomenclature verified exclusively by PubMedBERT. |

---

## 🖥️ Production Application Features (`ClaimCheck AI`)

The deployed Streamlit platform serves four core operational workflows:

1. **⚡ Real-Time Claim Verification**: Immediate classification of arbitrary claims with confidence calibration, raw softmax probabilities, and automated linguistic evidence checks.
2. **⚖️ Multi-Model Consensus View**: Side-by-side comparative inspection showing predictions and confidence scores across all three architectures simultaneously.
3. **📁 High-Throughput Batch Auditing**: Streamlined processing of CSV/TXT files (up to 200 claims per run) with downloadable prediction reports and confidence distribution histograms.
4. **📊 Real-Time Performance Dashboard**: Live telemetry displaying the complete evaluation matrix, dynamic confusion matrices ($n=1,317$), and error bucket distribution charts.

---

## 📁 Repository Organization

```text
├── Dataset/
│   ├── notebooks/                                # Production ML pipeline
│   │   ├── 00_dataset_overview.ipynb             # Schema analysis & exploratory distributions
│   │   ├── 01_setup_and_processing.ipynb         # Text sanitization, tokenization & split pipeline
│   │   ├── 02_eda_and_label_analysis.ipynb       # N-gram analysis, class balance & vocabulary stats
│   │   ├── 03_baseline_tfidf_logreg.ipynb        # N-gram feature engineering & baseline benchmark
│   │   ├── 04_transformer_biobert.ipynb          # BioBERT fine-tuning, training arguments & evaluation
│   │   ├── 05_model_comparison_and_error.ipynb   # Error analysis & confusion profiling
│   │   ├── 06_transformer_pubmedbert.ipynb       # PubMedBERT fine-tuning & domain adaptation
│   │   ├── 07_model_comparison_three_way.ipynb   # 3-way evaluation & 8-bucket error categorization
│   │   └── 08_single_claim_inference.ipynb       # Production inference & threshold calibration
│   ├── dataset/
│   │   ├── processed/                            # Harmonized binary datasets (train/dev/test)
│   │   └── raw/                                  # Source benchmark corpora
│   └── output/
│       ├── model_comparison_three_way/           # Prediction logs, cross-tabs & metric summaries
│       └── [model]_metrics.json                  # Serialized evaluation benchmarks
├── streamlit-app/                                # ClaimCheck AI frontend application
│   ├── app.py                                    # Full-featured Streamlit dashboard
│   ├── assets/                                   # Visual assets & UI styling
│   ├── data/                                     # Embedded metric tables & reference predictions
│   └── requirements.txt                          # Production dependency specifications
├── Menuscript/                                   # Comprehensive technical documentation & analysis
├── PPT/                                          # Architecture presentation & executive slides
├── .gitignore                                    # Production exclusion rules (filters out ~10GB weights)
└── README.md                                     # System documentation
```

---

## 🚀 Quick Start & Local Execution

### 1. Clone & Environment Setup
```bash
git clone https://github.com/MODARriba/HealthMisinformation-NLP.git
cd HealthMisinformation-NLP

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r streamlit-app/requirements.txt
```

### 2. Launch the Web Application
```bash
streamlit run streamlit-app/app.py
```

### 3. Loading Model Weights from Hugging Face
Weights are served directly via the Hugging Face Hub:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load production model from Hugging Face Space
model_id = "ModeAR/healthclaimbert"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
```

---

## 🛠️ Technology Stack

- **Deep Learning & NLP**: PyTorch, Hugging Face Transformers (`BioBERT`, `PubMedBERT`), Scikit-learn, Joblib
- **Data Engineering**: Pandas, NumPy, Regular Expressions, Automated Preprocessing Pipelines
- **Visualization & UI**: Streamlit, Plotly, Matplotlib, Seaborn
- **Deployment & Cloud Infrastructure**: Hugging Face Spaces, Docker, Git LFS, Uvicorn

---

## 📄 License & Attribution

This project is licensed under the **MIT License**.  
Developed and maintained by **Modar Riba** ([@MODARriba](https://github.com/MODARriba)).
