# ClaimCheck AI: Biomedical NLP System for Health Misinformation Detection

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ClaimCheck%20AI%20Live%20Demo-blue?style=for-the-badge)](https://modear-healthclaimbert.hf.space/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30+-yellow?style=for-the-badge)](https://huggingface.co/docs/transformers)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

> **Live Production Application:** [https://modear-healthclaimbert.hf.space/](https://modear-healthclaimbert.hf.space/)  
> **Model Repository & Hugging Face Space:** [Hugging Face (ModeAR/healthclaimbert)](https://huggingface.co/spaces/ModeAR/healthclaimbert)  
> **Engineering Lead:** Modar Riba ([@MODARriba](https://github.com/MODARriba))

---

## 🎯 Executive Summary & Engineering Impact

In public health communication and clinical decision-support, automated verification of medical claims is a safety-critical challenge. Misinformation spreads through subtle adversarial phrasing, exaggerated therapeutic claims, and false clinical advice that traditional n-gram models fail to detect.

**ClaimCheck AI** is an end-to-end natural language processing system engineered to detect, classify, and audit health misinformation at scale across text, web articles, screenshots, and video transcripts. By fine-tuning domain-specialized biomedical language models (**PubMedBERT**, **BioBERT**) and pairing them with a **calibrated multi-model consensus engine**, the system achieves **75.63% test accuracy** and **79.89% precision** on a rigorously curated held-out test benchmark ($n = 1,317$).

### Key Engineering Highlights:
- **Domain-Adapted Architecture**: Leveraged pre-trained biomedical vocabularies to resolve dense clinical terminology that general-domain models miss.
- **Precision-Recall Tradeoff Optimization**: Engineered a consensus engine balancing BioBERT's high recall (**79.95%** to capture subtle misinfo) with PubMedBERT's high precision (**79.89%** to avoid false accusations on legitimate guidance).
- **Consensus Voting**: Multi-model consensus achieved top benchmark scores across **Macro-F1 (0.7470)** and **Misinformation-F1 (0.7049)**.
- **Explainable AI (XAI)**: Full article visual heatmaps with sentence-level classification, word-level trigger keyword attribution, and printable PDF fact-check reports.
- **Live Scientific Grounding (RAG)**: Real-time queries against the **NCBI PubMed API** and Google Fact Check API, paired with a Domain Credibility Trust Index (0–100).
- **Multimodal Media Ingestion**: Dual-engine image OCR (In-Browser Tesseract.js + PyTesseract) and YouTube Video/Podcast transcript auditing with clickable timestamp pills (`▶ MM:SS`).
- **Model Robustness & Nuance**: Calibrated "Inconclusive / Emerging Research" class (45%–55% uncertainty band), multilingual translation pre-processing (Indic and global languages), and an adversarial perturbation test bench.

---

## 🚀 Live Interactive Demo

The system is deployed as a cloud microservice on Hugging Face Spaces and is **publicly accessible 24/7**. Anyone can test and audit claims without creating an account or running code locally:

<div align="center">

[![Launch Live Demo](https://img.shields.io/badge/▶%20Launch%20Live%20App-modear--healthclaimbert.hf.space-2ea44f?style=for-the-badge&logo=rocket&logoColor=white)](https://modear-healthclaimbert.hf.space/)
&nbsp;&nbsp;
[![View Space on Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace%20Space-ModeAR%2Fhealthclaimbert-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/ModeAR/healthclaimbert)

</div>

### 🧪 Try It Yourself: Benchmark Sample Claims
You can copy-paste and verify any of these claims directly in the web application:

| Claim Category | Sample Claim to Test | Expected Verdict | Primary Indicator |
|:---|:---|:---:|:---|
| **Pseudoscience / Cancer Cure** | *"Drinking lemon water every morning alkalizes your body and completely cures cancer."* | 🔴 **Misinformation** | Flags absolute cure claims lacking clinical trial evidence. |
| **Public Health Consensus** | *"Regular aerobic physical activity and a balanced diet reduce the long-term risk of cardiovascular disease."* | 🟢 **Reliable** | Matches CDC/WHO clinical guidance and preventive evidence. |
| **Vaccine Skepticism** | *"COVID-19 vaccines alter human genomic DNA and contain microchips."* | 🔴 **Misinformation** | Strong consensus detection across both PubMedBERT & BioBERT. |
| **Nutritional Guidance** | *"Vitamin D3 supplementation may help support immune function during respiratory viral infections."* | 🟢 **Reliable** | Recognized as cautious, evidence-supported medical wording. |
| **Emerging Research** | *"Intermittent fasting has shown promising metabolic markers in mouse models."* | 🟡 **Inconclusive** | Falls within the 45%–55% uncertainty band; flagged as emerging science. |

---

## 🌟 Production Features & Capabilities

```
ClaimCheck AI Platform
├── 1. Single Claim Checker & XAI
│   ├── Multi-Model Consensus (PubMedBERT + BioBERT + RoBERTa)
│   ├── Token Trigger Highlighting (Misinformation Keyword Attribution)
│   ├── Calibrated Uncertainty Band (Inconclusive / Emerging Research Class)
│   ├── Multilingual Auto-Translation (Indic & Global Languages → English)
│   ├── Live NCBI PubMed Scientific Literature Retrieval (RAG)
│   └── 1-Click Printable PDF Fact-Check Report Export
├── 2. Webpage & Article Auditor
│   ├── Live URL Scraper & Boilerplate Cleaner
│   ├── Sentence-by-Sentence Breakdown (< 30 Key Sentences)
│   ├── Inline Color-Coded Article Heatmap (Green / Red / Yellow)
│   ├── Source Domain Credibility (0–100 Trust Index: CDC/WHO vs Unverified Blogs)
│   └── Google Fact Check Tools API Integration (Snopes, PolitiFact, SciCheck)
├── 3. Multimodal Media Scanner
│   ├── Screenshot & Infographic OCR (Dual-Engine: Tesseract.js Client-Side + PyTesseract)
│   └── YouTube Video & Podcast Transcript Auditing
│       ├── Dual Mode: Auto-Fetch Link + Paste Transcript / Subtitles
│       ├── Temporal Dialogue Alignment & Clickable Timestamp Pills (▶ MM:SS)
│       ├── Sub-Second Vectorized Batch Auditing (predict_ensemble_batch)
│       └── 1-Click "Auto-Audit Spoken Claims (AI Fallback)" & "COVID News Sample"
├── 4. Adversarial & Robustness Lab
│   ├── Semantic Negation Stress Testing ("does not cure")
│   ├── Clinical Synonym Substitution ("malignant neoplasms")
│   ├── Typographical & Noise Perturbation ("cancr")
│   └── Real-Time Robustness Score (0–100%) & Confidence Delta Tracking
└── 5. Benchmarks & Telemetry
    ├── Side-by-Side Model Divergence Matrix
    ├── Streaming CSV Batch Ingestion (Up to 200 Claims)
    └── 8-Bucket Error Profiling Dashboard (n = 1,317)
```

---

## 🏗️ System Architecture & Data Flow

```text
[ Input Channels: Text / URL / Image Screenshot / Video Transcript ]
                                │
                                ▼
[ Ingestion & Sanitization Layer ]
  ├── Web Scraper & Readability Parser (Clean article paragraphs)
  ├── Dual OCR Engine (In-Browser Tesseract.js / PyTesseract)
  ├── YouTube Transcript Normalizer & Temporal Segmenter
  └── Multilingual Language Detector & Translation Bridge
                                │
                                ▼
[ Multi-Model Clinical Inference Layer ]
  ┌─────────────────────────────┼─────────────────────────────┐
  ▼                             ▼                             ▼
[ TF-IDF + Calibrated Linear ]  [ BioBERT v1.1 ]          [ PubMedBERT ]
  (Statistical Baseline)      (Biomedical BERT)          (Domain PubMed)
  └─────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
[ Calibrated Consensus & Uncertainty Engine ]
  ├── Hard Binary Vote vs. "Inconclusive / Emerging Research" Band (45%–55%)
  ├── Safety Priority Filter (Rule-based clinical contraindication triage)
  └── Word-Level Attribution (Token feature weight extraction)
                                │
                                ▼
[ Live Grounding & External Validation (RAG) ]
  ├── NCBI PubMed E-utilities API (Peer-reviewed study titles & DOI links)
  ├── Source Domain Trust Index (WHO, CDC, NIH, Mayo Clinic vs Fringe blogs)
  └── Google Fact Check ClaimSearch API (Snopes, PolitiFact, SciCheck)
                                │
                                ▼
[ Decision-Support Output & Reporting ]
  ├── Color-Coded Article Heatmap with Hover Tooltips
  ├── Interactive Timestamp Navigation Pills (▶ MM:SS)
  └── Printable Medical PDF Fact-Check Report (PdfPages Export)
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

## ⚙️ Operational Considerations & Cloud Architecture Realities

When deploying machine learning systems on public cloud containers (such as Hugging Face Spaces free tier), real-world infrastructure constraints surface:

1. **YouTube Automated Scraping & Bot Protection**:
   - **Challenge:** YouTube blocks automated scraping requests originating from cloud datacenter IP ranges (AWS, GCP, Hugging Face), resulting in connection drops (`SSLEOFError`). In addition, video uploaders frequently disable subtitles entirely on YouTube.
   - **Solution in ClaimCheck AI:** The application implements a dual-mode workflow. Users can paste native YouTube transcripts directly (`0:00 text`, `[01:15] text`), load pre-configured broadcast samples, or click **"⚡ Auto-Audit Spoken Claims (AI Fallback)"** to immediately audit spoken statements without being blocked.
2. **Server-Side OCR vs. Client-Side OCR**:
   - **Challenge:** Server-side `pytesseract` requires system-level Linux C++ binaries (`tesseract-ocr`) installed in the container environment.
   - **Solution in ClaimCheck AI:** Added an **In-Browser OCR widget powered by Tesseract.js**, processing images 100% locally on the user's browser canvas with zero server overhead and complete user privacy.
3. **CPU Inference Latency & Batch Optimization**:
   - **Challenge:** Evaluating long articles (50+ sentences) sentence-by-sentence sequentially across 3+ deep neural networks on CPU can take over 45 seconds.
   - **Solution in ClaimCheck AI:** Vectorized batch inference (`predict_ensemble_batch`) with sentence deduplication and an upper-bound cap of 25–30 priority sentences, dropping audit latency from ~45 seconds to **under 2 seconds**.
4. **Third-Party API Rate Limits (PubMed / Google Fact Check)**:
   - External retrieval uses NCBI E-utilities and Google APIs. Requests are designed to fail gracefully with non-blocking UI fallbacks if external public APIs throttle or experience downtime.

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
│   ├── app.py                                    # Full-featured Streamlit dashboard (~6,100 lines)
│   ├── assets/                                   # Visual assets & UI styling
│   ├── data/                                     # Embedded metric tables & reference predictions
│   └── requirements.txt                          # Production dependency specifications
├── Menuscript/                                   # Comprehensive technical documentation & analysis
├── PPT/                                          # Architecture presentation & executive slides
├── .gitignore                                    # Production exclusion rules
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

### 3. Loading Model Checkpoints from Hugging Face
Weights are served directly via the Hugging Face Hub:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "ModeAR/healthclaimbert"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
```

---

## 🛠️ Technology Stack

- **Deep Learning & NLP**: PyTorch, Hugging Face Transformers (`BioBERT`, `PubMedBERT`, `RoBERTa`, `DeBERTa`), Scikit-learn
- **Data Engineering**: Pandas, NumPy, Regular Expressions, Sentence Tokenization Pipelines
- **External Clinical Evidence**: NCBI PubMed E-utilities API, Google Fact Check Tools API
- **Computer Vision & Media**: Tesseract.js (Client-side), PyTesseract, YouTube Transcript API
- **Visualization & UI**: Streamlit, Plotly, Matplotlib, Seaborn
- **Deployment & Cloud Infrastructure**: Hugging Face Spaces, Git LFS, Docker

---

## 👤 Author & Contact

Developed and maintained by **Modar Riba** ([@MODARriba](https://github.com/MODARriba)).  
📧 **Email:** ribamodar@gmail.com
