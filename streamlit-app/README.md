# ClaimCheck AI - Decision-Support Frontend

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ClaimCheck%20AI-blue?style=flat-square)](https://modear-healthclaimbert.hf.space/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?style=flat-square)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat-square)](https://www.python.org/)

Production Streamlit web interface for **ClaimCheck AI**, an automated clinical NLP platform for health misinformation verification and evidence auditing.

---

## 🛠️ Architecture & Core Workflows

1. **⚡ Claim Checker (Single-Claim Inference)**:
   - Evaluates input claims against fine-tuned biomedical transformer representations (`PubMedBERT`, `BioBERT`).
   - Generates confidence ratings, class probability distributions, and domain safety evidence checks.

2. **⚖️ Multi-Model Consensus Engine**:
   - Executes parallel comparative inference across TF-IDF, BioBERT, and PubMedBERT.
   - Computes real-time majority consensus votes to mitigate single-model classification bias.

3. **📁 High-Throughput Batch Inference**:
   - Accepts CSV or TXT batch claim files (up to 200 rows per batch).
   - Generates streaming progress, downloadable prediction reports, and confidence distributions.

4. **📊 Performance & Error Profiling Dashboard**:
   - Displays real-time test-set metrics ($n = 1,317$).
   - Dynamically visualizes 2x2 confusion matrices and 8-bucket error categorization breakdowns.

---

## 📈 Test-Set Benchmark Results ($n = 1,317$)

| Model | Architecture | Accuracy | Precision (Rel) | Recall (Rel) | F1-Score (Rel) | Macro-F1 | Misinfo-F1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF + LogReg** | N-gram Baseline | 72.06% | 0.7933 | 0.7208 | 0.7553 | 0.7148 | 0.6743 |
| **BioBERT** | Biomedical Transformer | **75.63%** | 0.7945 | **0.7995** | **0.7970** | 0.7461 | 0.6952 |
| **PubMedBERT** | Domain-Specific (PubMed) | 74.03% | **0.7989** | 0.7563 | 0.7771 | 0.7331 | 0.6891 |
| **Majority Ensemble**| Consensus Engine | 75.40% | 0.7975 | 0.7805 | 0.7891 | **0.7470** | **0.7049** |

---

## 🚀 Local Deployment

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run application
streamlit run app.py
```

*Note: Pre-trained model weights are hosted on Hugging Face at [`ModeAR/healthclaimbert`](https://huggingface.co/spaces/ModeAR/healthclaimbert).*
