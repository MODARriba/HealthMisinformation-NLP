# Detection of Health-Related Misinformation Using NLP

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ClaimCheck%20AI-blue)](https://modear-healthclaimbert.hf.space/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30+-yellow)](https://huggingface.co/docs/transformers)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **MSc / MCA Research Dissertation**  
> **Author:** Modar Riba  
> **Topic:** Detection of Health-Related Misinformation Using Natural Language Processing  
> **Interactive App:** [ClaimCheck AI on Hugging Face Spaces](https://modear-healthclaimbert.hf.space/)

---

## Research Overview

Health misinformation presents severe risks to public health by spreading unverified medical treatments, vaccine skepticism, and misleading dietary cures. This research project presents an end-to-end NLP framework to evaluate and contrast the capability of general and domain-specific pre-trained transformer architectures against classical machine learning baselines for health claim verification.

### Core Objectives:
1. **Dataset Integration & Harmonization**: Aggregated, cleaned, and standardized multi-source health claim datasets (**FakeHealth** + **HealthFact**) into a balanced binary benchmark (*Misinformation* vs. *Reliable*).
2. **Comparative Modeling**: Benchmarked **TF-IDF + Logistic Regression**, **BioBERT** (`dmis-lab/biobert-v1.1`), and **PubMedBERT** (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`).
3. **8-Bucket Error Analysis**: Detailed behavioral failure-mode analysis categorizing claims based on model agreement, disagreement, and edge-case errors.
4. **Interactive Decision Support**: Deployed an interactive clinical decision-support application (**ClaimCheck AI**) with real-time inference, model explanations, confidence metrics, and batch verification.

---

## Evaluation & Benchmark Results

All models were evaluated on the unified, held-out test split of **$n = 1,317$ health claims** (788 Reliable, 529 Misinformation).

### Quantitative Performance Comparison

| Model | Architecture | Accuracy | Precision | Recall | F1-Score | Macro-F1 | Misinfo-F1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF + LogReg** | N-gram Baseline | 72.06% | 0.7933 | 0.7208 | 0.7553 | 0.7148 | 0.6743 |
| **BioBERT** | Pre-trained Biomedical | **75.63%** | 0.7945 | **0.7995** | **0.7970** | 0.7461 | 0.6952 |
| **PubMedBERT** | Domain-Specific (PubMed) | 74.03% | **0.7989** | 0.7563 | 0.7771 | 0.7331 | 0.6891 |
| **Majority Ensemble** | Hard Voting (3 Models) | 75.40% | 0.7975 | 0.7805 | 0.7891 | **0.7470** | **0.7049** |

*Note: Precision, Recall, and F1-Score reflect the Reliable class. Misinfo-F1 and Macro-F1 measure balanced performance across both classes.*

### Confusion Matrices ($n = 1,317$)

```text
TF-IDF + Logistic Regression              BioBERT                                PubMedBERT
                Pred:Rel  Pred:Mis                     Pred:Rel  Pred:Mis                     Pred:Rel  Pred:Mis
Act: Reliable     568       220          Act: Reliable     630       158          Act: Reliable     596       192
Act: Misinfo      148       381          Act: Misinfo      163       366          Act: Misinfo      150       379
```

### Key Findings:
- **BioBERT** achieved the highest overall test accuracy (**75.63%**) and highest Reliable recall (**79.95%**), excelling at recognizing legitimate medical evidence.
- **PubMedBERT** achieved the highest precision (**79.89%**), minimizing false accusations against reliable health guidance.
- **Majority Vote Ensemble** produced the highest **Macro-F1 (0.7470)** and **Misinformation-F1 (0.7049)**, demonstrating that combining statistical n-gram signals with transformer embeddings mitigates individual model blind spots.

---

## Repository Structure

```text
├── Dataset/
│   ├── notebooks/                                # 9-stage research pipeline
│   │   ├── 00_dataset_overview.ipynb             # Schema inspection & class distribution
│   │   ├── 01_setup_and_processing.ipynb         # Data cleaning, normalization, train/dev/test split
│   │   ├── 02_eda_and_label_analysis.ipynb       # Exploratory data analysis & vocabulary stats
│   │   ├── 03_baseline_tfidf_logreg.ipynb        # TF-IDF + Logistic Regression baseline
│   │   ├── 04_transformer_biobert.ipynb          # BioBERT fine-tuning & evaluation
│   │   ├── 05_model_comparison_and_error.ipynb   # Error analysis between baseline & BioBERT
│   │   ├── 06_transformer_pubmedbert.ipynb       # PubMedBERT fine-tuning & evaluation
│   │   ├── 07_model_comparison_three_way.ipynb   # 3-way benchmark & 8-bucket error analysis
│   │   └── 08_single_claim_inference.ipynb       # Production inference pipeline
│   ├── dataset/
│   │   ├── processed/                            # Standardized binary datasets
│   │   └── raw/                                  # Source health datasets
│   └── output/
│       ├── model_comparison_three_way/           # Prediction CSVs, 8-bucket breakdown, metrics
│       └── [model]_metrics.json                  # Serialized evaluation metrics
├── streamlit-app/                                # ClaimCheck AI frontend
│   ├── app.py                                    # Streamlit application
│   ├── assets/                                   # UI visual assets
│   ├── data/                                     # Embedded evaluation metrics & sample claims
│   └── requirements.txt                          # Streamlit app dependencies
├── Menuscript/                                   # Thesis documentation & report drafts
├── PPT/                                          # Presentation slides & defense decks
├── .gitignore                                    # Excludes heavy model weights & cache files
└── README.md                                     # Project documentation
```

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/MODARriba/HealthMisinformation-NLP.git
cd HealthMisinformation-NLP
```

### 2. Set Up Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r streamlit-app/requirements.txt
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run streamlit-app/app.py
```

### 4. Live Cloud Deployment
You can access the deployed application without local installation:  
👉 **[https://modear-healthclaimbert.hf.space/](https://modear-healthclaimbert.hf.space/)**

---

## Model Weights on Hugging Face

Because model weights (`.safetensors` / `.pt`) are ~415 MB each (exceeding GitHub file size limits), the fine-tuned checkpoints are hosted directly on Hugging Face:

- **Space / Hub Repo:** [`ModeAR/healthclaimbert`](https://huggingface.co/spaces/ModeAR/healthclaimbert)

To load the fine-tuned models directly in Python:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "ModeAR/healthclaimbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

---

## Citation & Contact

```bibtex
@mastersthesis{riba2026misinformation,
  author       = {Modar Riba},
  title        = {Detection of Health-Related Misinformation Using Natural Language Processing},
  year         = {2026},
  school       = {Master of Computer Applications (MCA) / Computer Science},
  url          = {https://github.com/MODARriba/HealthMisinformation-NLP}
}
```

**Author:** Modar Riba ([@MODARriba](https://github.com/MODARriba))  
**Email:** ribamodar@gmail.com
