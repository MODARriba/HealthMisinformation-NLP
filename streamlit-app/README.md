# HealthClaimBERT \U0001F9EC

Streamlit web app for **health misinformation detection**, built on the models
fine-tuned in the accompanying Jupyter notebooks (PubMedBERT, BioBERT,
TF-IDF + Logistic Regression) and evaluated on a combined FakeHealth +
HealthFact dataset (test set n = 1,311).

## Project layout

```
streamlit-app/
├── app.py                                  # 4-tab Streamlit frontend
├── requirements.txt
├── models/                                 # <- copy your trained models here
│   ├── pubmedbert_fakehealth_healthfact/   #    HF save_pretrained folder
│   ├── biobert_fakehealth_healthfact/      #    HF save_pretrained folder
│   └── tfidf_logreg.joblib                 #    scikit-learn pipeline
├── assets/                                 # confusion matrix PNGs (optional)
│   ├── tfidf_cm.png
│   ├── biobert_cm.png
│   └── pubmedbert_cm.png
├── data/
│   ├── comparison_summary.json             # from notebook 07 (optional)
│   └── eight_bucket.csv                    # bucket,count (optional)
└── 0X_*.ipynb                              # training & analysis notebooks
```

The app works even if model files are missing - it shows a friendly warning
and the Performance Dashboard still renders from embedded thesis metrics.

## 1. Export your trained artifacts from Jupyter

In the notebooks where the models were trained:

```python
# Transformers (notebooks 04 and 06)
model.save_pretrained("models/pubmedbert_fakehealth_healthfact")
tokenizer.save_pretrained("models/pubmedbert_fakehealth_healthfact")
# (same for biobert_fakehealth_healthfact)

# Baseline (notebook 03)
import joblib
joblib.dump(pipeline, "models/tfidf_logreg.joblib")

# Confusion matrices
fig.savefig("assets/pubmedbert_cm.png", dpi=150, bbox_inches="tight")
```

## 2. Add the files to this repo

Model weights are large - use **Git LFS**:

```bash
git lfs install
git lfs track "*.bin" "*.safetensors" "*.joblib" "*.pkl"
git add .gitattributes models/ assets/ data/
git commit -m "Add trained model weights and assets"
git push
```

## 3. Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tabs

1. **Claim Checker** - classify a single claim with the best transformer
   (auto-selected from `data/comparison_summary.json`, like notebook 08).
2. **Model Comparison** - run one claim through all three models side by side.
3. **Batch Inference** - upload a CSV (column `claim`) or TXT (one claim per
   line, max 200 rows), download results, view a confidence histogram.
4. **Performance Dashboard** - real thesis metrics, confusion matrices,
   Plotly radar chart, and the eight-bucket error analysis.

## Test-set results (thesis Chapter 6.2)

| Model        | Accuracy | Precision | Recall | F1     |
|--------------|----------|-----------|--------|--------|
| TF-IDF + LR  | 68.73%   | 0.7713    | 0.6790 | 0.7222 |
| BioBERT      | 73.38%   | 0.7290    | 0.8841 | 0.7991 |
| PubMedBERT   | 73.84%   | 0.7908    | 0.7656 | 0.7780 |

---
MSc Dissertation \u00b7 Health Misinformation Detection \u00b7 2024-26
