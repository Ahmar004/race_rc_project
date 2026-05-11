# Intelligent Reading Comprehension and Quiz Generation System

FAST-NUCES Islamabad — Artificial Intelligence Lab Project — BS CS Spring 2026

---

## Project Overview

This system is built on the RACE dataset and implements two ML pipelines exposed through an interactive Streamlit interface:

- **Model A** — Answer Verification and Question Generation (Logistic Regression, SVM, Naive Bayes, K-Means, Label Propagation, Soft Voting Ensemble)
- **Model B** — Distractor Generation and Hint Extraction (Logistic Regression ranker + Random Forest hint scorer)
- **UI** — Four-screen Streamlit application wiring both models together

---

## Repository Structure

```
race_rc_project/
├── data/
│   ├── raw/                    # original RACE CSVs (train.csv, val.csv, test.csv)
│   └── processed/              # cleaned splits from preprocessing notebook
│       ├── train_clean.csv
│       ├── dev_clean.csv
│       ├── test_clean.csv
│       ├── train_options.csv
│       ├── dev_options.csv
│       └── test_options.csv
├── models/
│   ├── model_a/                # pkl files from Model A notebook
│   │   ├── vectorizer.pkl
│   │   ├── logistic_regression.pkl
│   │   ├── svm_classifier.pkl
│   │   ├── naive_bayes.pkl
│   │   ├── scaler.pkl
│   │   ├── svd_reducer.pkl
│   │   ├── kmeans_model.pkl
│   │   └── label_propagation.pkl
│   └── model_b/                # pkl files from Model B notebook
│       ├── distractor_ranker.pkl
│       └── hint_scorer.pkl
├── ui/
│   └── app.py                  # Streamlit application (4 screens)
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── raceeda.ipynb
│   ├── Model_A.ipynb
│   └── Model_B.ipynb
├── images/                     # confusion matrix images from Model A notebook
├── images_b/                   # confusion matrix images from Model B notebook
├── report/
│   └── final_report.pdf
├── requirements.txt
└── README.md
```

---

## quick-run help, if you have already the project structure setup:

```bash
cd race_rc_project
pip install -r requirements.txt
streamlit run ui/app.py
```

---

## Setup Instructions

### 1. Clone / download the repository

```bash
git clone https://github.com/<your-username>/race-rc-project.git
cd race-rc-project
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Get the RACE dataset

Download from: https://www.kaggle.com/datasets/ankitdhiman7/race-dataset

Place `train.csv`, `val.csv`, `test.csv` inside `data/raw/`.

### 4. Run the training notebooks (Kaggle or Colab)

The four notebooks must be run in this order on Kaggle (or Google Colab with T4 GPU):

1. `notebooks/preprocessing.ipynb` — produces cleaned CSVs in `data/processed/`
2. `notebooks/raceeda.ipynb` — exploratory data analysis
3. `notebooks/Model_A.ipynb` — trains LR, SVM, NB, K-Means, Label Propagation, Ensemble; saves pkl files
4. `notebooks/Model_B.ipynb` — trains distractor ranker and hint scorer; saves pkl files

After running, download all generated `.pkl` files and place them as shown in the structure above:

- Files from Model A notebook → `models/model_a/`
- Files from Model B notebook → `models/model_b/`
- Confusion matrix images from Model A → `images/`
- Confusion matrix images from Model B → `images_b/`

### 5. Run the Streamlit app

```bash
streamlit run ui/app.py
```

The app opens at `http://localhost:8501`.

---

## Usage

### Screen 1 — Article Input
- Paste any English reading passage into the text area, or click **Load Random RACE Sample** to pull a real RACE test example.
- Click **Submit** to run both Model A and Model B inference.

### Screen 2 — Quiz View
- Displays the question and four options (A, B, C, D).
- Select an answer and click **Check Answer**. Model A's ensemble score is shown alongside the result.
- Green = correct, Red = incorrect.

### Screen 3 — Hint Panel
- Three graduated hints from Model B are unlocked one at a time.
- The **Reveal Answer** button appears only after all three hints have been used.

### Screen 4 — Analytics Dashboard
- Shows Model A and Model B evaluation metrics from the training run.
- Tracks live inference latency for the current session.
- Provides a CSV export of the session log.

---

## Dataset

RACE (ReAding Comprehension from Examinations) — Lai et al., EMNLP 2017.
~28,000 passages, ~100,000 multiple-choice questions from Chinese middle/high school English exams.

| Split | Rows (options) |
|-------|---------------|
| Train | 281,168       |
| Dev   | 35,148        |
| Test  | 35,148        |

---

## Model Summary

### Model A — Answer Verification

| Model                | Type             | Dev Accuracy | Dev F1 Macro |
|----------------------|------------------|-------------|-------------|
| Logistic Regression  | Supervised       | 0.7102      | 0.6218      |
| SVM (LinearSVC)      | Supervised       | 0.7334      | 0.6461      |
| Naive Bayes          | Supervised       | 0.6811      | 0.5847      |
| K-Means (mapped)     | Unsupervised     | 0.7450      | 0.5293      |
| Label Propagation    | Semi-Supervised  | 0.6933      | 0.5811      |
| Soft Voting Ensemble | Ensemble         | 0.7205      | 0.6324      |

Features: TF-IDF (max_features=10000, sublinear_tf, bigrams) on concatenated article + question + option text.

### Model B — Distractor & Hint Generation

| Component         | Dev Accuracy | Dev F1 Macro |
|-------------------|-------------|-------------|
| Distractor Ranker | 0.7612      | 0.7431      |
| Hint Scorer       | 0.7048      | 0.6814      |

BLEU/ROUGE evaluation on 100 test samples:

| Metric  | Score  |
|---------|--------|
| BLEU    | 0.0017 |
| ROUGE-1 | 0.0278 |
| ROUGE-2 | 0.0030 |
| ROUGE-L | 0.0270 |

---

## Requirements

See `requirements.txt`. Key dependencies:

```
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
rouge-score>=0.1.2
nltk>=3.8.0
scipy>=1.11.0
```

---

## Team

- Member 1 — EDA, Preprocessing, Model A, Model B (Kaggle notebooks)
- Member 2 — Streamlit UI, Final Report, README, Code Quality (this repository)

---

## References

- Lai et al. (2017). RACE: Large-scale ReAding Comprehension Dataset From Examinations. EMNLP.
- Du et al. (2017). Learning to Ask: Neural Question Generation for Reading Comprehension. ACL.
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. NAACL.
- Papineni et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. ACL.
- Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries. ACL Workshop.
