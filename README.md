# Fake News Detection

**NLP text-classification** pipeline that labels news articles as *Fake* or *True* using **TF-IDF** features and two scikit-learn classifiers: **Multinomial Naive Bayes** and **Logistic Regression**.

## Problem

Misinformation spreads quickly. This project builds a baseline classifier that learns, from ~45k labelled articles, to tell real news from fake news based on the article text alone.

## Approach

1. **Load & label**: merge `True.csv` and `Fake.csv`, add a `Label` column, shuffle.
2. **Data checks**: missing values and subject distribution per class.
3. **Split**: 80/20 train/test split (`random_state=10`).
4. **Pipelines**: `TfidfVectorizer(stop_words="english")` → `MultinomialNB` and `TfidfVectorizer` → `LogisticRegression`.
5. **Evaluate**: accuracy and confusion matrix for each model.
6. **Persist & reuse**: the Logistic Regression pipeline is saved with `joblib` (`models/Log_Reg.joblib`) and reloaded to classify new headlines.

## Results

Output of `python fake_news_detection.py` on the held-out 20% test set (8,980 articles):

| Model | Accuracy |
|-------|----------|
| TF-IDF + Multinomial Naive Bayes | 0.938 |
| TF-IDF + Logistic Regression | 0.984 |

**Caveat:** in this dataset, real articles come mostly from Reuters and often start with a "CITY (Reuters) -" dateline, so part of the accuracy can come from source-style cues rather than content. The model should not be expected to generalise to other news sources without further work (e.g. stripping datelines, cross-source evaluation).

## Tech stack

Python · pandas · NumPy · scikit-learn (TfidfVectorizer, MultinomialNB, LogisticRegression, Pipeline) · joblib

## Getting started

```bash
git clone https://github.com/raimondoDiSangro/fake-news-detection.git
cd fake-news-detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset (not committed, ~115 MB) as described in [`data/README.md`](data/README.md), then run:

```bash
python fake_news_detection.py
```

## Project structure

```
fake-news-detection/
├── fake_news_detection.py   # data prep, training, evaluation, model persistence
├── data/README.md           # how to download True.csv / Fake.csv
├── requirements.txt
└── LICENSE
```

## Dataset

[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle, Clément Bisaillon).

## License

[MIT](LICENSE)
