import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "Log_Reg.joblib"

if not (DATA_DIR / "True.csv").exists() or not (DATA_DIR / "Fake.csv").exists():
    sys.exit("Dataset not found: download True.csv and Fake.csv into data/ (see data/README.md).")

true = pd.read_csv(DATA_DIR / "True.csv")
fake = pd.read_csv(DATA_DIR / "Fake.csv")

# looking at the data
print(true.head())
print(fake.head())

# searching for missing data in the datasets
print("Missing data per category in true news:\n", true.isna().sum())
print("Missing data per category in fake news:\n", fake.isna().sum())

# verifying the different subject of the articles in the datasets
print("DATA SUBJECTS IN THE TRUE SET:\n", true['subject'].value_counts())
print("DATA SUBJECTS IN THE FAKE SET:\n", fake['subject'].value_counts())

# adding a new value in the dataset to indicate if a news is fake
true['Label'] = 'True'
fake['Label'] = 'Fake'

# joining the two datasets and scrambling the sets
data = pd.concat([true, fake])
data = data.reset_index()
# scramble
data = data.sample(frac=1, random_state=10)

print("New dataset\n", data.head())

print(data['Label'].shape)

# Preprocess and split DATA SET into training and test
TRAIN_SET, TEST_SET, TRAIN_label, TEST_label = train_test_split(data['text'], data['Label'], random_state=10,
                                                                test_size=0.20)
print(f'TRAIN FEATURE SET: {TRAIN_SET.shape}')
print(f'TRAIN LABEL SET: {TRAIN_label.shape}')
print(f'TEST FEATURE SET: {TEST_SET.shape}')

# Building Pipeline for raw text transformation
clf = Pipeline([('vect', TfidfVectorizer(stop_words="english")), ('classifier', MultinomialNB())])

model_NB = clf.fit(TRAIN_SET, TRAIN_label)

score = model_NB.score(TEST_SET, TEST_label)
print(f'Naive Bayes accuracy: {score:.4f}')

# score validation with a confusion matrix
y_predicted_labels = model_NB.predict(TEST_SET)
cnf_matrix = confusion_matrix(TEST_label, y_predicted_labels)
np.set_printoptions(precision=2)
print("Confusion matrix:\n", cnf_matrix)

# Building Pipeline for the logistic regression model
clf = Pipeline([
    ('vect', TfidfVectorizer(stop_words="english")),
    ('classifier', LogisticRegression(max_iter=1000)),
])

model_LogReg = clf.fit(TRAIN_SET, TRAIN_label)

score = model_LogReg.score(TEST_SET, TEST_label)
print(f'Logistic Regression accuracy: {score:.4f}')

# Confusion Matrix
# Compute confusion matrix
y_predicted_labels = model_LogReg.predict(TEST_SET)
cnf_matrix = confusion_matrix(TEST_label, y_predicted_labels)
np.set_printoptions(precision=2)
print("Confusion matrix:\n", cnf_matrix)

MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model_LogReg, MODEL_PATH)
sample_example = ['two hundred people flew to Mars from my Country',
                  'India won the test series against Australia',
                  'World war started in 2020 and thanks to COVID it was stopped',
                  'World conference held in Winter']

model_loaded = joblib.load(MODEL_PATH)
for text in sample_example:
    print(f'Predicted label for new:{model_loaded.predict([text])[0]}')
