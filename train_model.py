import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_excel("email_dataset_new.xlsx")

df["subject"] = df["subject"].fillna("")
df["snippet"] = df["snippet"].fillna("")
df["text"] = df["subject"] + " " + df["snippet"]

vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")