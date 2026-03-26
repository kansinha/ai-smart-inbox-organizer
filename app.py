import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_excel("email_dataset_new.xlsx")

# Handle missing values and combine subject + snippet
df["subject"] = df["subject"].fillna("")
df["snippet"] = df["snippet"].fillna("")
df["text"] = df["subject"] + " " + df["snippet"]

# Vectorize text
#vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
#X = vectorizer.fit_transform(df["text"])
#y = df["label"]

# Train model
#model = LogisticRegression(max_iter=1000)
#model.fit(X, y)

# Action mapping
def get_action(label):
    action_map = {
        "Promotion": "Move to Promotions folder",
        "Newsletter": "Move to Read Later",
        "Job Alert": "Move to Jobs folder",
        "Notification": "Keep in Inbox",
        "Important": "Mark as High Priority"
    }
    return action_map.get(label, "No action")

# Streamlit UI
st.title("AI Smart Inbox Organizer")
st.write("Paste an email message and AI will categorize it.")

email_input = st.text_area("Enter Email Text")

if st.button("Classify Email"):
    if email_input.strip() != "":
        probs = model.predict_proba(vectorizer.transform([email_input]))[0]
        prediction = model.classes_[probs.argmax()]
        confidence = probs.max()

        action = get_action(prediction)
        threshold = 0.70

        st.success(f"Predicted Category: {prediction}")
        st.info(f"Suggested Action: {action}")
        st.write(f"Confidence Score: {confidence:.2%}")

        if confidence < threshold:
            st.warning("⚠️ Low confidence prediction. Flagging for LLM review.")
            st.caption(
                f"ML predicted '{prediction}' but only with {confidence:.2%} certainty. "
                f"In production, this would be escalated to Gemini for deeper analysis."
            )
        else:
            st.success("✅ High confidence prediction. ML result accepted.")
            st.caption(
                f"Model is {confidence:.2%} certain — no LLM call needed. "
                f"This saves cost and latency."
            )
    else:
        st.error("Please enter some email text.")

# ---------------------------
# Inbox Analytics Dashboard
# ---------------------------
st.header("Inbox Analytics Dashboard")

label_counts = df["label"].value_counts()

col1, col2, col3 = st.columns(3)
col1.metric("Total Emails", len(df))
col2.metric("Top Category", label_counts.idxmax())
col3.metric("Top Category Count", int(label_counts.max()))

st.write("### Email Category Counts")
st.write(label_counts)

fig, ax = plt.subplots()
label_counts.plot(kind="bar", ax=ax)

ax.set_title("Email Category Distribution")
ax.set_xlabel("Category")
ax.set_ylabel("Count")

st.pyplot(fig)