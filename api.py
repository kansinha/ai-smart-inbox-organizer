from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
from google import genai

# ==============================
# LLM FUNCTION
# ==============================
def ask_llm(email_text):
    try:
        client = genai.Client(api_key="YOUR_GEMINI_API_KEY_HERE")
        prompt = f"""You are an expert email classifier for a smart inbox organizer.

CATEGORIES AND DEFINITIONS:
1. Important
   - Personal emails requiring your direct action or response
   - Deadlines, urgent requests, direct messages from people you know

2. Notification
   - Automated alerts, reminders, confirmations, event invites
   - System updates, shipping updates, calendar reminders
   - LinkedIn/social event invitations and webinar invites

3. Newsletter
   - Regular content digests sent to subscribers
   - News roundups, blog updates, weekly/monthly publications

4. Promotion
   - Emails SELLING products or services with discounts/offers
   - Marketing emails with promo codes, sales, special offers

5. Job Alert
   - Specific job postings and recruitment emails

IMPORTANT RULES:
- LinkedIn event invites and webinars → Notification
- School/community reminders → Notification
- Emails with discount codes or product sales → Promotion
- Specific job postings → Job Alert
- Direct requests, meeting requests, deadlines → Important

Classify this email into EXACTLY ONE category.
Reply with ONLY the category name — no explanation.

Email:
{email_text}

Category:"""

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        if not response or not response.candidates:
            print("⚠️ Empty LLM response")
            return None

        result = response.candidates[0].content.parts[0].text.strip()
        print("🤖 LLM RAW RESPONSE:", result)

        valid = ["Important", "Notification", "Newsletter",
                 "Promotion", "Job Alert"]
        for category in valid:
            if category.lower() in result.lower():
                return category
        return "Notification"

    except Exception as e:
        print("❌ Gemini error:", e)
        return None


# ==============================
# LOAD ML MODEL
# ==============================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# ==============================
# CREATE APP
# ==============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# REQUEST FORMAT
# ==============================
class EmailInput(BaseModel):
    text: str = None
    email_text: str = None


# ==============================
# ACTION MAPPING
# ==============================
action_map = {
    "Promotion":    "Move to Promotions folder",
    "Newsletter":   "Move to Read Later",
    "Job Alert":    "Move to Jobs folder",
    "Notification": "Keep in Inbox",
    "Important":    "Mark as High Priority"
}


# ==============================
# API ENDPOINT
# ==============================
@app.post("/classify")
def classify_email(data: EmailInput):
    text = data.text or data.email_text or ""

    if not text.strip():
        return {"error": "Empty email text"}

    # Step 1 — ML prediction
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    ml_prediction = model.classes_[probs.argmax()]
    confidence = float(probs.max())

    print(f"📧 Email received")
    print(f"⚡ ML prediction: {ml_prediction} ({confidence:.2%})")

    threshold = 0.70

    # Step 2 — High confidence: trust ML
    if confidence >= threshold:
        print(f"✅ High confidence — using ML result")
        return {
            "category":     ml_prediction,
            "ml_category":  ml_prediction,
            "llm_category": None,
            "confidence":   round(confidence * 100, 1),
            "action":       action_map.get(ml_prediction, "Review manually"),
            "needs_llm":    False,
            "llm_used":     False,  # ← ML handled it
            "message":      "✅ High confidence — ML result accepted"
        }

    # Step 3 — Low confidence: call Gemini
    print(f"⚠️ Low confidence ({confidence:.2%}) — calling Gemini...")
    llm_prediction = ask_llm(text)

    if llm_prediction:
        # Gemini succeeded ✅
        clean_prediction = llm_prediction.strip().replace(".", "")
        print(f"🤖 Gemini result: {clean_prediction}")
        return {
            "category":     clean_prediction,
            "ml_category":  ml_prediction,
            "llm_category": clean_prediction,
            "confidence":   round(confidence * 100, 1),
            "action":       action_map.get(clean_prediction, "Review manually"),
            "needs_llm":    True,
            "llm_used":     True,   # ← Gemini handled it
            "message":      f"🤖 ML uncertain ({round(confidence*100,1)}%) — Gemini classified as: {clean_prediction}"
        }
    else:
        # Gemini failed ❌ — fall back to ML
        print(f"❌ Gemini failed — falling back to ML result")
        return {
            "category":     ml_prediction,
            "ml_category":  ml_prediction,
            "llm_category": None,
            "confidence":   round(confidence * 100, 1),
            "action":       action_map.get(ml_prediction, "Review manually"),
            "needs_llm":    True,
            "llm_used":     False,  # ← Gemini failed, ML used
            "message":      "⚠️ LLM failed — fallback to ML prediction"
        }


# ==============================
# HEALTH CHECK
# ==============================
@app.get("/")
def root():
    return {"status": "✅ AI Smart Inbox API running — Hybrid ML + Gemini"}