import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

# -------------------------------
# ENV SETUP
# -------------------------------

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# -------------------------------
# FASTAPI APP
# -------------------------------

app = FastAPI(title="AI Public Health Chatbot")

# -------------------------------
# PRIVACY NOTICE
# -------------------------------

PRIVACY_NOTICE = (
    "Privacy Notice: This chatbot does not collect or store any personal data. "
    "All responses are for awareness only."
)

LOW_TECH_MODE = True

# -------------------------------
# KNOWLEDGE BASE
# -------------------------------

KNOWLEDGE_BASE = {
    "dengue": {
        "symptoms": ["High fever", "Severe headache", "Pain behind the eyes", "Joint pain", "Skin rash"],
        "prevention": ["Avoid stagnant water", "Use mosquito repellents", "Wear full sleeves"],
        "when_to_seek_help": ["Severe abdominal pain", "Bleeding", "Persistent vomiting"]
    },
    "malaria": {
        "symptoms": ["Fever with chills", "Sweating", "Headache"],
        "prevention": ["Mosquito nets", "Repellents"],
        "when_to_seek_help": ["High fever for more than 2 days"]
    },
    "tuberculosis": {
        "symptoms": ["Persistent cough", "Weight loss", "Night sweats"],
        "prevention": ["BCG vaccination", "Ventilation"],
        "when_to_seek_help": ["Coughing blood", "Chest pain"]
    },
    "diabetes": {
        "symptoms": ["Frequent urination", "Thirst", "Fatigue"],
        "prevention": ["Healthy diet", "Exercise"],
        "when_to_seek_help": ["Very high blood sugar"]
    },
}

OUTBREAK_DATA = {
    "pune": {"dengue": 120},
    "mumbai": {"chikungunya": 90}
}
OUTBREAK_THRESHOLD = 100

# -------------------------------
# HELPERS
# -------------------------------

def detect_language_simple(text: str):
    if any('\u0900' <= ch <= '\u097F' for ch in text):
        return "marathi"
    return "english"

def translate_with_groq(text: str, target_language: str):
    if target_language == "english" or not groq_client:
        return text

    try:
        prompt = f"Translate this text to {target_language}. Only translation:\n{text}"
        resp = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return text  # FAIL SAFE

def detect_disease(text: str):
    for d in KNOWLEDGE_BASE:
        if d in text:
            return d
    return None

def detect_intent(text: str):
    text = text.lower()
    if "outbreak" in text or "cases" in text:
        return "outbreak"
    if "vaccine" in text:
        return "vaccination"
    if "symptom" in text:
        return "symptoms"
    return "general"

def simplify_for_sms(text: str):
    return text.replace("\n\n", "\n")

# -------------------------------
# CORE RESPONSE ENGINE
# -------------------------------

def generate_response(user_text: str):
    disease = detect_disease(user_text)
    intent = detect_intent(user_text)

    if disease:
        data = KNOWLEDGE_BASE[disease]
        response = f"About {disease.capitalize()}:\n"
        response += "\nSymptoms:\n" + "\n".join(f"- {s}" for s in data["symptoms"])
        response += "\n\nPrevention:\n" + "\n".join(f"- {p}" for p in data["prevention"])
        response += "\n\nSeek help if:\n" + "\n".join(f"- {w}" for w in data["when_to_seek_help"])
        return response

    if intent == "outbreak":
        for city, diseases in OUTBREAK_DATA.items():
            alerts = [f"{d}: {c} cases" for d, c in diseases.items() if c >= OUTBREAK_THRESHOLD]
            if alerts:
                return f"Outbreak alert in {city}:\n" + "\n".join(alerts)
        return "No major outbreaks reported."

    return (
        "I can help with disease info, symptoms, prevention, vaccines, and outbreaks.\n"
        "Example: What is dengue?"
    )

def chatbot_response(user_text: str):
    user_lang = detect_language_simple(user_text)
    english_input = translate_with_groq(user_text, "english")
    english_response = generate_response(english_input.lower())

    final_response = (
        translate_with_groq(english_response, user_lang)
        if user_lang != "english"
        else english_response
    )

    if LOW_TECH_MODE:
        final_response = simplify_for_sms(final_response)

    return final_response + "\n\n" + PRIVACY_NOTICE

# -------------------------------
# API MODELS
# -------------------------------

class UserInput(BaseModel):
    text: str

# -------------------------------
# ROUTES
# -------------------------------

@app.get("/")
def root():
    return {
        "status": "running",
        "groq_configured": bool(GROQ_API_KEY),
        "docs": "/docs"
    }

@app.post("/predict")
def predict(input: UserInput):
    return {"answer": chatbot_response(input.text)}

@app.post("/sms")
def sms(input: UserInput):
    return {"reply": chatbot_response(input.text)}
