import os
import traceback
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------
# ENV SETUP
# -------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -------------------------------
# FASTAPI APP
# -------------------------------

app = FastAPI(title="HealthBot OpenAI API")

# -------------------------------
# REQUEST MODEL
# -------------------------------

class PredictRequest(BaseModel):
    text: str

# -------------------------------
# ROUTES
# -------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "openai_configured": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
        "docs": "/docs"
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if not client:
        return JSONResponse(
            status_code=500,
            content={"error": "OPENAI_API_KEY not set on server"}
        )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise public-health assistant."},
                {"role": "user", "content": req.text}
            ],
            temperature=0.2,
            max_tokens=300
        )

        answer = response.choices[0].message.content
        return {"answer": answer}

    except Exception as e:
        print("=== OpenAI ERROR ===")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "error": "OpenAI request failed",
                "detail": str(e)
            }
        )

# -------------------------------
# MOCK ENDPOINT (NO OPENAI)
# -------------------------------

@app.post("/predict_mock")
def predict_mock(req: PredictRequest):
    return {"answer": f"[MOCK RESPONSE] You said: {req.text}"}
