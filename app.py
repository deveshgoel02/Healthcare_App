# app.py — HealthBot Backend (FINAL + Twilio SMS & WhatsApp)

import os
import threading
import time
import traceback
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from groq import Groq, BadRequestError
from twilio.rest import Client as TwilioClient
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker


# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_API_KEY_SID = os.getenv("TWILIO_API_KEY_SID")
TWILIO_API_KEY_SECRET = os.getenv("TWILIO_API_KEY_SECRET")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

twilio_client = (
    TwilioClient(
        username=TWILIO_API_KEY_SID,
        password=TWILIO_API_KEY_SECRET,
        account_sid=TWILIO_ACCOUNT_SID,
    )
    if TWILIO_API_KEY_SID and TWILIO_API_KEY_SECRET
    else None
)


# --------------------------------------------------
# DATABASE
# --------------------------------------------------
engine = create_engine("sqlite:///health.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Reminder(Base):
    __tablename__ = "reminders"
    id = Column(Integer, primary_key=True)
    phone = Column(String)
    message = Column(Text)
    remind_at = Column(DateTime)
    sent = Column(Boolean, default=False)


Base.metadata.create_all(bind=engine)


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="HealthBot Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/{path:path}")
async def preflight_handler(path: str, request: Request):
    return Response(status_code=200)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------
class ChatRequest(BaseModel):
    text: str
    language: str = "english"
    city: Optional[str] = None


class MessageRequest(BaseModel):
    to: str
    message: str


# --------------------------------------------------
# LANGUAGE INSTRUCTIONS
# --------------------------------------------------
def get_language_instruction(language: str) -> str:
    rules = {
        "english": "Respond in English.",
        "hindi": "Respond ONLY in Hindi using Devanagari script. No English letters.",
        "marathi": "Respond ONLY in Marathi using Devanagari script.",
        "tamil": "Respond ONLY in Tamil script.",
        "telugu": "Respond ONLY in Telugu script.",
    }
    return rules.get(language, "Respond in English.")


# --------------------------------------------------
# IP → LOCATION
# --------------------------------------------------
def get_real_ip(request: Request):
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0] if forwarded else request.client.host


def get_location_from_ip(ip: str):
    try:
        r = requests.get(f"https://ipapi.co/{ip}/json/", timeout=5)
        data = r.json()
        return data.get("city")
    except Exception:
        return None


# --------------------------------------------------
# LIVE OUTBREAK CHECK
# --------------------------------------------------
def check_live_outbreaks(city: str):
    if not city or not NEWS_API_KEY:
        return None

    query = f"dengue OR malaria OR covid outbreak {city}"
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={query}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    )

    try:
        data = requests.get(url, timeout=5).json()
        articles = data.get("articles", [])[:3]
        if not articles:
            return None

        headlines = "\n".join(f"- **{a['title']}**" for a in articles if a.get("title"))

        return f"🚨 **Local Health Alert — {city}**\n\n{headlines}\n\n---\n"
    except Exception:
        return None


# --------------------------------------------------
# MAIN CHAT
# --------------------------------------------------
@app.post("/predict")
def predict(req: ChatRequest, request: Request):
    if not groq_client:
        return JSONResponse(500, {"error": "GROQ_API_KEY missing"})

    try:
        language = req.language.lower()
        instruction = get_language_instruction(language)

        city = req.city or get_location_from_ip(get_real_ip(request))
        outbreak = check_live_outbreaks(city) if city else None

        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": f"You are a health assistant.\n{instruction}"},
                {"role": "user", "content": req.text},
            ],
            max_tokens=500,
        )

        answer = resp.choices[0].message.content
        if outbreak:
            answer = outbreak + answer

        return {"answer": answer}

    except Exception:
        print(traceback.format_exc())
        return JSONResponse(500, {"error": "internal_error"})


# --------------------------------------------------
# SMS ENDPOINT
# --------------------------------------------------
@app.post("/send_sms")
def send_sms(req: MessageRequest):
    if not twilio_client:
        return JSONResponse(500, {"error": "Twilio not configured"})

    msg = twilio_client.messages.create(
        body=req.message,
        from_=TWILIO_PHONE_NUMBER,
        to=req.to,
    )

    return {"status": "sent", "sid": msg.sid}


# --------------------------------------------------
# WHATSAPP ENDPOINT
# --------------------------------------------------
@app.post("/send_whatsapp")
def send_whatsapp(req: MessageRequest):
    if not twilio_client:
        return JSONResponse(500, {"error": "Twilio not configured"})

    msg = twilio_client.messages.create(
        body=req.message,
        from_=TWILIO_WHATSAPP_NUMBER,
        to=f"whatsapp:{req.to}",
    )

    return {"status": "sent", "sid": msg.sid}
