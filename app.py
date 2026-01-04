# app.py — HealthBot Backend (FINAL – Production Ready)
# Features:
# ✅ Multilingual (script-correct)
# ✅ Frontend city override + IP fallback
# ✅ Live outbreak alerts via NewsAPI
# ✅ Render/Vercel safe
# ✅ Markdown formatted output

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
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker


# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None


# --------------------------------------------------
# DATABASE (unchanged)
# --------------------------------------------------
DB_URL = os.getenv("HEALTH_DB_URL", "sqlite:///health.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
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
# BACKGROUND WORKER
# --------------------------------------------------
STOP_FLAG = False
_worker_thread = None


def reminder_worker(interval: int = 15):
    while not STOP_FLAG:
        try:
            db = SessionLocal()
            now = datetime.utcnow()

            rows = db.query(Reminder).filter(
                Reminder.sent == False,
                Reminder.remind_at <= now
            ).all()

            for r in rows:
                r.sent = True
                db.add(r)

            db.commit()
            db.close()
        except Exception:
            pass

        time.sleep(interval)


def start_worker():
    global _worker_thread
    if not _worker_thread or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(
            target=reminder_worker, daemon=True
        )
        _worker_thread.start()


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    start_worker()
    yield


app = FastAPI(
    title="HealthBot Backend",
    description="Multilingual AI Health Assistant with Live Outbreak Alerts",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/{path:path}")
async def preflight_handler(path: str, request: Request):
    return Response(status_code=200)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------
class ChatRequest(BaseModel):
    text: str
    language: str = "english"
    city: Optional[str] = None


# --------------------------------------------------
# LANGUAGE → STRICT SCRIPT ENFORCEMENT 🔥
# --------------------------------------------------
def get_language_instruction(language: str) -> str:
    instructions = {
        "english": "Respond in English.",
        "hindi": (
            "Respond ONLY in Hindi using **Devanagari script**.\n"
            "DO NOT use English letters.\n"
            "DO NOT use Hinglish.\n"
            "Example (correct): आपको सिरदर्द है\n"
            "Example (wrong): aapko headache hai"
        ),
        "marathi": (
            "Respond ONLY in Marathi using **Devanagari script**.\n"
            "Do NOT use English letters."
        ),
        "tamil": (
            "Respond ONLY in Tamil using **Tamil script**.\n"
            "Do NOT use English letters."
        ),
        "telugu": (
            "Respond ONLY in Telugu using **Telugu script**.\n"
            "Do NOT use English letters."
        ),
    }
    return instructions.get(language, "Respond in English.")


# --------------------------------------------------
# REAL CLIENT IP
# --------------------------------------------------
def get_real_ip(request: Request) -> Optional[str]:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip

    return request.client.host


# --------------------------------------------------
# IP → LOCATION
# --------------------------------------------------
def get_location_from_ip(ip: str):
    if not ip:
        return None

    try:
        r = requests.get(
            f"https://ipapi.co/{ip}/json/",
            timeout=5
        )
        data = r.json()
        return data.get("city"), data.get("region"), data.get("country_name")
    except Exception:
        return None


# --------------------------------------------------
# LIVE OUTBREAK CHECK (NewsAPI)
# --------------------------------------------------
def check_live_outbreaks(city: str):
    if not city or not NEWS_API_KEY:
        return None

    query = f"dengue OR malaria OR covid outbreak {city}"
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={query}&"
        "language=en&"
        "sortBy=publishedAt&"
        f"apiKey={NEWS_API_KEY}"
    )

    try:
        resp = requests.get(url, timeout=6).json()
        articles = resp.get("articles", [])[:3]

        if not articles:
            return None

        headlines = "\n".join(
            f"- **{a['title']}**"
            for a in articles
            if a.get("title")
        )

        return (
            f"🚨 **Local Health Alert — {city}**\n\n"
            f"{headlines}\n\n"
            "⚠️ Follow official health advisories.\n\n---\n"
        )
    except Exception:
        return None


# --------------------------------------------------
# ROOT
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "model": GROQ_MODEL,
        "groq_key_present": bool(GROQ_KEY),
        "news_api_present": bool(NEWS_API_KEY),
    }


# --------------------------------------------------
# MAIN CHAT ENDPOINT (FINAL)
# --------------------------------------------------
@app.post("/predict")
def predict(req: ChatRequest, request: Request):
    if not client:
        return JSONResponse(500, {"error": "GROQ_API_KEY missing"})

    try:
        language = req.language.lower()
        language_instruction = get_language_instruction(language)

        # City priority: frontend → IP → none
        city = None
        if req.city:
            city = req.city.strip().title()
        else:
            client_ip = get_real_ip(request)
            location = get_location_from_ip(client_ip)
            if location:
                city, _, _ = location

        outbreak_alert = check_live_outbreaks(city) if city else None

        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a public health assistant.\n\n"
                        f"{language_instruction}\n\n"
                        "Formatting rules:\n"
                        "- Use Markdown\n"
                        "- Each bullet point on a new line\n"
                        "- Use **bold headings**\n"
                        "- Keep paragraphs short\n"
                        "- If the script is incorrect, rewrite correctly"
                    )
                },
                {"role": "user", "content": req.text},
            ],
            max_tokens=500,
        )

        answer = resp.choices[0].message.content

        if outbreak_alert:
            answer = outbreak_alert + answer

        return {"answer": answer}

    except BadRequestError as e:
        return JSONResponse(400, {"error": str(e)})

    except Exception:
        print(traceback.format_exc())
        return JSONResponse(500, {"error": "internal_error"})
