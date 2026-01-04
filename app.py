# app.py — HealthBot Backend (FINAL: IP-based + Live Outbreak Alerts)

import os
import threading
import time
import traceback
from datetime import datetime
from typing import List
from contextlib import asynccontextmanager

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


def reminder_worker(check_interval: int = 15):
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

        time.sleep(check_interval)


def start_worker():
    global _worker_thread
    if not _worker_thread or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=reminder_worker, daemon=True)
        _worker_thread.start()


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    start_worker()
    yield


app = FastAPI(title="HealthBot Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
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


# --------------------------------------------------
# HELPER: Detect outbreak intent
# --------------------------------------------------
def is_outbreak_query(text: str) -> bool:
    keywords = [
        "outbreak", "cases", "spread", "near me",
        "nearby", "epidemic", "dengue", "malaria", "covid"
    ]
    text = text.lower()
    return any(k in text for k in keywords)


# --------------------------------------------------
# HELPER: IP → LOCATION
# --------------------------------------------------
def get_location_from_ip(ip: str):
    try:
        resp = requests.get(
            f"https://ipapi.co/{ip}/json/",
            timeout=5
        )
        data = resp.json()
        return data.get("city"), data.get("country_name")
    except Exception:
        return None, None


# --------------------------------------------------
# LIVE OUTBREAK CHECK (News API)
# --------------------------------------------------
def check_live_outbreaks(city: str):
    if not city or not NEWS_API_KEY:
        return None

    query = (
        f"{city} dengue OR malaria OR covid outbreak "
        "health advisory"
    )

    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 3,
        "apiKey": NEWS_API_KEY,
    }

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=5
        )
        data = resp.json()
        articles = data.get("articles", [])

        if not articles:
            return None

        headlines = "\n".join(
            f"- **{a['title']}**"
            for a in articles
            if a.get("title")
        )

        return (
            f"🚨 **Live Health Alerts near {city}**\n\n"
            f"{headlines}\n\n"
            "_Source: NewsAPI_\n\n---\n"
        )
    except Exception:
        return None


# --------------------------------------------------
# API ROUTES
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "model": GROQ_MODEL,
        "groq_key_present": bool(GROQ_KEY),
        "news_api_present": bool(NEWS_API_KEY),
    }


@app.post("/predict")
def predict(req: ChatRequest, request: Request):
    if not client:
        return JSONResponse(
            status_code=500,
            content={"error": "GROQ_API_KEY missing"}
        )

    try:
        language = req.language.lower()
        client_ip = request.client.host

        city, country = get_location_from_ip(client_ip)

        outbreak_alert = None
        if is_outbreak_query(req.text):
            outbreak_alert = check_live_outbreaks(city)

        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a public health assistant.\n"
                        f"Respond ONLY in {language}.\n\n"
                        "Formatting rules:\n"
                        "- Use Markdown\n"
                        "- Bullet points on new lines\n"
                        "- Short paragraphs\n"
                        "- Clear medical guidance\n"
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
        return JSONResponse(status_code=400, content={"error": str(e)})

    except Exception:
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error"}
        )
