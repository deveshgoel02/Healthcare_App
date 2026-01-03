# app.py — HealthBot Backend (API-only, Render-safe)

import os
import threading
import time
import traceback
from datetime import datetime
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Groq client
from groq import Groq, BadRequestError

# SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker


# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None


# --------------------------------------------------
# DATABASE SETUP
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


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    message = Column(Text)
    region = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# --------------------------------------------------
# REMINDER BACKGROUND WORKER
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
                print(f"[Reminder] {r.message}")
                r.sent = True
                db.add(r)

            db.commit()
            db.close()
        except Exception as e:
            print("Reminder worker error:", e)

        time.sleep(check_interval)


def start_worker():
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return
    _worker_thread = threading.Thread(target=reminder_worker, daemon=True)
    _worker_thread.start()


# --------------------------------------------------
# FASTAPI LIFESPAN
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    start_worker()
    yield


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
app = FastAPI(title="HealthBot Backend", lifespan=lifespan)


# --------------------------------------------------
# CORS
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later to Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------
class ChatRequest(BaseModel):
    text: str


class SymptomCheckRequest(BaseModel):
    symptoms: List[str]
    age: int | None = None
    comorbidities: List[str] | None = None


# --------------------------------------------------
# HEALTH LOGIC
# --------------------------------------------------
DISEASE_SYMPTOMS = {
    "dengue": {"fever", "headache", "joint pain", "rash"},
    "malaria": {"fever", "chills", "sweating"},
    "covid-19": {"fever", "cough", "fatigue"},
}


def match_diseases(symptoms: List[str]):
    s = {x.lower() for x in symptoms}
    results = []
    for d, core in DISEASE_SYMPTOMS.items():
        overlap = len(s & core)
        if overlap:
            results.append((d, round(overlap / len(core) * 100, 1)))
    return sorted(results, key=lambda x: -x[1])


# --------------------------------------------------
# API ROUTES
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "model": GROQ_MODEL,
        "groq_key_present": bool(GROQ_KEY),
    }


@app.post("/predict")
def predict(req: ChatRequest):
    if not client:
        return JSONResponse(
            status_code=500,
            content={"error": "GROQ_API_KEY missing"}
        )

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a public health assistant."},
                {"role": "user", "content": req.text},
            ],
            max_tokens=300,
        )
        return {"answer": resp.choices[0].message.content}

    except BadRequestError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    except Exception:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": "internal_error"})


@app.post("/symptom_check")
def symptom_check(req: SymptomCheckRequest):
    return {"probable": match_diseases(req.symptoms)}
