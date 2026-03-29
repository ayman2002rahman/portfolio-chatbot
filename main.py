from contextlib import asynccontextmanager
from datetime import datetime, timezone
import asyncio
import json
import logging
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

log = logging.getLogger("portfolio-chatbot")

MONGODB_URI = os.environ["MONGODB_URI"]
MONGODB_DB_NAME = os.environ["MONGODB_DB_NAME"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

GEMINI_MODEL = "gemini-2.5-flash"
MAX_SESSIONS = 10
SESSION_TIMEOUT_MINUTES = 30
REAPER_INTERVAL_SECONDS = 60

COLLECTION_NAMES = ["projects", "skills", "experience", "education", "about"]

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

portfolio_context: str = ""

sessions: dict[str, dict] = {}


async def fetch_portfolio_data() -> str:
    """Connect to MongoDB Atlas, pull all portfolio collections, and return a
    formatted string suitable for a system prompt. Closes the connection when done."""
    mongo = AsyncIOMotorClient(MONGODB_URI)
    db = mongo[MONGODB_DB_NAME]

    sections: list[str] = []
    for name in COLLECTION_NAMES:
        docs = await db[name].find().to_list(length=None)
        if not docs:
            continue
        for doc in docs:
            doc.pop("_id", None)
        section = f"## {name.replace('_', ' ').title()}\n{json.dumps(docs, indent=2, default=str)}"
        sections.append(section)

    mongo.close()

    if not sections:
        log.warning("No portfolio data found in MongoDB -- system prompt will be empty")
        return ""

    return "\n\n".join(sections)


def build_system_prompt(portfolio_data: str) -> str:
    return (
        "You are a friendly and professional AI assistant embedded on a personal "
        "portfolio website. The portofilo owner is Ayman Rahman. Your role is to answer questions about the portfolio owner "
        "using ONLY the data provided below. If a question falls outside this data, "
        "politely say you don't have that information, but encourage them to reach out to Ayman via the contact page\n\n"
        "--- PORTFOLIO DATA ---\n"
        f"{portfolio_data}\n"
        "--- END PORTFOLIO DATA ---"
    )


async def reap_stale_sessions():
    """Periodically close sessions that have been idle too long."""
    while True:
        await asyncio.sleep(REAPER_INTERVAL_SECONDS)
        now = datetime.now(timezone.utc)
        stale = [
            sid
            for sid, s in sessions.items()
            if (now - s["last_activity"]).total_seconds() > SESSION_TIMEOUT_MINUTES * 60
        ]
        for sid in stale:
            entry = sessions.pop(sid, None)
            if entry and entry.get("ws"):
                try:
                    await entry["ws"].close(code=1000, reason="Session timed out")
                except Exception:
                    pass
            log.info("Reaped stale session %s", sid)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global portfolio_context

    log.info("Fetching portfolio data from MongoDB Atlas...")
    raw = await fetch_portfolio_data()
    portfolio_context = build_system_prompt(raw)
    log.info("Portfolio context loaded (%d chars)", len(portfolio_context))

    reaper_task = asyncio.create_task(reap_stale_sessions())

    yield

    reaper_task.cancel()
    try:
        await reaper_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Portfolio AI Chat", lifespan=lifespan)


@app.get("/")
def health():
    return {"status": "ok"}


@app.websocket("/ws/chat")
async def chat(ws: WebSocket):
    if len(sessions) >= MAX_SESSIONS:
        await ws.close(code=1013, reason="Server at capacity")
        return

    await ws.accept()

    session_id = uuid.uuid4().hex
    chat_session = gemini_client.aio.chats.create(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=portfolio_context,
        ),
    )
    sessions[session_id] = {
        "chat": chat_session,
        "ws": ws,
        "last_activity": datetime.now(timezone.utc),
    }
    log.info("Session %s connected (%d active)", session_id, len(sessions))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
                message = data.get("message", "")
            except (json.JSONDecodeError, AttributeError):
                message = raw

            if not message:
                await ws.send_json({"type": "error", "content": "Empty message"})
                continue

            sessions[session_id]["last_activity"] = datetime.now(timezone.utc)

            try:
                stream = await chat_session.send_message_stream(message)
                async for chunk in stream:
                    if chunk.text:
                        await ws.send_json({"type": "chunk", "content": chunk.text})
                await ws.send_json({"type": "end"})
            except Exception as e:
                log.exception("Gemini error in session %s", session_id)
                await ws.send_json({"type": "error", "content": str(e)})

    except WebSocketDisconnect:
        log.info("Session %s disconnected", session_id)
    finally:
        sessions.pop(session_id, None)
