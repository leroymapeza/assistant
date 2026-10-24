import json
import logging
import contextvars
import os
import re
import secrets
import socket
import sqlite3
import ssl
import threading
import time
import urllib.request
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from http.cookies import SimpleCookie
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

logging.basicConfig(level=logging.INFO, format="[ziva] %(message)s")
log = logging.getLogger("ziva")
for noisy in ("google_genai", "httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

_original_getaddrinfo = socket.getaddrinfo
_dns_cache = {}
DOH_SERVERS = ["1.1.1.1", "8.8.8.8", "9.9.9.9"]
DNS_TTL = 300
DNS_FAIL_TTL = 120
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _doh_lookup(host):
    # Certificate checks are skipped for the lookup only; the API connection itself is still verified.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    errors = []
    for server in DOH_SERVERS:
        try:
            req = urllib.request.Request(
                f"https://{server}/dns-query?name={host}&type=A",
                headers={"Accept": "application/dns-json"},
            )
            with urllib.request.urlopen(req, context=ctx, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for answer in data.get("Answer", []):
                if answer.get("type") == 1:
                    log.info("DoH resolved %s to %s via %s", host, answer["data"], server)
                    return answer["data"]
            errors.append(f"{server}: no A record")
        except Exception as e:
            errors.append(f"{server}: {type(e).__name__} {e}")
    log.warning("DoH failed for %s (%s)", host, "; ".join(errors))
    return None


def _doh_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if (not isinstance(host, str) or host == "localhost" or _IPV4.match(host)
            or ":" in host or family == socket.AF_INET6):
        return _original_getaddrinfo(host, port, family, type, proto, flags)
    hit = _dns_cache.get(host)
    if not hit or hit[1] < time.time():
        ip = _doh_lookup(host)
        hit = _dns_cache[host] = (ip, time.time() + (DNS_TTL if ip else DNS_FAIL_TTL))
    return _original_getaddrinfo(hit[0] or host, port, family, type, proto, flags)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ziva.db")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEFAULT_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"]
DEMO = os.environ.get("ZIVA_DEMO", "").lower() in ("1", "true", "yes")
HOURLY = int(os.environ.get("ZIVA_HOURLY", "20"))
DAILY = int(os.environ.get("ZIVA_DAILY", "400"))
TTL_HOURS = int(os.environ.get("ZIVA_TTL_HOURS", "24"))
MAX_LEN = 600 if DEMO else 4000
_owner = contextvars.ContextVar("owner", default="")
_VID = re.compile(r"[A-Za-z0-9_-]{22,}")
_state = {"model": None}


class ZivaError(Exception):
    pass


def load_config():
    cfg = {"google_api_key": "", "models": []}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("config.json unreadable: %s", e)
    return cfg


if load_config().get("doh"):
    socket.getaddrinfo = _doh_getaddrinfo


@lru_cache(maxsize=2)
def _make_client(key):
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=20000))


def get_client():
    key = os.environ.get("GOOGLE_API_KEY") or load_config().get("google_api_key", "")
    if not key or "PASTE" in key:
        return None
    return _make_client(key)


def _discover(client):
    try:
        names = [
            m.name.removeprefix("models/") for m in client.models.list()
            if "generateContent" in (m.supported_actions or []) and "flash" in (m.name or "")
            and not re.search(r"image|tts|live|audio|embed|lite|preview", m.name)
        ]
    except Exception as e:
        log.warning("Model discovery failed: %s", e)
        return []
    return sorted(names, key=lambda n: [float(x) for x in re.findall(r"\d+\.?\d*", n)], reverse=True)[:4]


def model_sequence(client, fast=False):
    seen = set()
    _state["not_found"] = False
    known = ([_state["model"]] if _state["model"] else []) + list(load_config().get("models") or DEFAULT_MODELS)
    if fast:
        known.sort(key=lambda m: "lite" not in m)
    for source in (lambda: known, lambda: _discover(client) if _state["not_found"] else []):
        for model in source():
            if model not in seen:
                seen.add(model)
                yield model


def _check(e):
    code = getattr(e, "code", None)
    if code in (401, 403) or (code == 400 and "API key" in str(e)):
        raise ZivaError("Google rejected the API key. Check google_api_key in config.json.")
    if code == 404:
        _state["not_found"] = True


def _transient(e):
    return getattr(e, "code", None) in (429, 500, 502, 503, 504) or "timed out" in str(e).lower()


def _failure_message(e):
    code = getattr(e, "code", None)
    if code == 429:
        return "Gemini’s rate limit was reached. Wait a moment and try again."
    if code in (500, 502, 503, 504):
        return "Gemini is overloaded right now. Try again in a few seconds."
    if _transient(e):
        return "Gemini took too long to answer. Try again in a few seconds."
    return "Couldn’t reach Gemini. Check your internet connection and try again."


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_owner(conn, table):
    if "owner" not in [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN owner TEXT NOT NULL DEFAULT ''")


def init_db():
    with db() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, created_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, created_at TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0)")
        conn.execute("CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE, role TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT NOT NULL, error INTEGER NOT NULL DEFAULT 0, stopped INTEGER NOT NULL DEFAULT 0)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id)")
        for table in ("notes", "reminders", "conversations"):
            _ensure_owner(conn, table)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(owner, updated_at)")


@asynccontextmanager
async def lifespan(_app):
    init_db()
    if DEMO:
        _purge(force=True)
    yield


class Visitor:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not DEMO:
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        jar = SimpleCookie(headers.get(b"cookie", b"").decode("latin-1"))
        vid = jar["ziva_vid"].value if "ziva_vid" in jar else ""
        fresh = not _VID.fullmatch(vid)
        if fresh:
            vid = secrets.token_urlsafe(16)
        secure = headers.get(b"x-forwarded-proto", b"") == b"https"

        async def wrapped(message):
            if fresh and message["type"] == "http.response.start":
                cookie = f"ziva_vid={vid}; Path=/; Max-Age=2592000; HttpOnly; SameSite=Lax" + ("; Secure" if secure else "")
                message["headers"] = [*message.get("headers", []), (b"set-cookie", cookie.encode())]
            await send(message)

        token = _owner.set(vid)
        try:
            await self.app(scope, receive, wrapped)
        finally:
            _owner.reset(token)


app = FastAPI(lifespan=lifespan)
app.add_middleware(Visitor)

_hits, _hits_lock, _last_purge = {}, threading.Lock(), [0.0]


def _rate_limit(request):
    ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()) or (request.client.host if request.client else "")
    rules = (((("v", _owner.get())), HOURLY, 3600), ((("i", ip)), HOURLY * 2, 3600), ((("all",)), DAILY, 86400))
    now = time.time()
    with _hits_lock:
        if len(_hits) > 5000:
            for k in [k for k, q in _hits.items() if not q or now - q[-1] > 86400]:
                del _hits[k]
        queues = []
        for key, cap, window in rules:
            q = [t for t in _hits.get(key, []) if now - t < window]
            _hits[key] = q
            if len(q) >= cap:
                return "Today’s demo capacity has been used up. Please check back tomorrow." if key == ("all",) else f"You’ve reached the demo limit of {HOURLY} messages per hour. Please try again a little later."
            queues.append(q)
        for q in queues:
            q.append(now)
    return None


def _purge(force=False):
    if not force and time.time() - _last_purge[0] < 600:
        return
    _last_purge[0] = time.time()
    cutoff = (datetime.now() - timedelta(hours=TTL_HOURS)).isoformat(timespec="seconds")
    with db() as conn:
        conn.execute("DELETE FROM conversations WHERE updated_at < ?", (cutoff,))
        conn.execute("DELETE FROM notes WHERE created_at < ?", (cutoff,))
        conn.execute("DELETE FROM reminders WHERE created_at < ?", (cutoff,))


class Turn(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    message: str
    conversation_id: int | None = None
    replace_from: int | None = None
    retry: bool = False


class Rename(BaseModel):
    title: str


DEFAULT_TITLE = "New chat"


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _title_from(text, limit=48):
    t = re.sub(r"\s+", " ", text or "").strip()
    return t if len(t) <= limit else t[:limit - 1].rstrip() + "…"


def _create_conversation(conn):
    now = _now()
    return conn.execute("INSERT INTO conversations (title, created_at, updated_at, owner) VALUES (?, ?, ?, ?)", (DEFAULT_TITLE, now, now, _owner.get())).lastrowid


def _add_message(conn, conv, role, text, error=False):
    now = _now()
    mid = conn.execute(
        "INSERT INTO messages (conversation_id, role, text, created_at, error) VALUES (?, ?, ?, ?, ?)",
        (conv, role, text, now, int(error)),
    ).lastrowid
    conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv))
    return mid


def _history(conn, conv):
    rows = conn.execute("SELECT role, text, error FROM messages WHERE conversation_id = ? ORDER BY id", (conv,)).fetchall()
    turns, pending = [], None
    for r in rows:
        if r["role"] == "user":
            pending = r["text"]
        else:
            if pending is not None and not r["error"] and r["text"].strip():
                turns += [Turn(role="user", text=pending), Turn(role="assistant", text=r["text"])]
            pending = None
    return turns[-12:]


def add_note(text):
    with db() as conn:
        conn.execute("INSERT INTO notes (text, created_at, owner) VALUES (?, ?, ?)", (text, _now(), _owner.get()))


def add_reminder(text):
    with db() as conn:
        conn.execute("INSERT INTO reminders (text, created_at, completed, owner) VALUES (?, ?, 0, ?)", (text, _now(), _owner.get()))


def _when(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%d %b, %H:%M")
    except ValueError:
        return iso


def list_notes():
    with db() as conn:
        rows = conn.execute("SELECT text, created_at FROM notes WHERE owner = ? ORDER BY id DESC LIMIT 20", (_owner.get(),)).fetchall()
    if not rows:
        return "You have no notes yet."
    return "**Your notes**\n\n" + "\n".join(f"- {r['text']} *({_when(r['created_at'])})*" for r in rows)


def list_reminders():
    with db() as conn:
        rows = conn.execute("SELECT text, created_at, completed FROM reminders WHERE owner = ? ORDER BY completed, id DESC LIMIT 20", (_owner.get(),)).fetchall()
    if not rows:
        return "You have no reminders yet."
    lines = [f"- {r['text']} *({'done' if r['completed'] else _when(r['created_at'])})*" for r in rows]
    return "**Your reminders**\n\n" + "\n".join(lines)


def help_text():
    return (
        "**What I can do**\n\n"
        "- Answer questions and talk things through\n"
        "- Save notes: “note that the router password is on the fridge”\n"
        "- Add reminders: “remind me to call Tendai on Friday”\n"
        "- Show them back: “show my notes” or “what are my reminders?”\n\n"
        "Type or tap the microphone to speak."
    )


COMMANDS = {
    "help": help_text, "/help": help_text, "?": help_text,
    "notes": list_notes, "show notes": list_notes, "list notes": list_notes,
    "reminders": list_reminders, "show reminders": list_reminders, "list reminders": list_reminders,
}

ROUTER_PROMPT = """You are an AI router. Read the user's latest message and output ONLY a JSON object with keys "action" and "content".
Actions: "create_note", "create_reminder", "list_notes", "list_reminders", "chat".
- "create_note": user wants to save a fact, thought or text. content = the text to save.
- "create_reminder": user wants a task or reminder. content = the task.
- "list_notes": user wants to see saved notes. content = "".
- "list_reminders": user wants to see reminders. content = "".
- "chat": ANYTHING else. Questions, facts, current events, casual greetings. content = the user's question.
Output ONLY valid JSON. No markdown, no explanations."""

ACTIONS = {"create_note", "create_reminder", "list_notes", "list_reminders", "chat"}


def route_message(client, message, history):
    context = "\n".join(f"{t.role}: {t.text}" for t in history[-4:])
    prompt = f"Recent conversation:\n{context}\n\nLatest message: {message}" if context else message
    config = types.GenerateContentConfig(
        system_instruction=ROUTER_PROMPT, temperature=0.0, response_mime_type="application/json",
        http_options=types.HttpOptions(timeout=10000),
    )
    last = None
    for attempt in range(2):
        for model in model_sequence(client, fast=True):
            try:
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (resp.text or "").strip())
                data = json.loads(raw)
                action = data.get("action") if data.get("action") in ACTIONS else "chat"
                content = (data.get("content") or "").strip() or message
                _state["model"] = model
                return action, content
            except Exception as e:
                log.warning("Router model %s failed: %s", model, e)
                _check(e)
                last = e
        if attempt == 0 and _transient(last):
            time.sleep(2)
            continue
        break
    if isinstance(last, ValueError):
        return "chat", message
    raise ZivaError(_failure_message(last))


def stream_reply(client, history, message):
    contents = [types.Content(role="user" if t.role == "user" else "model", parts=[types.Part(text=t.text)]) for t in history]
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))
    config = types.GenerateContentConfig(system_instruction=(
        "You are Ziva, a warm, concise personal assistant from Africa. Answer naturally and briefly. "
        f"Light markdown is fine. Today is {datetime.now().strftime('%A, %d %B %Y')}."
    ))
    last = None
    for attempt in range(2):
        for model in model_sequence(client):
            started = False
            try:
                for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
                    if chunk.text:
                        started = True
                        yield chunk.text
                _state["model"] = model
                return
            except Exception as e:
                log.warning("Stream model %s failed: %s", model, e)
                if started:
                    yield "\n\n*The reply was cut short. Send your message again.*"
                    return
                _check(e)
                last = e
        if attempt == 0 and _transient(last):
            time.sleep(2)
            continue
        break
    raise ZivaError(_failure_message(last))


_LIST = re.compile(r"^(?:show|list|view|display|see|open|get|what(?:['’]s|\s+is|\s+are)|which|do\s+i\s+have|any)\b.*\bmy\b.*\b(notes?|reminders?|to-?dos?|tasks?)\b", re.I)
_SEP = r"(?:\s*[:,\-]\s*|\s+)(.+)$"
_REMIND = re.compile(r"^(?:please\s+)?(?:remind\s+me(?:\s+(?:to|about|that))?|(?:set|add|create|make)\s+(?:a\s+)?reminder(?:\s+(?:to|for|that|about))?|reminder:|don[’']?t\s+(?:let\s+me\s+)?forget(?:\s+to)?|remember\s+to)" + _SEP, re.I | re.S)
_NOTE = re.compile(r"^(?:please\s+)?(?:(?:take|make|add|save|write)\s+(?:a\s+)?note(?:\s+(?:that|of))?|note\s+(?:that|down)|note:|jot\s+down|write\s+down|remember\s+that)" + _SEP, re.I | re.S)
SYSTEM_PROMPT = "You are Ziva, a warm, concise personal assistant from Africa. Answer naturally and briefly. Light markdown is fine."


def route_rules(message):
    text = message.strip()
    m = _LIST.match(text)
    if m:
        return ("list_notes" if m.group(1).lower().startswith("note") else "list_reminders"), ""
    for pattern, action in ((_REMIND, "create_reminder"), (_NOTE, "create_note")):
        m = pattern.match(text)
        if m:
            return action, m.group(1).strip()
    return "chat", message


def stream_openai(cfg, history, message):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": t.role, "content": t.text} for t in history]
    messages.append({"role": "user", "content": message})
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = "Bearer " + cfg["api_key"]
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps({"model": cfg.get("model", "local"), "messages": messages, "stream": True}).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=cfg.get("timeout", 120)) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            choices = json.loads(data).get("choices") or [{}]
            piece = choices[0].get("delta", {}).get("content")
            if piece:
                yield piece


def reply_stream(client, history, message):
    cfg = load_config()
    local = cfg.get("openai_compat")
    backends = []
    if client:
        backends.append(lambda: stream_reply(client, history, message))
    if local:
        backends.append(lambda: stream_openai(local, history, message))
    if cfg.get("chat_backend") == "local":
        backends.reverse()
    err = None
    for make in backends:
        started = False
        try:
            for piece in make():
                started = True
                yield piece
            return
        except ZivaError as e:
            err = e
        except Exception as e:
            log.warning("Chat backend failed: %s", e)
            if started:
                yield "\n\n*The reply was cut short. Send your message again.*"
                return
            err = ZivaError("The chat backend didn’t respond. Check that it is running and try again.")
    raise err or ZivaError("No chat backend is configured.")


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


def _produce(message, history):
    command = COMMANDS.get(message.lower())
    if command:
        return "text", command(), False, False
    cfg = load_config()
    client = get_client()
    if not client and not cfg.get("openai_compat"):
        return "text", "No API key found. Add your Google API key to config.json (google_api_key) or set GOOGLE_API_KEY.", True, False
    try:
        action, content = route_message(client, message, history) if client and cfg.get("llm_router") else route_rules(message)
        log.info("Router: action=%s", action)
        if action == "create_note":
            add_note(content)
            return "text", f"Saved to your notes: {content}", False, True
        if action == "create_reminder":
            add_reminder(content)
            return "text", f"Added to your reminders: {content}", False, True
        if action == "list_notes":
            return "text", list_notes(), False, False
        if action == "list_reminders":
            return "text", list_reminders(), False, False
        stream = reply_stream(client, history, message)
        return "stream", stream, next(stream)
    except StopIteration:
        return "text", "I didn’t get an answer back. Try rephrasing that.", True, False
    except ZivaError as e:
        return "text", str(e), True, False


def _start_turn(req, blocked=None):
    message = (req.message or "").strip()[:MAX_LEN]
    if not message:
        raise HTTPException(400, "Empty message")
    with db() as conn:
        conv = req.conversation_id
        if not conv or not conn.execute("SELECT 1 FROM conversations WHERE id = ? AND owner = ?", (conv, _owner.get())).fetchone():
            conv = _create_conversation(conn)
        start = req.replace_from
        if req.retry:
            start = conn.execute("SELECT MAX(id) FROM messages WHERE conversation_id = ? AND role = 'user'", (conv,)).fetchone()[0]
        if start:
            conn.execute("DELETE FROM messages WHERE conversation_id = ? AND id >= ?", (conv, start))
        if not conn.execute("SELECT 1 FROM messages WHERE conversation_id = ?", (conv,)).fetchone():
            conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (DEFAULT_TITLE, conv))
        history = _history(conn, conv)
        user_id = _add_message(conn, conv, "user", message)
        conn.execute("UPDATE conversations SET title = ? WHERE id = ? AND title = ?", (_title_from(message), conv, DEFAULT_TITLE))
    return conv, user_id, ("text", blocked, True, False) if blocked else _produce(message, history)


def _save_assistant(conv, text, error=False):
    with db() as conn:
        return _add_message(conn, conv, "assistant", text, error)


@app.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    blocked = None
    if DEMO:
        _purge()
        blocked = _rate_limit(request)
    conv, user_id, out = await run_in_threadpool(_start_turn, req, blocked)
    gone = await request.is_disconnected()
    ids = {"conversation_id": conv, "user_id": user_id}
    if out[0] == "text":
        if gone:
            return Response(status_code=204)
        _, text, error, changed = out
        return {"reply": text, "error": error, "changed": changed, "assistant_id": _save_assistant(conv, text, error), **ids}

    _, stream, first = out
    if gone:
        stream.close()
        return Response(status_code=204)
    aid = _save_assistant(conv, first)

    async def body():
        parts, done = [first], False
        try:
            yield first
            while True:
                piece = await run_in_threadpool(next, stream, None)
                if piece is None:
                    done = True
                    break
                parts.append(piece)
                yield piece
        finally:
            with db() as conn:
                conn.execute("UPDATE messages SET text = ?, stopped = ? WHERE id = ?", ("".join(parts), int(not done), aid))
            try:
                stream.close()
            except ValueError:
                pass

    headers = {"X-Conversation": str(conv), "X-User-Msg": str(user_id), "X-Asst-Msg": str(aid)}
    return StreamingResponse(body(), media_type="text/plain; charset=utf-8", headers=headers)


@app.get("/api/config")
def public_config():
    return {"demo": DEMO, "hourly": HOURLY, "ttl": TTL_HOURS}


@app.get("/api/conversations")
def conversations(q: str = ""):
    q = q.strip()
    with db() as conn:
        if q:
            like = f"%{q}%"
            rows = conn.execute(
                "SELECT id, title, updated_at FROM conversations WHERE owner = ? AND (title LIKE ? OR id IN (SELECT conversation_id FROM messages WHERE text LIKE ?)) ORDER BY updated_at DESC, id DESC LIMIT 200",
                (_owner.get(), like, like),
            )
        else:
            rows = conn.execute("SELECT id, title, updated_at FROM conversations WHERE owner = ? ORDER BY updated_at DESC, id DESC LIMIT 200", (_owner.get(),))
        return [dict(r) for r in rows]


@app.post("/api/conversations")
def create_conversation():
    if DEMO:
        _purge()
    with db() as conn:
        return {"id": _create_conversation(conn), "title": DEFAULT_TITLE}


@app.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: int):
    with db() as conn:
        conv = conn.execute("SELECT id, title FROM conversations WHERE id = ? AND owner = ?", (conv_id, _owner.get())).fetchone()
        if not conv:
            raise HTTPException(404)
        rows = conn.execute("SELECT id, role, text, error, stopped FROM messages WHERE conversation_id = ? ORDER BY id", (conv_id,))
        return {**dict(conv), "messages": [dict(r) for r in rows]}


@app.patch("/api/conversations/{conv_id}")
def rename_conversation(conv_id: int, body: Rename):
    title = _title_from(body.title, 80) or DEFAULT_TITLE
    with db() as conn:
        found = conn.execute("UPDATE conversations SET title = ? WHERE id = ? AND owner = ?", (title, conv_id, _owner.get())).rowcount
    if not found:
        raise HTTPException(404)
    return {"title": title}


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: int):
    with db() as conn:
        found = conn.execute("DELETE FROM conversations WHERE id = ? AND owner = ?", (conv_id, _owner.get())).rowcount
    if not found:
        raise HTTPException(404)
    return {"ok": True}


@app.get("/api/items")
def items():
    with db() as conn:
        owner = (_owner.get(),)
        notes = [dict(r) for r in conn.execute("SELECT id, text, created_at FROM notes WHERE owner = ? ORDER BY id DESC LIMIT 200", owner)]
        reminders = [dict(r) for r in conn.execute("SELECT id, text, created_at, completed FROM reminders WHERE owner = ? ORDER BY completed, id DESC LIMIT 200", owner)]
    return {"notes": notes, "reminders": reminders}


@app.post("/api/reminders/{item_id}/toggle")
def toggle_reminder(item_id: int):
    with db() as conn:
        found = conn.execute("UPDATE reminders SET completed = 1 - completed WHERE id = ? AND owner = ?", (item_id, _owner.get())).rowcount
    if not found:
        raise HTTPException(404)
    return {"ok": True}


@app.delete("/api/{kind}/{item_id}")
def delete_item(kind: str, item_id: int):
    if kind not in ("notes", "reminders"):
        raise HTTPException(404)
    with db() as conn:
        found = conn.execute(f"DELETE FROM {kind} WHERE id = ? AND owner = ?", (item_id, _owner.get())).rowcount
    if not found:
        raise HTTPException(404)
    return {"ok": True}


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Ziva</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2048%2048'%3E%3Cpath%20d='M24%203%2045%2024%2024%2045%203%2024Z'%20fill='none'%20stroke='%23E0714A'%20stroke-width='2.6'%20stroke-linecap='round'%20stroke-dasharray='0%201'%20pathLength='24'/%3E%3Cpath%20d='M24%208%2040%2024%2024%2040%208%2024Z'%20fill='%23F0B34A'/%3E%3Cpath%20d='M24%2012.5%2035.5%2024%2024%2035.5%2012.5%2024Z'%20fill='%232A1D05'/%3E%3Cpath%20d='M19.5%2020h9l-9%208h9'%20fill='none'%20stroke='%23F0B34A'%20stroke-width='2.6'%20stroke-linejoin='bevel'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600&family=Young+Serif&display=swap" media="print" onload="this.media='all'">
<style>
:root{--bg:#10111A;--panel:#181A27;--raised:#222537;--user:#262A40;--line:rgba(255,255,255,.07);--text:#ECE5D8;--muted:#8E90A8;--gold:#F0B34A;--ink:#2A1D05;--copper:#E0714A;--serif:"Young Serif",Georgia,serif;--sans:"Hanken Grotesk",system-ui,"Segoe UI",sans-serif;color-scheme:dark}
*{box-sizing:border-box;scrollbar-width:thin;scrollbar-color:#2E3248 transparent}
[hidden]{display:none!important}
html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--text);font:15.5px/1.65 var(--sans);-webkit-font-smoothing:antialiased;overflow:hidden}
button{font:inherit;color:inherit;cursor:pointer}
:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
.shell{height:100dvh;display:flex}
.app{flex:1;min-width:0;height:100dvh;position:relative;display:grid;grid-template-rows:auto auto minmax(0,1fr) auto}
.wall{height:8px;background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='8'%3E%3Cpath d='M0 7.5 5 1.5l5 6 5-6 5 6' fill='none' stroke='%23F0B34A' stroke-width='1.6'/%3E%3C/svg%3E") repeat-x;opacity:.85;-webkit-mask-image:linear-gradient(90deg,transparent,#000 22%,#000 78%,transparent);mask-image:linear-gradient(90deg,transparent,#000 22%,#000 78%,transparent)}
.col{width:min(720px,calc(100% - 32px));margin-inline:auto}
.mark{width:28px;height:28px;flex:none;display:grid;place-items:center}
.mark svg{width:100%;height:100%;display:block}
.brand{display:flex;align-items:center;gap:10px}
.brand h1{margin:0;font:400 22px/1 var(--serif)}

#side{width:288px;flex:none;display:flex;flex-direction:column;background:var(--panel);box-shadow:1px 0 0 var(--line);padding:max(14px,env(safe-area-inset-top)) 10px 10px;margin-left:0;transition:margin-left .25s ease,transform .25s ease,visibility 0s}
body.noside #side{margin-left:-288px;visibility:hidden;transition:margin-left .25s ease,visibility 0s .25s}
.shead{display:flex;align-items:center;justify-content:space-between;padding:2px 6px 14px}
#sideClose{display:none}
.newbtn{display:flex;align-items:center;gap:10px;width:100%;border:0;border-radius:12px;padding:10px 12px;background:var(--gold);color:var(--ink);font-weight:600;font-size:14.5px;transition:background .15s}
.newbtn:hover{background:#F6C26A}
.search{display:flex;align-items:center;gap:8px;margin:10px 0 4px;padding:0 12px;border-radius:12px;background:var(--bg);color:var(--muted);box-shadow:inset 0 0 0 1px var(--line)}
.search:focus-within{box-shadow:inset 0 0 0 1px rgba(240,179,74,.55)}
.search input{flex:1;min-width:0;background:none;border:0;outline:0;color:var(--text);font:inherit;font-size:14px;padding:9px 0}
.search input::placeholder{color:var(--muted)}
#convs{flex:1;overflow-y:auto;margin:0 -4px;padding:0 4px 8px}
#convs h3{margin:16px 8px 4px;font:600 11px/1 var(--sans);letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.conv{position:relative;display:flex;align-items:center;border-radius:10px;transition:background .15s}
.conv:hover,.conv.active{background:var(--raised)}
.conv.active{box-shadow:inset 3px 0 0 var(--gold)}
.ctitle{flex:1;min-width:0;text-align:left;background:none;border:0;padding:9px 12px;font-size:14.5px;line-height:1.35;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cacts{position:absolute;right:2px;top:50%;transform:translateY(-50%);display:none;padding-left:16px;background:linear-gradient(90deg,transparent,var(--raised) 16px)}
.conv:hover .cacts,.conv:focus-within .cacts,.conv.active .cacts{display:flex}
.cacts button{width:28px;height:28px;display:grid;place-items:center;border:0;border-radius:8px;background:none;color:var(--muted)}
.cacts button:hover{color:var(--text);background:rgba(255,255,255,.06)}
.cacts button.arm{color:#fff;background:var(--copper)}
.rename{flex:1;min-width:0;margin:3px 4px;background:var(--bg);border:0;border-radius:8px;padding:6px 9px;color:var(--text);font:inherit;font-size:14.5px;outline:2px solid var(--gold)}
.sfoot{padding:10px 8px 2px;font-size:12px;line-height:1.5;color:var(--muted);border-top:1px solid var(--line)}
#sscrim{display:none}

.bar{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:10px 0 8px}
.lead{display:flex;align-items:center;gap:8px;min-width:0}
.lead .mark{display:none;width:24px;height:24px}
body.noside .lead .mark{display:grid}
#ttl{margin:0;font:400 17px/1.2 var(--serif);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.acts{display:flex;align-items:center;gap:2px;flex:none}
.pill{display:flex;align-items:center;gap:8px;background:none;border:0;border-radius:999px;padding:8px 12px;font-size:14px;color:var(--muted);transition:background .15s,color .15s}
.pill:hover{background:var(--panel);color:var(--text)}
#badge{background:var(--gold);color:var(--ink);border-radius:999px;font-size:11px;font-weight:600;min-width:18px;height:18px;padding:0 5px;display:grid;place-items:center}
#chat{overflow-y:auto;-webkit-mask-image:linear-gradient(transparent,#000 14px,#000 calc(100% - 22px),transparent);mask-image:linear-gradient(transparent,#000 14px,#000 calc(100% - 22px),transparent)}
.thread{padding:18px 0 30px}
.empty{padding-top:8vh;animation:rise .5s ease both}
.empty .mark{width:64px;height:64px;margin-bottom:18px}
.empty h2{margin:0 0 8px;font:400 clamp(30px,5vw,42px)/1.1 var(--serif)}
.empty p{margin:0 0 26px;color:var(--muted);font-size:16.5px}
.chips{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
@media(max-width:560px){.chips{grid-template-columns:1fr}}
.chip{text-align:left;background:none;border:1px solid var(--line);border-radius:12px;padding:12px 14px;font-size:14.5px;line-height:1.4;color:var(--muted);transition:background .15s,color .15s,border-color .15s}
.chip:hover{background:var(--panel);color:var(--text);border-color:rgba(240,179,74,.35)}
@keyframes rise{from{opacity:0;transform:translateY(8px)}}
.msg{display:flex;margin-bottom:16px;animation:rise .25s ease both}
.msg.user{justify-content:flex-end}
.stack{display:flex;flex-direction:column;align-items:flex-end;max-width:80%;min-width:0}
.stack.editing{width:100%;max-width:100%}
.bubble{background:var(--user);padding:9px 15px;border-radius:18px 18px 4px 18px;max-width:100%;white-space:pre-wrap;overflow-wrap:anywhere}
.main{flex:1;min-width:0}
.body{overflow-wrap:anywhere}
.body p{margin:0 0 .75em}
.body>:last-child{margin-bottom:0}
.body ul,.body ol{margin:.2em 0 .75em;padding-left:1.3em}
.body li{margin:.2em 0}
.body li::marker{color:var(--gold)}
.body h4{margin:1em 0 .4em;font:400 1.05em var(--serif)}
.body a{color:var(--gold);text-underline-offset:3px}
.body code{background:var(--raised);padding:.1em .35em;border-radius:6px;font:.88em ui-monospace,Consolas,monospace}
.body pre{position:relative;background:var(--panel);border-radius:12px;padding:12px 14px;overflow-x:auto}
.body pre code{background:none;padding:0}
.cp{position:absolute;top:8px;right:8px;background:var(--raised);border:0;border-radius:8px;padding:3px 10px;font-size:12px;color:var(--muted);opacity:0;transition:opacity .15s}
.body pre:hover .cp,.cp:focus-visible{opacity:1}
.cp:hover{color:var(--text)}
@media(hover:none){.cp{opacity:1}}
.hint{color:var(--muted)}
.msg.error .body{color:#F3B8A3;border-left:2px solid var(--copper);padding-left:12px}
.streaming .body>:last-child::after{content:"";display:inline-block;width:8px;height:8px;margin-left:6px;border-radius:50%;background:var(--gold);animation:beat 1s ease-in-out infinite}
@keyframes beat{50%{transform:scale(.45);opacity:.5}}
.think{display:inline-flex;align-items:center;gap:12px;padding:4px 0}
.dots{display:inline-flex;align-items:center;gap:6px;height:20px}
.dots i{width:9px;height:9px;border-radius:50%;background:var(--gold);animation:wave 1.15s cubic-bezier(.45,.05,.55,.95) infinite}
.dots i:nth-child(2){animation-delay:.15s}
.dots i:nth-child(3){animation-delay:.3s}
@keyframes wave{0%,62%,100%{transform:translateY(0) scale(.75);opacity:.5;background:var(--gold);box-shadow:none}30%{transform:translateY(-8px) scale(1.2);opacity:1;background:var(--copper);box-shadow:0 0 12px rgba(224,113,74,.55)}}
.lbl{font-size:14px;color:transparent;background:linear-gradient(90deg,var(--muted) 35%,var(--text) 50%,var(--muted) 65%) 120% 0/250% 100%;-webkit-background-clip:text;background-clip:text;animation:sheen 2.2s linear infinite}
@keyframes sheen{to{background-position:-120% 0}}
.tools{height:26px;display:flex;align-items:center;gap:2px}
.msg.user .tools{justify-content:flex-end}
.tool{opacity:0;background:none;border:0;border-radius:8px;padding:5px;color:var(--muted);display:grid;place-items:center}
.msg:hover .tool,.msg:last-child .tool,.tool:focus-visible{opacity:1}
.tool:hover{background:var(--raised);color:var(--text)}
@media(hover:none){.tool{opacity:1}}
.tag{font-size:12px;color:var(--muted);margin-right:6px;padding:1px 9px;border-radius:999px;background:var(--raised)}
.edit{width:100%;resize:none;background:var(--panel);border:0;border-radius:14px;padding:10px 14px;color:inherit;font:inherit;line-height:1.5;box-shadow:inset 0 0 0 1px rgba(240,179,74,.55);outline:0}
.editbar{display:flex;align-items:center;gap:8px;margin-top:8px;width:100%;font-size:12.5px;color:var(--muted)}
.editbar span{flex:1}
.btn{border:0;border-radius:999px;padding:7px 16px;background:var(--gold);color:var(--ink);font-weight:600;font-size:14px}
.btn.ghost{background:var(--raised);color:var(--text)}
.composer{position:relative;padding:6px 0 max(18px,env(safe-area-inset-bottom))}
.jump{position:absolute;left:50%;top:-44px;transform:translateX(-50%);width:36px;height:36px;border:0;border-radius:50%;background:var(--raised);color:var(--text);display:grid;place-items:center;box-shadow:0 6px 18px rgba(0,0,0,.4),inset 0 0 0 1px var(--line)}
.box{display:flex;align-items:flex-end;gap:2px;background:var(--panel);border-radius:26px;padding:6px 6px 6px 20px;box-shadow:inset 0 0 0 1px var(--line),0 10px 30px rgba(0,0,0,.35);transition:box-shadow .2s}
.box:focus-within{box-shadow:inset 0 0 0 1px rgba(240,179,74,.55),0 10px 30px rgba(0,0,0,.35)}
textarea{flex:1;min-width:0;resize:none;background:none;border:0;outline:0;color:inherit;font:inherit;line-height:1.5;padding:8px 0;max-height:160px}
textarea:focus-visible{outline:0}
textarea::placeholder{color:var(--muted)}
.ibtn{width:38px;height:38px;flex:none;border:0;border-radius:50%;background:none;color:var(--muted);display:grid;place-items:center;transition:background .15s,color .15s}
.ibtn:hover{background:var(--raised);color:var(--text)}
.ibtn[aria-pressed=true]{color:var(--gold)}
.ibtn:disabled{opacity:.35;cursor:not-allowed}
#mic.on{background:var(--copper);color:#fff;animation:pulse 1.2s infinite}
@keyframes pulse{50%{box-shadow:0 0 0 7px rgba(224,113,74,.2)}}
#send{background:var(--gold);color:var(--ink);margin-left:4px}
#send:hover{background:#F6C26A}
#send:disabled{background:var(--raised);color:var(--muted);opacity:1;cursor:default}
#send.stop{background:var(--copper);color:#fff}
#scrim{position:fixed;inset:0;background:rgba(6,7,12,.55);opacity:0;pointer-events:none;transition:opacity .25s}
#scrim.open{opacity:1;pointer-events:auto}
#drawer{position:fixed;top:0;right:0;bottom:0;width:min(380px,100vw);background:rgba(24,26,39,.88);-webkit-backdrop-filter:blur(20px);backdrop-filter:blur(20px);box-shadow:-1px 0 0 var(--line);transform:translateX(100%);transition:transform .25s ease;display:flex;flex-direction:column;padding-top:env(safe-area-inset-top);z-index:40}
#drawer.open{transform:none}
.dhead{display:flex;align-items:center;justify-content:space-between;padding:16px 16px 4px 22px}
.dhead h2{margin:0;font:400 20px var(--serif)}
.tabs{display:flex;gap:4px;margin:8px 22px 0;padding:3px;background:var(--bg);border-radius:999px;width:fit-content}
.tab{background:none;border:0;border-radius:999px;padding:6px 16px;font-size:14px;color:var(--muted)}
.tab[aria-selected=true]{background:var(--raised);color:var(--text)}
#list{list-style:none;margin:0;padding:10px 22px 24px;overflow-y:auto;flex:1}
.item{display:flex;gap:12px;align-items:flex-start;padding:12px 0;border-bottom:1px solid var(--line)}
.check{width:20px;height:20px;flex:none;margin-top:2px;border-radius:50%;border:1.5px solid var(--muted);background:none;display:grid;place-items:center;color:transparent;padding:0;transition:background .15s,border-color .15s}
.check:hover{border-color:var(--gold)}
.item.done .check{background:var(--gold);border-color:var(--gold);color:var(--ink)}
.item .t{flex:1;min-width:0;overflow-wrap:anywhere}
.item.done .t div{color:var(--muted);text-decoration:line-through}
.item small{color:var(--muted);font-size:12.5px}
.del{opacity:0;background:none;border:0;color:var(--muted);padding:4px;border-radius:8px}
.item:hover .del,.del:focus-visible{opacity:1}
.del:hover{color:var(--copper)}
@media(hover:none){.del{opacity:1}}
.none{color:var(--muted);padding:18px 8px;margin:0;font-size:14px}
li.none{padding:22px 0}
@media(max-width:900px){
  #side{position:fixed;inset:0 auto 0 0;z-index:30;width:min(320px,88vw);margin-left:0!important;transform:translateX(-100%);visibility:hidden;transition:transform .25s ease,visibility 0s .25s}
  body.sideopen #side{transform:none;visibility:visible;transition:transform .25s ease,visibility 0s}
  #sideClose{display:grid}
  .lead .mark{display:grid}
  #sscrim{display:block;position:fixed;inset:0;z-index:25;background:rgba(6,7,12,.55);opacity:0;pointer-events:none;transition:opacity .25s}
  body.sideopen #sscrim{opacity:1;pointer-events:auto}
  .conv .cacts{display:flex}
  .conv:not(.active) .cacts{display:none}
}
@media(max-width:520px){.pill span{display:none}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<div class="shell">
  <aside id="side" aria-label="Conversations">
    <div class="shead">
      <div class="brand"><span class="mark" data-mark></span><h1>Ziva</h1></div>
      <button id="sideClose" class="ibtn" type="button" aria-label="Close chats" data-ic="close"></button>
    </div>
    <button id="newChat" class="newbtn" type="button" title="New chat (Ctrl+Shift+O)"><i data-ic="plus" style="display:contents"></i>New chat</button>
    <label class="search"><i data-ic="search" style="display:contents"></i><input id="q" type="search" placeholder="Search chats" aria-label="Search chats" title="Search chats (Ctrl+K)" autocomplete="off"></label>
    <div id="convs"></div>
    <div class="sfoot" id="sfoot" hidden></div>
  </aside>
  <div class="app">
    <div class="wall" aria-hidden="true"></div>
    <header class="col bar">
      <div class="lead">
        <button id="toggleSide" class="ibtn" type="button" aria-label="Chats" title="Chats" data-ic="panel"></button>
        <span class="mark" data-mark></span>
        <h2 id="ttl">New chat</h2>
      </div>
      <div class="acts">
        <button id="exportBtn" class="ibtn" type="button" aria-label="Download chat" title="Download chat" data-ic="download" disabled></button>
        <button id="open" class="pill" type="button" aria-controls="drawer" aria-expanded="false"><i data-ic="list" style="display:contents"></i><span>Notes &amp; reminders</span><b id="badge" hidden>0</b></button>
      </div>
    </header>
    <main id="chat"><div class="col thread" id="thread"></div></main>
    <form id="form" class="col composer">
      <button id="jump" class="jump" type="button" aria-label="Scroll to latest" data-ic="down" hidden></button>
      <div class="box">
        <textarea id="input" rows="1" placeholder="Message Ziva" aria-label="Message"></textarea>
        <button id="speak" class="ibtn" type="button" aria-pressed="false" title="Read replies aloud" data-ic="speaker"></button>
        <button id="mic" class="ibtn" type="button" aria-pressed="false" title="Speak to Ziva" data-ic="mic"></button>
        <button id="send" class="ibtn" type="submit" title="Send" aria-label="Send" data-ic="up" disabled></button>
      </div>
    </form>
  </div>
</div>
<div id="sscrim"></div>
<div id="scrim"></div>
<aside id="drawer" inert aria-label="Notes and reminders">
  <div class="dhead"><h2>Notes &amp; reminders</h2><button id="close" class="ibtn" type="button" aria-label="Close" data-ic="close"></button></div>
  <div class="tabs" role="tablist">
    <button class="tab" role="tab" data-tab="reminders" aria-selected="true">Reminders</button>
    <button class="tab" role="tab" data-tab="notes" aria-selected="false">Notes</button>
  </div>
  <ul id="list"></ul>
</aside>
<script>
const $ = s => document.querySelector(s);
const chat = $("#chat"), thread = $("#thread"), form = $("#form"), input = $("#input"), sendBtn = $("#send"), mic = $("#mic"), speakBtn = $("#speak");
const drawer = $("#drawer"), scrim = $("#scrim"), openBtn = $("#open"), badge = $("#badge"), list = $("#list");
const convsEl = $("#convs"), qEl = $("#q"), titleEl = $("#ttl"), exportBtn = $("#exportBtn"), jump = $("#jump");
const JSON_H = { "Content-Type": "application/json" };
const ICONS = {
  up: '<path d="M12 19V5M6 11l6-6 6 6"/>',
  down: '<path d="M12 5v14M6 13l6 6 6-6"/>',
  stop: '<rect x="7" y="7" width="10" height="10" rx="2.5" fill="currentColor"/>',
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
  speaker: '<path d="M11 5 6 9H3v6h3l5 4zM15.5 8.5a5 5 0 0 1 0 7"/>',
  list: '<path d="M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01"/>',
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>',
  check: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
  close: '<path d="M6 6l12 12M18 6 6 18"/>',
  trash: '<path d="M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>',
  panel: '<rect x="3.5" y="4.5" width="17" height="15" rx="3"/><path d="M9.5 4.5v15"/>',
  download: '<path d="M12 4v11M7 11l5 5 5-5M5 20h14"/>',
  edit: '<path d="M4 20h4L19 9l-4-4L4 16zM13.5 6.5l4 4"/>',
  retry: '<path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/>'
};
const ic = (name, size = 18) => `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;
const MARK = '<svg viewBox="0 0 48 48" aria-hidden="true"><path d="M24 3 45 24 24 45 3 24Z" fill="none" stroke="#E0714A" stroke-width="2.6" stroke-linecap="round" stroke-dasharray="0 1" pathLength="24"/><path d="M24 8 40 24 24 40 8 24Z" fill="#F0B34A"/><path d="M24 12.5 35.5 24 24 35.5 12.5 24Z" fill="#2A1D05"/><path d="M19.5 20h9l-9 8h9" fill="none" stroke="#F0B34A" stroke-width="2.6" stroke-linejoin="bevel"/></svg>';
document.querySelectorAll("[data-ic]").forEach(el => el.insertAdjacentHTML("afterbegin", ic(el.dataset.ic)));
document.querySelectorAll("[data-mark]").forEach(el => { el.innerHTML = MARK; });

const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
function inline(s) {
  return s.split(/(`[^`]*`)/).map((p, i) => i % 2 ? "<code>" + esc(p.slice(1, -1)) + "</code>" :
    esc(p).replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\s][^*]*)\*(?!\*)/g, "$1<em>$2</em>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')).join("");
}
const pre = c => '<pre><button type="button" class="cp">Copy</button><code>' + esc(c.join("\n")) + "</code></pre>";
function render(md) {
  let html = "", list = null, para = [], code = null;
  const flushPara = () => { if (para.length) { html += "<p>" + para.map(inline).join("<br>") + "</p>"; para = []; } };
  const flushList = () => { if (list) { html += "</" + list + ">"; list = null; } };
  for (const line of md.replace(/\r/g, "").split("\n")) {
    if (code !== null) {
      if (/^```/.test(line)) { html += pre(code); code = null; } else code.push(line);
      continue;
    }
    if (/^```/.test(line)) { flushPara(); flushList(); code = []; continue; }
    const li = line.match(/^\s*([-*+]|\d+\.)\s+(.*)$/);
    if (li) {
      flushPara();
      const tag = /\d/.test(li[1]) ? "ol" : "ul";
      if (list !== tag) { flushList(); html += "<" + tag + ">"; list = tag; }
      html += "<li>" + inline(li[2]) + "</li>";
      continue;
    }
    if (!line.trim()) { flushPara(); flushList(); continue; }
    const h = line.match(/^#{1,6}\s+(.*)$/);
    if (h) { flushPara(); flushList(); html += "<h4>" + inline(h[1]) + "</h4>"; continue; }
    flushList(); para.push(line);
  }
  if (code !== null) html += pre(code);
  flushPara(); flushList();
  return html;
}

let convId = null, msgs = [], busy = false, ctl = null, speakOn = false, thinkTimer = null;
try { speakOn = localStorage.getItem("ziva.speak") === "1"; } catch (e) {}
speakBtn.setAttribute("aria-pressed", speakOn);

const mobile = () => matchMedia("(max-width:900px)").matches;
const nearBottom = () => chat.scrollHeight - chat.scrollTop - chat.clientHeight < 120;
const toBottom = () => { chat.scrollTop = chat.scrollHeight; };
chat.addEventListener("scroll", () => { jump.hidden = nearBottom(); });
jump.onclick = () => chat.scrollTo({ top: chat.scrollHeight, behavior: "smooth" });
const slug = s => (s || "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "ziva";
function save(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/markdown;charset=utf-8" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const PHASES = [[0, "Thinking"], [6, "Still thinking"], [15, "Taking longer than usual"]];
function startThink(body) {
  stopThink();
  body.innerHTML = '<div class="think"><span class="dots"><i></i><i></i><i></i></span><span class="lbl">Thinking…</span></div>';
  const lbl = body.querySelector(".lbl"), t0 = Date.now();
  thinkTimer = setInterval(() => { lbl.textContent = PHASES.filter(p => (Date.now() - t0) / 1000 >= p[0]).pop()[1] + "…"; }, 1000);
}
function stopThink() { clearInterval(thinkTimer); thinkTimer = null; }

const SUGGESTIONS = ["Remind me to follow up on invoices on Monday", "Save a note: ideas for the new landing page", "What are my open reminders?", "Explain how DNS over HTTPS works"];
function showEmpty() {
  const h = new Date().getHours();
  const hello = h < 5 ? "Still up?" : h < 12 ? "Good morning." : h < 18 ? "Good afternoon." : "Good evening.";
  thread.innerHTML = `<section class="empty"><span class="mark">${MARK}</span><h2>${hello}</h2><p>Ask anything, save a note, or add a reminder.</p><div class="chips">${SUGGESTIONS.map(s => `<button type="button" class="chip">${s}</button>`).join("")}</div></section>`;
  thread.querySelectorAll(".chip").forEach(b => b.onclick = () => send(b.textContent));
}

function shell(role) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.innerHTML = role === "user"
    ? '<div class="stack"><div class="bubble"></div><div class="tools"></div></div>'
    : '<div class="main"><div class="body"></div><div class="tools"></div></div>';
  return el;
}
function paintBody(el, m) {
  el.querySelector(".body").innerHTML = m.text ? render(m.text) : m.stopped ? '<p class="hint">Stopped before a reply.</p>' : "";
}
function toolBtn(icon, title, fn) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "tool";
  b.title = title;
  b.setAttribute("aria-label", title);
  b.innerHTML = ic(icon, 15);
  b.onclick = () => fn(b);
  return b;
}
const copyBtn = get => toolBtn("copy", "Copy", async b => {
  try { await navigator.clipboard.writeText(get()); b.innerHTML = ic("check", 15); setTimeout(() => { b.innerHTML = ic("copy", 15); }, 1500); } catch (e) {}
});
function fillTools(el, m, i) {
  const t = el.querySelector(".tools");
  t.innerHTML = "";
  if (m.pending) return;
  if (m.role === "user") {
    if (m.id) t.append(toolBtn("edit", "Edit message", () => editMsg(i, el)));
    t.append(copyBtn(() => m.text));
    return;
  }
  if (m.stopped && m.text) { const s = document.createElement("span"); s.className = "tag"; s.textContent = "Stopped"; t.append(s); }
  if (m.text) t.append(copyBtn(() => m.text), toolBtn("download", "Download reply", () => save(`ziva-reply-${slug(titleEl.textContent)}.md`, m.text)));
  if (i === msgs.length - 1 && msgs[i - 1]) t.append(toolBtn("retry", "Try again", () => send(msgs[i - 1].text, null, true)));
}
function msgEl(m, i) {
  const el = shell(m.role);
  if (m.role === "user") el.querySelector(".bubble").textContent = m.text;
  else {
    if (m.error) el.classList.add("error");
    if (m.pending) startThink(el.querySelector(".body")); else paintBody(el, m);
  }
  fillTools(el, m, i);
  return el;
}
function fillAll() {
  msgs.forEach((m, i) => { const el = thread.children[i]; if (el) fillTools(el, m, i); });
  exportBtn.disabled = !msgs.length;
}
function renderAll() {
  stopThink();
  thread.innerHTML = "";
  if (!msgs.length) showEmpty();
  else { msgs.forEach((m, i) => thread.append(msgEl(m, i))); toBottom(); }
  exportBtn.disabled = !msgs.length;
}
function say(text, err) {
  const el = shell("assistant");
  if (err) el.classList.add("error");
  el.querySelector(".body").innerHTML = render(text);
  thread.append(el);
  toBottom();
}
function speak(text) {
  if (!speakOn || !("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text.replace(/[*_`#>]/g, "").replace(/\s+/g, " "));
  u.lang = "en-US";
  speechSynthesis.speak(u);
}
thread.addEventListener("click", async e => {
  const b = e.target.closest(".cp");
  if (!b) return;
  try { await navigator.clipboard.writeText(b.parentElement.querySelector("code").textContent); b.textContent = "Copied"; setTimeout(() => { b.textContent = "Copy"; }, 1500); } catch (err) {}
});

function editMsg(i, el) {
  const m = msgs[i];
  if (busy || !m.id) return;
  const stack = el.querySelector(".stack");
  stack.classList.add("editing");
  stack.innerHTML = "";
  const ta = document.createElement("textarea");
  ta.className = "edit";
  ta.rows = 1;
  ta.value = m.text;
  const fit = () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 240) + "px"; };
  const bar = document.createElement("div");
  bar.className = "editbar";
  const note = document.createElement("span");
  note.textContent = "Later replies will be replaced.";
  const cancel = document.createElement("button");
  cancel.type = "button"; cancel.className = "btn ghost"; cancel.textContent = "Cancel";
  const ok = document.createElement("button");
  ok.type = "button"; ok.className = "btn"; ok.textContent = "Save & send";
  const restore = () => {
    stack.classList.remove("editing");
    stack.innerHTML = '<div class="bubble"></div><div class="tools"></div>';
    stack.querySelector(".bubble").textContent = m.text;
    fillTools(el, m, i);
  };
  cancel.onclick = restore;
  ok.onclick = () => send(ta.value, m.id);
  ta.oninput = fit;
  ta.onkeydown = e => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ok.click(); }
    else if (e.key === "Escape") { e.stopPropagation(); restore(); }
  };
  bar.append(note, cancel, ok);
  stack.append(ta, bar);
  fit();
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);
}

function grow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
  sendBtn.disabled = !busy && !input.value.trim();
}
function setBusy(on) {
  busy = on;
  sendBtn.innerHTML = ic(on ? "stop" : "up");
  sendBtn.classList.toggle("stop", on);
  const label = on ? "Stop generating" : "Send";
  sendBtn.title = label;
  sendBtn.setAttribute("aria-label", label);
  grow();
}
function stop() {
  if (ctl) ctl.abort();
  if ("speechSynthesis" in window) speechSynthesis.cancel();
}
async function fixIds(id, um) {
  try {
    const d = await (await fetch(`/api/conversations/${id}`)).json();
    const u = [...d.messages].reverse().find(m => m.role === "user");
    if (u) um.id = u.id;
  } catch (e) {}
}
async function send(text, replaceFrom, retry) {
  text = (text || "").trim();
  if (busy || !text) return;
  const fresh = !replaceFrom && !retry;
  setBusy(true);
  if (!convId) {
    try { convId = (await (await fetch("/api/conversations", { method: "POST" })).json()).id; rememberConv(); }
    catch (e) { setBusy(false); say("Couldn’t reach Ziva. Check that the server is still running, then try again.", true); return; }
  }
  const my = convId;
  if (fresh) { input.value = ""; grow(); }
  let at = -1;
  if (replaceFrom) at = msgs.findIndex(m => m.id === replaceFrom);
  else if (retry) at = msgs.map(m => m.role).lastIndexOf("user");
  if (at >= 0) msgs.length = at;
  const um = { id: null, role: "user", text }, am = { id: null, role: "assistant", text: "", pending: true };
  msgs.push(um, am);
  renderAll();
  const el = thread.lastElementChild, body = el.querySelector(".body");
  ctl = new AbortController();
  let full = "", failed = false, aborted = false;
  try {
    const res = await fetch("/chat", { method: "POST", signal: ctl.signal, headers: JSON_H, body: JSON.stringify({ message: text, conversation_id: my, replace_from: replaceFrom || null, retry: !!retry }) });
    if (!res.ok) throw new Error("Server returned " + res.status);
    loadConvs();
    if ((res.headers.get("content-type") || "").includes("application/json")) {
      const d = await res.json();
      full = d.reply || "";
      failed = !!d.error;
      um.id = d.user_id; am.id = d.assistant_id;
      stopThink();
      body.innerHTML = render(full);
      if (d.changed) loadItems();
    } else {
      um.id = +res.headers.get("X-User-Msg"); am.id = +res.headers.get("X-Asst-Msg");
      const reader = res.body.getReader(), dec = new TextDecoder();
      let started = false;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!started) { started = true; stopThink(); el.classList.add("streaming"); }
        full += dec.decode(value, { stream: true });
        const stick = nearBottom();
        body.innerHTML = render(full);
        if (stick) toBottom();
      }
      full += dec.decode();
    }
  } catch (err) {
    if (err.name === "AbortError") aborted = true;
    else { failed = true; full = "Couldn’t reach Ziva. Check that the server is still running, then try again."; }
  }
  stopThink();
  ctl = null;
  if (convId !== my) { setBusy(false); return; }
  Object.assign(am, { text: full, pending: false, error: failed, stopped: aborted });
  el.classList.remove("streaming");
  el.classList.toggle("error", failed);
  paintBody(el, am);
  if (!um.id) await fixIds(my, um);
  if (convId !== my) { setBusy(false); return; }
  if (!failed && !aborted) speak(full);
  setBusy(false);
  fillAll();
  loadConvs();
  input.focus();
}

input.addEventListener("input", grow);
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!busy) form.requestSubmit(); }
});
form.addEventListener("submit", e => { e.preventDefault(); if (busy) stop(); else send(input.value); });
speakBtn.onclick = () => {
  speakOn = !speakOn;
  speakBtn.setAttribute("aria-pressed", speakOn);
  if (!speakOn && "speechSynthesis" in window) speechSynthesis.cancel();
  try { localStorage.setItem("ziva.speak", speakOn ? "1" : "0"); } catch (e) {}
};
exportBtn.onclick = () => {
  const t = titleEl.textContent;
  const md = `# ${t}\n\n` + msgs.filter(m => m.text && !m.error).map(m => `**${m.role === "user" ? "You" : "Ziva"}**\n\n${m.text}`).join("\n\n---\n\n") + "\n";
  save(`ziva-${slug(t)}.md`, md);
};

const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, listening = false;
function setListening(on) {
  listening = on;
  mic.classList.toggle("on", on);
  mic.setAttribute("aria-pressed", on);
  input.placeholder = on ? "Listening…" : "Message Ziva";
}
if (SR) {
  rec = new SR();
  rec.lang = "en-US"; rec.interimResults = true; rec.continuous = false;
  rec.onresult = e => {
    let fin = "", interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) { if (e.results[i].isFinal) fin += e.results[i][0].transcript; else interim += e.results[i][0].transcript; }
    input.value = fin || interim;
    grow();
    if (fin) send(fin);
  };
  rec.onend = () => setListening(false);
  rec.onerror = e => { setListening(false); if (!["no-speech", "aborted"].includes(e.error)) say("Microphone problem: " + e.error, true); };
  mic.onclick = () => {
    if (listening) { rec.stop(); return; }
    if ("speechSynthesis" in window) speechSynthesis.cancel();
    try { rec.start(); setListening(true); } catch (e) { setListening(false); }
  };
} else { mic.disabled = true; mic.title = "Voice input needs Chrome or Edge"; }

const sideOpen = () => mobile() ? document.body.classList.contains("sideopen") : !document.body.classList.contains("noside");
function setSide(on) {
  if (mobile()) document.body.classList.toggle("sideopen", on);
  else {
    document.body.classList.toggle("noside", !on);
    try { localStorage.setItem("ziva.side", on ? "1" : "0"); } catch (e) {}
  }
}
try { if (!mobile() && localStorage.getItem("ziva.side") === "0") document.body.classList.add("noside"); } catch (e) {}
$("#toggleSide").onclick = () => setSide(!sideOpen());
$("#sideClose").onclick = $("#sscrim").onclick = () => setSide(false);

const rememberConv = () => { try { convId ? localStorage.setItem("ziva.conv", convId) : localStorage.removeItem("ziva.conv"); } catch (e) {} };
function groupOf(iso) {
  const n = new Date(), day = 864e5, t = new Date(iso).getTime();
  const start = new Date(n.getFullYear(), n.getMonth(), n.getDate()).getTime();
  return t >= start ? "Today" : t >= start - day ? "Yesterday" : t >= start - 7 * day ? "Previous 7 days" : "Older";
}
function convRow(c) {
  const row = document.createElement("div");
  row.className = "conv" + (c.id === convId ? " active" : "");
  const a = document.createElement("button");
  a.type = "button"; a.className = "ctitle"; a.textContent = c.title; a.title = c.title;
  a.onclick = () => openConv(c.id);
  const acts = document.createElement("div");
  acts.className = "cacts";
  const ren = document.createElement("button");
  ren.type = "button"; ren.title = "Rename"; ren.setAttribute("aria-label", "Rename chat"); ren.innerHTML = ic("edit", 15);
  const del = document.createElement("button");
  del.type = "button"; del.title = "Delete"; del.setAttribute("aria-label", "Delete chat"); del.innerHTML = ic("trash", 15);
  ren.onclick = () => {
    const inp = document.createElement("input");
    inp.className = "rename"; inp.value = c.title; inp.maxLength = 80;
    row.replaceChild(inp, a);
    acts.remove();
    inp.focus(); inp.select();
    let done = false;
    const finish = async ok => {
      if (done) return;
      done = true;
      const t = inp.value.trim();
      if (ok && t && t !== c.title) {
        try { await fetch(`/api/conversations/${c.id}`, { method: "PATCH", headers: JSON_H, body: JSON.stringify({ title: t }) }); } catch (e) {}
        if (c.id === convId) titleEl.textContent = t;
      }
      loadConvs();
    };
    inp.onkeydown = e => { if (e.key === "Enter") finish(true); else if (e.key === "Escape") finish(false); };
    inp.onblur = () => finish(true);
  };
  del.onclick = async () => {
    if (!del.classList.contains("arm")) {
      del.classList.add("arm"); del.title = "Click again to delete";
      setTimeout(() => { del.classList.remove("arm"); del.title = "Delete"; }, 2500);
      return;
    }
    try { await fetch(`/api/conversations/${c.id}`, { method: "DELETE" }); } catch (e) {}
    if (c.id === convId) newChat();
    loadConvs();
  };
  acts.append(ren, del);
  row.append(a, acts);
  return row;
}
function drawConvs(rows) {
  convsEl.innerHTML = "";
  if (!rows.length) { convsEl.innerHTML = `<p class="none">${qEl.value.trim() ? "No chats match." : "Your chats will appear here."}</p>`; return; }
  let last = "";
  for (const c of rows) {
    const g = groupOf(c.updated_at);
    if (g !== last) { const h = document.createElement("h3"); h.textContent = g; convsEl.append(h); last = g; }
    convsEl.append(convRow(c));
  }
  const cur = rows.find(r => r.id === convId);
  if (cur) titleEl.textContent = cur.title;
}
async function loadConvs() {
  try { drawConvs(await (await fetch("/api/conversations?q=" + encodeURIComponent(qEl.value.trim()))).json()); } catch (e) {}
}
let qTimer = null;
qEl.addEventListener("input", () => { clearTimeout(qTimer); qTimer = setTimeout(loadConvs, 180); });

function newChat() {
  if (busy) stop();
  convId = null; msgs = [];
  titleEl.textContent = "New chat";
  rememberConv();
  renderAll();
  loadConvs();
  if (mobile()) setSide(false);
  input.focus();
}
async function openConv(id) {
  if (busy) stop();
  let d;
  try {
    const r = await fetch(`/api/conversations/${id}`);
    if (!r.ok) throw new Error();
    d = await r.json();
  } catch (e) { return false; }
  convId = d.id;
  msgs = d.messages.map(m => ({ ...m, error: !!m.error, stopped: !!m.stopped }));
  if (msgs.length && msgs[msgs.length - 1].role === "user") msgs.push({ id: null, role: "assistant", text: "", stopped: true });
  titleEl.textContent = d.title;
  rememberConv();
  renderAll();
  loadConvs();
  if (mobile()) setSide(false);
  return true;
}
$("#newChat").onclick = newChat;

let items = { notes: [], reminders: [] }, tab = "reminders";
const fmtDate = iso => new Date(iso).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
async function act(method, url) { try { await fetch(url, { method }); } catch (e) {} loadItems(); }
async function loadItems() {
  try { items = await (await fetch("/api/items")).json(); } catch (e) {}
  drawList();
}
function drawList() {
  const open = items.reminders.filter(r => !r.completed).length;
  badge.textContent = open;
  badge.hidden = !open;
  list.innerHTML = "";
  const rows = items[tab];
  if (!rows.length) {
    list.innerHTML = `<li class="none">${tab === "reminders" ? "No reminders yet. Tell Ziva “remind me to…” and it lands here." : "No notes yet. Say “note that…” and Ziva will keep it."}</li>`;
    return;
  }
  for (const r of rows) {
    const li = document.createElement("li");
    li.className = "item" + (r.completed ? " done" : "");
    if (tab === "reminders") {
      const c = document.createElement("button");
      c.className = "check";
      c.setAttribute("aria-label", r.completed ? "Mark as open" : "Mark as done");
      c.innerHTML = ic("check", 13);
      c.onclick = () => act("POST", `/api/reminders/${r.id}/toggle`);
      li.append(c);
    }
    const t = document.createElement("div");
    t.className = "t";
    const p = document.createElement("div");
    p.textContent = r.text;
    const d = document.createElement("small");
    d.textContent = fmtDate(r.created_at);
    t.append(p, d);
    const x = document.createElement("button");
    x.className = "del";
    x.setAttribute("aria-label", "Delete");
    x.innerHTML = ic("trash", 16);
    x.onclick = () => act("DELETE", `/api/${tab}/${r.id}`);
    li.append(t, x);
    list.append(li);
  }
}
document.querySelectorAll(".tab").forEach(b => b.onclick = () => {
  tab = b.dataset.tab;
  document.querySelectorAll(".tab").forEach(o => o.setAttribute("aria-selected", o === b));
  drawList();
});
function setDrawer(on) {
  drawer.classList.toggle("open", on);
  scrim.classList.toggle("open", on);
  drawer.inert = !on;
  openBtn.setAttribute("aria-expanded", on);
  if (on) { loadItems(); $("#close").focus(); } else openBtn.focus();
}
openBtn.onclick = () => setDrawer(true);
$("#close").onclick = scrim.onclick = () => setDrawer(false);
document.addEventListener("keydown", e => {
  const mod = e.ctrlKey || e.metaKey, k = e.key.toLowerCase();
  if (e.key === "Escape") {
    if (drawer.classList.contains("open")) setDrawer(false);
    else if (mobile() && sideOpen()) setSide(false);
    else if (busy) stop();
  } else if (mod && e.shiftKey && k === "o") { e.preventDefault(); newChat(); }
  else if (mod && !e.shiftKey && k === "k") { e.preventDefault(); if (!sideOpen()) setSide(true); qEl.focus(); }
});

fetch("/api/config").then(r => r.json()).then(c => {
  if (!c.demo) return;
  const f = $("#sfoot");
  f.textContent = `Public demo. Your chats stay private to this browser, are deleted after ${c.ttl} hours, and are limited to ${c.hourly} messages an hour.`;
  f.hidden = false;
}).catch(() => {});
renderAll();
loadItems();
loadConvs();
(async () => {
  let saved = null;
  try { saved = localStorage.getItem("ziva.conv"); } catch (e) {}
  if (saved && !(await openConv(+saved))) { convId = null; rememberConv(); }
  input.focus();
})();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
