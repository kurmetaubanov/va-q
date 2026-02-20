"""
Scheduling Agent — Groq + function calling
pip install groq fastapi uvicorn
GROQ_API_KEY=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 uvicorn app.main:app --reload
"""
"""
Voice Scheduling Agent
- STT: Deepgram Streaming
- LLM: Groq (llama-3.3-70b-versatile) + function calling
- TTS: ElevenLabs mp3

pip install groq fastapi uvicorn websockets icalendar httpx
GROQ_API_KEY=...
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...
uvicorn main:app --reload
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import httpx
import websockets
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from groq import Groq
from starlette.websockets import WebSocketDisconnect, WebSocketState

from calendar_export import save_ics

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

GROQ_API_KEY       = os.environ["GROQ_API_KEY"]
DEEPGRAM_API_KEY   = os.environ["DEEPGRAM_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE   = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # default: George

MODEL = "llama-3.3-70b-versatile"

groq_client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────
# State
# ─────────────────────────────────────────

def empty_state():
    return {
        "user_name": None,
        "date": None,
        "time": None,
        "title": None,
        "schedule_finalized": False,
    }

app_state = {
    "history": [],
    "state": empty_state(),
    "ics_path": None,
    "iso_datetime": None,
}

def is_finalized(state):
    return bool(
        state.get("schedule_finalized")
        and state.get("user_name")
        and state.get("date")
        and state.get("time")
    )

def merge_state(old, updates):
    if not updates:
        return old
    new = old.copy()
    for k in new:
        if k in updates and updates[k] is not None:
            new[k] = updates[k]
    return new

# ─────────────────────────────────────────
# Tools
# ─────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "flag_secondary_intent",
            "description": (
                "Call this when the user's message contains a secondary intent "
                "that is not necessary for completing the scheduling task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent_description": {
                        "type": "string",
                        "description": "Short description of the off-topic request"
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["soft", "hard"],
                        "description": (
                            "soft = minor tangent, user still focused on scheduling. "
                            "hard = user clearly wants something unrelated."
                        )
                    }
                },
                "required": ["intent_description", "severity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_schedule",
            "description": (
                "Extract and update scheduling information from the conversation. "
                "Call this tool on every user message to keep state up to date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_name": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": "User's name. Only set if the user clearly stated it."
                    },
                    "date": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": "Single specific date. Null if multiple or ambiguous."
                    },
                    "time": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": "Single specific time. Null if multiple or ambiguous."
                    },
                    "title": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": "Optional meeting title."
                    },
                    "schedule_finalized": {
                        "type": "boolean",
                        "description": (
                            "Set to true ONLY when user_name, date, and time are all known "
                            "AND the user has explicitly confirmed the booking."
                        )
                    }
                },
                "required": ["schedule_finalized"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_datetime",
            "description": (
                "Call this when schedule_finalized becomes true. "
                "Convert the natural language date and time from the schedule "
                "into an ISO 8601 datetime string, using today's date as reference."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "iso_datetime": {
                        "type": "string",
                        "description": "Resolved datetime in ISO 8601 format, e.g. '2026-02-21T17:00:00'"
                    }
                },
                "required": ["iso_datetime"]
            }
        }
    }
]

# ─────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────

def build_system(state, secondary_intent=None):
    today = datetime.now().strftime("%A, %Y-%m-%d")

    drift_instruction = ""
    if secondary_intent:
        desc = secondary_intent.get("intent_description", "")
        severity = secondary_intent.get("severity", "soft")
        if severity == "hard":
            drift_instruction = f"""
IMPORTANT: The user just asked something off-topic: "{desc}".
Do NOT answer it. Briefly say you can only help with scheduling here,
then redirect back to completing the booking.
"""
        else:
            drift_instruction = f"""
NOTE: The user's message contained a side request: "{desc}".
Acknowledge it very briefly (one short clause), then refocus on scheduling.
Do not provide the off-topic content.
"""

    return f"""You are a friendly voice scheduling assistant.

Today is {today}.

Current schedule state:
{json.dumps(state, indent=2)}

Rules:
- Greet the user only once at the very start of the conversation.
- Collect: user_name, date, time, and optionally a meeting title.
- Schedule only ONE meeting at a time.
- If the user mentions multiple dates or times, ask them to choose ONE. Never assume.
- Finalize only after the user explicitly confirms all details.
- Always call update_schedule to reflect any new information extracted.
- If off-topic content is detected, call flag_secondary_intent and do NOT answer it.
- When all info is collected and confirmed, set schedule_finalized=true and respond
  with ONE short closing sentence. Do NOT ask if there is anything else you can help with.
  Do NOT invite further conversation.
- Keep responses short and conversational.
{drift_instruction}"""

# ─────────────────────────────────────────
# LLM dialogue step
# ─────────────────────────────────────────

def dialogue_step(history, state, user_msg):
    """
    Single LLM call with two tools:
      - flag_secondary_intent  → detected off-topic content
      - update_schedule        → extract scheduling fields

    Both can be called in the same turn.
    Returns (reply_text, new_state, should_reset).
    """

    messages = history + [{"role": "user", "content": user_msg}]

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": build_system(state)}] + messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.5,
        max_tokens=512,
    )

    message = response.choices[0].message
    reply_text = message.content or ""
    new_state = state
    secondary_intent = None

    if message.tool_calls:
        for tc in message.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                continue

            if name == "update_schedule":
                new_state = merge_state(state, args)
                print(f"[TOOL] update_schedule: {args}")
                print(f"[STATE] {new_state}")

            elif name == "flag_secondary_intent":
                secondary_intent = args
                print(f"[TOOL] flag_secondary_intent: {args}")

            elif name == "resolve_datetime":
                app_state["iso_datetime"] = args.get("iso_datetime")
                print(f"[TOOL] resolve_datetime: {args}")

    # ── If model returned tool calls but no text → request reply ──
    # secondary_intent is injected into system prompt so model knows to block the drift
    if message.tool_calls and not reply_text:
        tool_results = [
            {"role": "tool", "tool_call_id": tc.id, "content": "ok"}
            for tc in message.tool_calls
        ]
        followup = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": build_system(new_state, secondary_intent)},
                *messages,
                {"role": "assistant", "content": None, "tool_calls": message.tool_calls},
                *tool_results,
            ],
            temperature=0.5,
            max_tokens=256,
        )
        reply_text = followup.choices[0].message.content or "Got it!"

    return reply_text, new_state, is_finalized(new_state)

# ─────────────────────────────────────────
# ElevenLabs TTS
# ─────────────────────────────────────────

async def text_to_speech(text: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8},
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.content  # mp3 bytes

# ─────────────────────────────────────────
# Initiator
# ─────────────────────────────────────────

def initiate_text() -> str:
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly scheduling assistant. "
                    "Start the conversation. Ask for the user's name, "
                    "preferred date and time, and optionally a meeting title. "
                    "Be brief and natural. One or two sentences."
                )
            },
            {"role": "user", "content": "__start__"}
        ],
        temperature=0.7,
        max_tokens=128,
    )
    return response.choices[0].message.content.strip()

# ─────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────

app = FastAPI()

@app.get("/download-ics")
def download_ics():
    path = app_state.get("ics_path")
    if not path or not Path(path).exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="No .ics file available yet")
    return FileResponse(path, media_type="text/calendar", filename=Path(path).name)

@app.get("/debug")
def debug():
    return {k: v for k, v in app_state.items() if k != "history"}

# ─────────────────────────────────────────
# Main WebSocket — browser ↔ server
# ─────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    async def send_text(msg: str):
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_text(msg)

    async def send_audio(data: bytes):
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_bytes(data)

    async def speak(text: str):
        """TTS → send mp3 to browser."""
        try:
            audio = await text_to_speech(text)
            await send_audio(audio)
        except Exception as e:
            print(f"[TTS ERROR] {e}")

    async def handle_transcript(transcript: str):
        """Process final transcript through LLM pipeline."""
        if not transcript.strip():
            return

        print(f"[TRANSCRIPT] {transcript}")
        await send_text(f"__user__{transcript}")  # echo to UI

        history = app_state["history"]
        state = app_state["state"]

        reply, new_state, should_reset = dialogue_step(history, state, transcript)
        print(f"[REPLY] {reply}")

        new_history = history + [
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": reply},
        ]

        if should_reset:
            try:
                iso_dt = app_state.get("iso_datetime")
                if not iso_dt:
                    raise ValueError("iso_datetime not resolved")
                ics_path = save_ics(new_state, iso_dt)
                app_state["ics_path"] = str(ics_path)
                print(f"[ICS] saved to {ics_path}")
                await send_text("__ics_ready__")
            except Exception as e:
                print(f"[ICS ERROR] {e}")

            app_state["history"] = []
            app_state["state"] = empty_state()
            app_state["iso_datetime"] = None
            await speak(reply)
            await send_text("__show_start__")
        else:
            app_state["history"] = new_history
            app_state["state"] = new_state
            await speak(reply)

    try:
        while True:
            msg = await ws.receive()

            # ── Text control messages ──
            if "text" in msg:
                text = msg["text"]

                if text == "__start__":
                    app_state["history"] = []
                    app_state["state"] = empty_state()
                    app_state["iso_datetime"] = None
                    greeting = initiate_text()
                    app_state["history"] = [{"role": "assistant", "content": greeting}]
                    await send_text(f"__agent__{greeting}")
                    await speak(greeting)

                else:
                    # typed text fallback (for testing without mic)
                    await handle_transcript(text)

            # ── Binary audio from browser → stream to Deepgram ──
            elif "bytes" in msg:
                audio_chunk = msg["bytes"]
                # forward to Deepgram via shared queue
                await audio_queue.put(audio_chunk)

    except WebSocketDisconnect:
        print("[WS] Client disconnected")

    except Exception as e:
        print(f"[WS ERROR] {e}")

# ─────────────────────────────────────────
# Deepgram streaming task (per connection)
# ─────────────────────────────────────────

audio_queue: asyncio.Queue = asyncio.Queue()

DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?encoding=webm-opus"
    "&sample_rate=48000"
    "&channels=1"
    "&model=nova-2"
    "&language=en-US"
    "&punctuate=true"
    "&interim_results=true"
    "&endpointing=400"
)

@app.on_event("startup")
async def start_deepgram_task():
    asyncio.create_task(deepgram_streaming_task())

async def deepgram_streaming_task():
    """
    Persistent connection to Deepgram.
    Reads audio from audio_queue, sends to Deepgram,
    receives transcripts and processes them.
    """
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

    while True:
        try:
            async with websockets.connect(DEEPGRAM_URL, extra_headers=headers) as dg_ws:
                print("[DEEPGRAM] connected")

                async def sender():
                    while True:
                        chunk = await audio_queue.get()
                        if dg_ws.open:
                            await dg_ws.send(chunk)

                async def receiver():
                    async for raw in dg_ws:
                        try:
                            data = json.loads(raw)
                            alt = (
                                data.get("channel", {})
                                    .get("alternatives", [{}])[0]
                            )
                            transcript = alt.get("transcript", "")
                            is_final = data.get("is_final", False)

                            if transcript and is_final:
                                # Run in main event loop — we need access to ws
                                # We'll use a second queue to pass transcripts back
                                await transcript_queue.put(transcript)
                        except Exception as e:
                            print(f"[DEEPGRAM PARSE ERROR] {e}")

                await asyncio.gather(sender(), receiver())

        except Exception as e:
            print(f"[DEEPGRAM ERROR] {e} — reconnecting in 2s")
            await asyncio.sleep(2)

transcript_queue: asyncio.Queue = asyncio.Queue()

@app.on_event("startup")
async def start_transcript_processor():
    asyncio.create_task(transcript_processor())

async def transcript_processor():
    """
    Reads from transcript_queue and processes through LLM.
    Needs access to active WebSocket — stored in active_ws.
    """
    while True:
        transcript = await transcript_queue.get()
        ws = active_ws.get("ws")
        if ws and ws.client_state == WebSocketState.CONNECTED:
            # Re-use handle logic via a small shim
            await _process_transcript_for_ws(ws, transcript)

active_ws: dict = {}

async def _process_transcript_for_ws(ws: WebSocket, transcript: str):
    async def send_text(msg):
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_text(msg)

    async def speak(text):
        try:
            audio = await text_to_speech(text)
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_bytes(audio)
        except Exception as e:
            print(f"[TTS ERROR] {e}")

    print(f"[TRANSCRIPT] {transcript}")
    await send_text(f"__user__{transcript}")

    history = app_state["history"]
    state = app_state["state"]
    reply, new_state, should_reset = dialogue_step(history, state, transcript)
    print(f"[REPLY] {reply}")

    new_history = history + [
        {"role": "user", "content": transcript},
        {"role": "assistant", "content": reply},
    ]

    if should_reset:
        try:
            iso_dt = app_state.get("iso_datetime")
            if not iso_dt:
                raise ValueError("iso_datetime not resolved")
            ics_path = save_ics(new_state, iso_dt)
            app_state["ics_path"] = str(ics_path)
            await send_text("__ics_ready__")
        except Exception as e:
            print(f"[ICS ERROR] {e}")
        app_state["history"] = []
        app_state["state"] = empty_state()
        app_state["iso_datetime"] = None
        await speak(reply)
        await send_text("__show_start__")
    else:
        app_state["history"] = new_history
        app_state["state"] = new_state
        await speak(reply)

# Register active ws on connect
_original_ws = ws_endpoint.__wrapped__ if hasattr(ws_endpoint, "__wrapped__") else None

# ─────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────

INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Voice Scheduling Agent</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: Arial, sans-serif; background: #f0f2f5; display: flex;
         justify-content: center; padding: 40px 16px; }
  .container { width: 100%; max-width: 560px; }
  h2 { text-align: center; margin-bottom: 20px; color: #222; }
  #chat { background: #fff; border: 1px solid #ddd; border-radius: 10px;
          padding: 16px; height: 360px; overflow-y: auto; margin-bottom: 14px; }
  .user  { color: #222; margin: 6px 0; }
  .agent { color: #0b5cff; margin: 6px 0; }
  .system { color: #999; font-size: 12px; margin: 4px 0; }
  .controls { display: flex; gap: 10px; margin-bottom: 10px; }
  #startBtn { flex: 1; padding: 10px; background: #28a745; color: #fff;
              border: none; border-radius: 6px; cursor: pointer; font-size: 15px; }
  #micBtn { flex: 1; padding: 10px; background: #0b5cff; color: #fff;
            border: none; border-radius: 6px; cursor: pointer; font-size: 15px; }
  #micBtn.recording { background: #dc3545; }
  .text-row { display: flex; gap: 8px; }
  #msg { flex: 1; padding: 9px; border-radius: 6px; border: 1px solid #ccc; }
  #sendBtn { padding: 9px 16px; background: #555; color: #fff;
             border: none; border-radius: 6px; cursor: pointer; }
</style>
</head>
<body>
<div class="container">
  <h2>&#128197; Voice Scheduling Agent</h2>
  <div id="chat"></div>
  <div class="controls">
    <button id="startBtn" onclick="startChat()">START</button>
    <button id="micBtn" onclick="toggleMic()" disabled>&#127908; Hold to speak</button>
  </div>
  <div class="text-row">
    <input id="msg" placeholder="Or type a message..." />
    <button id="sendBtn" onclick="sendText()">Send</button>
  </div>
</div>

<script>
const chat = document.getElementById("chat");
const micBtn = document.getElementById("micBtn");
const startBtn = document.getElementById("startBtn");

let ws, mediaRecorder, audioCtx, isRecording = false;

function addMsg(text, cls) {
  const d = document.createElement("div");
  d.className = cls;
  d.textContent = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    addMsg("connected", "system");
    startBtn.style.display = "inline-block";
  };

  ws.onmessage = async ({ data }) => {
    if (typeof data === "string") {
      if (data === "__show_start__") {
        startBtn.style.display = "inline-block";
        micBtn.disabled = true;
        addMsg("--- conversation finished ---", "system");
        return;
      }
      if (data === "__ics_ready__") {
        const d = document.createElement("div");
        d.className = "system";
        d.innerHTML = '&#128197; <a href="/download-ics" download>Download calendar event (.ics)</a>';
        chat.appendChild(d);
        chat.scrollTop = chat.scrollHeight;
        return;
      }
      if (data.startsWith("__agent__")) {
        addMsg("Agent: " + data.slice(9), "agent");
        return;
      }
      if (data.startsWith("__user__")) {
        addMsg("You: " + data.slice(8), "user");
        return;
      }
    }

    // Binary mp3 audio → play
    if (data instanceof ArrayBuffer) {
      const blob = new Blob([data], { type: "audio/mpeg" });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.play().catch(e => console.warn("Audio play failed:", e));
    }
  };

  ws.onerror = () => addMsg("connection error", "system");
  ws.onclose = () => addMsg("disconnected", "system");
}

async function startChat() {
  startBtn.style.display = "none";
  ws.send("__start__");
  micBtn.disabled = false;
}

// ── Microphone recording ──
async function toggleMic() {
  if (!isRecording) {
    await startRecording();
  } else {
    stopRecording();
  }
}

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
      ws.send(e.data);
    }
  };

  mediaRecorder.start(250); // send chunks every 250ms
  isRecording = true;
  micBtn.textContent = "🔴 Recording... (click to stop)";
  micBtn.classList.add("recording");
}

function stopRecording() {
  if (mediaRecorder) {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
  }
  isRecording = false;
  micBtn.textContent = "🎤 Hold to speak";
  micBtn.classList.remove("recording");
}

function sendText() {
  const msg = document.getElementById("msg").value.trim();
  if (!msg || !ws || ws.readyState !== WebSocket.OPEN) return;
  addMsg("You: " + msg, "user");
  ws.send(msg);
  document.getElementById("msg").value = "";
}

document.getElementById("msg")
  .addEventListener("keypress", e => { if (e.key === "Enter") sendText(); });

connectWS();
</script>
</body>
</html>"""

@app.get("/")
def index():
    return HTMLResponse(INDEX_HTML)