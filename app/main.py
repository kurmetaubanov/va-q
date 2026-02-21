"""
Voice Scheduling Agent
STT: Deepgram Flux  |  LLM: Groq  |  TTS: ElevenLabs

pip install groq fastapi uvicorn websockets icalendar httpx python-dotenv deepgram-sdk
uvicorn app.main:app --reload
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from groq import Groq
from starlette.websockets import WebSocketDisconnect, WebSocketState

load_dotenv()
from .calendar_export import save_ics

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

GROQ_API_KEY       = os.environ["GROQ_API_KEY"]
DEEPGRAM_API_KEY   = os.environ["DEEPGRAM_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE   = os.environ["ELEVENLABS_VOICE_ID"]
MODEL              = "llama-3.3-70b-versatile"

groq_client = Groq(api_key=GROQ_API_KEY)

# Flux v2 endpoint — eot_threshold=0.7 (default, explicit for clarity)
# StartOfTurn fires when user starts speaking → barge-in
# EndOfTurn fires when Flux is ≥70% confident user finished → trigger LLM
DEEPGRAM_FLUX_URL = (
    "wss://api.deepgram.com/v2/listen"
    "?model=flux-general-en"
    "&encoding=linear16"
    "&sample_rate=16000"
    "&eot_threshold=0.7"
    "&eot_timeout_ms=4000"
)

# ─────────────────────────────────────────────────────────────
# Scheduling state
# ─────────────────────────────────────────────────────────────

def empty_state() -> dict:
    return {"user_name": None, "date": None, "time": None,
            "title": None, "schedule_finalized": False}

def merge_state(old: dict, updates: dict) -> dict:
    new = old.copy()
    for k in new:
        if updates.get(k) is not None:
            new[k] = updates[k]
    return new

def is_finalized(state: dict) -> bool:
    return bool(
        state.get("schedule_finalized")
        and state.get("user_name")
        and state.get("date")
        and state.get("time")
    )


# ─────────────────────────────────────────────────────────────
# Groq tools (unchanged from va-q)
# ─────────────────────────────────────────────────────────────

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
            "name": "validate_datetime",
            "description": (
                "Call this every time date or time is updated — including when the user changes "
                "a previously given date or time. Do not skip if validate_datetime was already "
                "called earlier in the conversation. "
                "Set is_past=true if: "
                "(1) the date is clearly before today, OR "
                "(2) the date is today AND the time has already passed. "
                "Set is_past=false when the date is in the future."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "is_past": {
                        "type": "boolean",
                        "description": "True only if the full datetime is confirmed to be in the past."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation, e.g. 'February 15 at 9 AM has already passed.'"
                    }
                },
                "required": ["is_past"]
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
                    },
                    "iso_datetime": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": (
                            "ISO 8601 datetime string e.g. '2026-02-27T05:00:00'. "
                            "Set this whenever both date and time are known. "
                            "MUST be non-null when schedule_finalized=true."
                        )
                    }
                },
                "required": ["schedule_finalized"]
            }
        }
    }
]


# ─────────────────────────────────────────────────────────────
# LLM helpers (sync, run via asyncio.to_thread)
# ─────────────────────────────────────────────────────────────

def _build_system(state: dict, secondary_intent: dict | None = None, past_datetime: str | None = None) -> str:
    now   = datetime.now()
    today = now.strftime("%A, %Y-%m-%d %H:%M")

    past_instruction = ""
    if past_datetime:
        past_instruction = f"""
IMPORTANT: The date/time the user provided is in the past: "{past_datetime}".
Do NOT finalize. Tell the user this date/time has already passed and ask them to provide a future date and time.
"""

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
- If the user requests a recurring schedule (e.g. "every Tuesday"), inform them that
  only a single appointment can be booked, and offer the nearest upcoming occurrence.
- Finalize only after the user explicitly confirms all details.
- Always call update_schedule to reflect any new information extracted.
- If off-topic content is detected, call flag_secondary_intent and do NOT answer it.
- When all info is collected and confirmed, set schedule_finalized=true and respond
  with ONE short closing sentence. Do NOT ask if there is anything else you can help with.
  Do NOT invite further conversation.
- Keep responses short and conversational.
{past_instruction}{drift_instruction}"""


def groq_initiate() -> str:
    r = groq_client.chat.completions.create(
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
    return r.choices[0].message.content.strip()


def groq_dialogue(history: list, state: dict, user_msg: str) -> tuple[str, dict, str | None]:
    messages = history + [{"role": "user", "content": user_msg}]

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": _build_system(state, past_datetime=None)}] + messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.5,
        max_tokens=512,
    )

    message      = response.choices[0].message
    reply_text   = message.content or ""
    new_state    = state
    secondary_intent = None
    iso_datetime = None
    past_datetime = None

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
                if args.get("iso_datetime"):
                    iso_datetime = args["iso_datetime"]
                    print(f"[TOOL] iso_datetime: {iso_datetime}")

            elif name == "validate_datetime":
                print(f"[TOOL] validate_datetime: {args}")
                if args.get("is_past"):
                    past_datetime = args.get("reason", "The specified date and time is in the past.")
                    print(f"[TOOL] past datetime detected: {past_datetime}")
                    new_state = {**new_state, "schedule_finalized": False}

            elif name == "flag_secondary_intent":
                secondary_intent = args
                print(f"[TOOL] flag_secondary_intent: {args}")

    if message.tool_calls and not reply_text:
        tool_results = [
            {"role": "tool", "tool_call_id": tc.id, "content": "ok"}
            for tc in message.tool_calls
        ]
        followup = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _build_system(new_state, secondary_intent, past_datetime)},
                *messages,
                {"role": "assistant", "content": None, "tool_calls": message.tool_calls},
                *tool_results,
            ],
            temperature=0.5,
            max_tokens=256,
        )
        reply_text = followup.choices[0].message.content or "Got it!"

    return reply_text, new_state, iso_datetime


# ─────────────────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────────────────

async def elevenlabs_tts(text: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            url,
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
            json={
                "text": text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {"stability": 0.4, "similarity_boost": 0.8},
            },
        )
        r.raise_for_status()
        return r.content


# ─────────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────────

app = FastAPI()

session = {
    "history":      [],
    "state":        empty_state(),
    "iso_datetime": None,
    "ics_path":     None,
}


@app.get("/")
def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/favicon.ico")
def favicon():
    return FileResponse(Path(__file__).with_name("favicon.ico"))


@app.get("/download-ics")
def download_ics():
    from fastapi import HTTPException
    p = session.get("ics_path")
    if not p or not Path(p).exists():
        raise HTTPException(404, "No .ics file yet")
    return FileResponse(p, media_type="text/calendar", filename=Path(p).name)


@app.get("/debug")
def debug():
    return {k: v for k, v in session.items() if k != "history"}


# ─────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    import websockets as wsl

    await websocket.accept()

    # Queue: EndOfTurn transcripts → LLM processor
    transcript_q: asyncio.Queue = asyncio.Queue()

    # Audio queue: raw PCM chunks from browser → Deepgram
    audio_q: asyncio.Queue = asyncio.Queue()

    # Flag: is agent currently playing TTS audio in browser?
    agent_speaking = asyncio.Event()

    # ── send helpers ─────────────────────────────────────────

    async def tx(msg: str):
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(msg)
        except Exception:
            pass

    async def speak(text: str):
        """Send TTS audio to browser, set/clear agent_speaking around it."""
        try:
            mp3 = await elevenlabs_tts(text)
            if websocket.client_state == WebSocketState.CONNECTED:
                agent_speaking.set()
                await websocket.send_bytes(mp3)
                # agent_speaking cleared when browser sends __audio_done__
        except Exception as e:
            print(f"[TTS ERROR] {e}")
            agent_speaking.clear()

    # ── Deepgram Flux background task ────────────────────────

    async def deepgram_task():
        """
        Maintains a persistent Deepgram Flux WebSocket.
        Events handled:
          - StartOfTurn  → user started speaking → interrupt agent TTS
          - EndOfTurn    → user finished turn    → push transcript to LLM queue
          - Update       → interim transcript    → show in UI
        """
        while True:
            try:
                async with wsl.connect(
                    DEEPGRAM_FLUX_URL,
                    additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                    ping_interval=None,
                ) as dg:
                    print("[FLUX] connected")

                    async def sender():
                        """Forward mic audio from browser to Deepgram."""
                        while True:
                            chunk = await audio_q.get()
                            if chunk is None:
                                return
                            try:
                                await dg.send(chunk)
                            except Exception as e:
                                print(f"[FLUX] send error: {e}")
                                return

                    async def receiver():
                        """
                        Parse Deepgram Flux TurnInfo messages.

                        TurnInfo shape:
                        {
                          "type": "TurnInfo",
                          "event": "StartOfTurn" | "EndOfTurn" | "Update" | "EagerEndOfTurn" | "TurnResumed",
                          "transcript": "...",
                          "end_of_turn_confidence": 0.86,
                          "turn_index": 0
                        }
                        """
                        async for raw in dg:
                            try:
                                data = json.loads(raw)
                            except Exception:
                                continue

                            msg_type = data.get("type", "")

                            # ── TurnInfo is the main event type from Flux ──
                            if msg_type == "TurnInfo":
                                event      = data.get("event", "")
                                transcript = data.get("transcript", "").strip()
                                confidence = data.get("end_of_turn_confidence", 0.0)

                                print(f"[FLUX] {event} conf={confidence:.2f} '{transcript[:60]}'")

                                if event == "StartOfTurn":
                                    # User started speaking → interrupt agent if playing
                                    if agent_speaking.is_set():
                                        agent_speaking.clear()
                                        await tx("__interrupt__")
                                        print("[FLUX] barge-in → __interrupt__")
                                    await tx("__user_started__")

                                elif event == "Update":
                                    # Interim transcript — show live in UI
                                    if transcript:
                                        await tx(f"__interim__{transcript}")

                                elif event == "EndOfTurn":
                                    # High-confidence turn complete (≥0.7 by config)
                                    if transcript:
                                        await tx(f"__user__{transcript}")
                                        await transcript_q.put(transcript)
                                    else:
                                        print("[FLUX] EndOfTurn with empty transcript — ignored")

                                # EagerEndOfTurn / TurnResumed — not used (no eager_eot_threshold set)

                            elif msg_type == "Error":
                                print(f"[FLUX] error: {data}")

                    await asyncio.gather(sender(), receiver())
                    print("[FLUX] session ended — reconnecting")

            except Exception as e:
                print(f"[FLUX ERROR] {e} — retry 1s")
                await asyncio.sleep(1)

    # ── LLM processor ────────────────────────────────────────

    async def transcript_processor():
        """Consumes EndOfTurn transcripts → Groq → ElevenLabs → browser."""
        while True:
            transcript = await transcript_q.get()
            if not transcript.strip():
                continue

            print(f"[LLM] processing: '{transcript}'")

            hist  = session["history"]
            state = session["state"]

            try:
                reply, new_state, iso_dt = await asyncio.to_thread(
                    groq_dialogue, hist, state, transcript
                )
            except Exception as e:
                import traceback
                print(f"[LLM ERROR] {e}")
                traceback.print_exc()
                await tx("__agent__Sorry, I had an error. Please try again.")
                continue

            print(f"[LLM] reply: '{reply}'")

            if iso_dt:
                session["iso_datetime"] = iso_dt

            await tx(f"__agent__{reply}")

            if is_finalized(new_state):
                try:
                    ics_path = save_ics(new_state, session["iso_datetime"])
                    session["ics_path"] = str(ics_path)
                    await tx("__ics_ready__")
                except Exception as e:
                    print(f"[ICS ERROR] {e}")
                session["history"]      = []
                session["state"]        = empty_state()
                session["iso_datetime"] = None
                # __finalized__ tells browser: stop mic, play this audio, then show START
                await tx("__finalized__")
                await speak(reply)
            else:
                session["history"] = hist + [
                    {"role": "user",      "content": transcript},
                    {"role": "assistant", "content": reply},
                ]
                session["state"] = new_state
                await speak(reply)

    # ── start background tasks ────────────────────────────────

    dg_task  = asyncio.create_task(deepgram_task())
    llm_task = asyncio.create_task(transcript_processor())

    # ── receive loop ──────────────────────────────────────────

    try:
        while True:
            msg = await websocket.receive()

            if "text" in msg:
                cmd = msg["text"]

                if cmd == "__start__":
                    # Reset session and send greeting
                    session.update(history=[], state=empty_state(), iso_datetime=None)
                    greeting = await asyncio.to_thread(groq_initiate)
                    session["history"] = [{"role": "assistant", "content": greeting}]
                    await tx(f"__agent__{greeting}")
                    await speak(greeting)

                elif cmd == "__audio_done__":
                    # Browser finished playing agent audio → clear speaking flag
                    agent_speaking.clear()
                    print("[WS] agent audio done")

                elif cmd == "__stop__":
                    # Emergency stop — reset session
                    session.update(history=[], state=empty_state(), iso_datetime=None)
                    agent_speaking.clear()
                    print("[WS] session stopped by user")

            elif "bytes" in msg:
                # Raw PCM from browser mic → forward to Deepgram Flux
                chunk = msg["bytes"]
                await audio_q.put(chunk)

    except WebSocketDisconnect:
        print("[WS] disconnected")
    except Exception as e:
        print(f"[WS ERROR] {e}")
    finally:
        dg_task.cancel()
        llm_task.cancel()
        await audio_q.put(None)  # unblock sender


# ─────────────────────────────────────────────────────────────
# Frontend HTML
# ─────────────────────────────────────────────────────────────

INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Voice Scheduling Agent</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: Arial, sans-serif; background: #f0f2f5;
         display: flex; justify-content: center; padding: 40px 16px; }
  .container { width: 100%; max-width: 480px; }
  h2 { text-align: center; margin-bottom: 20px; color: #222; }
  #chat { background: #fff; border: 1px solid #ddd; border-radius: 10px;
          padding: 16px; height: 380px; overflow-y: auto; margin-bottom: 16px; }
  .user    { color: #222; margin: 6px 0; }
  .interim { color: #aaa; font-style: italic; margin: 4px 0; font-size: 13px; }
  .agent   { color: #0b5cff; margin: 6px 0; }
  .system  { color: #999; font-size: 12px; margin: 4px 0; }

  /* Controls row: indicator left, button right */
  .controls {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  /* Listening indicator — always visible during session */
  #indicator {
    display: none;          /* hidden before session starts */
    align-items: center;
    gap: 8px;
    flex: 1;
    font-size: 13px;
    color: #555;
  }
  #indicator .dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #4caf50;
    flex-shrink: 0;
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.8); }
  }

  /* START / STOP button */
  #mainBtn {
    padding: 12px 24px;
    border: none; border-radius: 8px;
    cursor: pointer; font-size: 15px; font-weight: 600;
    transition: background 0.2s;
    white-space: nowrap;
  }
  #mainBtn.start { background: #28a745; color: #fff; width: 100%; }
  #mainBtn.stop  { background: #dc3545; color: #fff; }
</style>
</head>
<body>
<div class="container">
  <h2>&#128197; Voice Scheduling Agent</h2>
  <div id="chat"></div>

  <div class="controls">
    <div id="indicator">
      <div class="dot"></div>
      <span id="indicatorText">Listening...</span>
    </div>
    <button id="mainBtn" class="start" onclick="handleBtn()">START</button>
  </div>
</div>
<script>
const chat          = document.getElementById("chat");
const mainBtn       = document.getElementById("mainBtn");
const indicator     = document.getElementById("indicator");
const indicatorText = document.getElementById("indicatorText");

let ws, micState = null, currentAudio = null;
let interimDiv = null;
let pendingStart = false;
let sessionActive = false;
let finalizing = false;   // true after __finalized__, mic stops, waiting for audio to end

function addMsg(text, cls) {
  if (cls !== "interim" && interimDiv) {
    interimDiv.remove();
    interimDiv = null;
  }
  const d = document.createElement("div");
  d.className = cls;
  d.textContent = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  if (cls === "interim") interimDiv = d;
  return d;
}

function setIndicator(text) {
  indicatorText.textContent = text;
}

function stopCurrentAudio() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.src = "";
    currentAudio = null;
  }
}

function stopMic() {
  if (micState) {
    micState.processor.disconnect();
    micState.source.disconnect();
    micState.stream.getTracks().forEach(t => t.stop());
    micState.ctx.close();
    micState = null;
  }
}

function showStartBtn() {
  sessionActive = false;
  finalizing = false;
  indicator.style.display = "none";
  mainBtn.className = "start";
  mainBtn.style.width = "100%";
  mainBtn.textContent = "START";
}

function showSessionControls() {
  sessionActive = true;
  indicator.style.display = "flex";
  mainBtn.className = "stop";
  mainBtn.style.width = "";
  mainBtn.textContent = "STOP";
}

function handleBtn() {
  if (!sessionActive) {
    startChat();
  } else {
    emergencyStop();
  }
}

function emergencyStop() {
  stopCurrentAudio();
  stopMic();
  if (ws && ws.readyState === WebSocket.OPEN) ws.send("__stop__");
  showStartBtn();
  addMsg("--- stopped ---", "system");
}

function connectWS() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    addMsg("connected", "system");
    if (pendingStart) {
      pendingStart = false;
      ws.send("__start__");
    }
  };
  ws.onclose = () => {
    addMsg("disconnected", "system");
    if (sessionActive) showStartBtn();
  };
  ws.onerror = () => addMsg("connection error", "system");

  ws.onmessage = async ({ data }) => {

    // ── Binary: agent TTS audio ──────────────────────────
    if (data instanceof ArrayBuffer) {
      stopCurrentAudio();
      setIndicator("Agent speaking...");

      const audio = new Audio(URL.createObjectURL(
        new Blob([data], { type: "audio/mpeg" })
      ));
      currentAudio = audio;

      const onDone = () => {
        currentAudio = null;
        if (finalizing) {
          // Last phrase played — now show START
          stopMic();
          showStartBtn();
          addMsg("--- session complete ---", "system");
          ws.send("__audio_done__");
        } else {
          ws.send("__audio_done__");
          setIndicator("Listening...");
        }
      };

      audio.onended = onDone;
      audio.onerror = onDone;
      audio.play().catch(onDone);
      return;
    }

    // ── Text messages ────────────────────────────────────
    if (typeof data !== "string") return;

    if (data === "__interrupt__") {
      if (!finalizing) {
        stopCurrentAudio();
        ws.send("__audio_done__");
        setIndicator("Listening...");
      }
      return;
    }

    if (data === "__user_started__") {
      if (!finalizing) setIndicator("Listening...");
      return;
    }

    if (data.startsWith("__interim__")) {
      if (finalizing) return;
      const text = data.slice(11);
      if (interimDiv) {
        interimDiv.textContent = "... " + text;
      } else {
        addMsg("... " + text, "interim");
      }
      return;
    }

    if (data.startsWith("__user__")) {
      addMsg("You: " + data.slice(8), "user");
      return;
    }

    if (data.startsWith("__agent__")) {
      addMsg("Agent: " + data.slice(9), "agent");
      if (!finalizing) setIndicator("Listening...");
      return;
    }

    if (data === "__finalized__") {
      // Session done — stop sending mic audio, wait for TTS to finish
      finalizing = true;
      setIndicator("Finishing...");
      return;
    }

    if (data === "__ics_ready__") {
      const d = document.createElement("div");
      d.className = "system";
      d.innerHTML = '&#128197; <a href="/download-ics" download>Download .ics</a>';
      chat.appendChild(d);
      chat.scrollTop = chat.scrollHeight;
      return;
    }

    if (data.startsWith("__system__")) {
      addMsg(data.slice(10), "system");
      return;
    }
  };
}

async function startMic() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });
    const ctx       = new AudioContext({ sampleRate: 16000 });
    const source    = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(2048, 1, 1);

    processor.onaudioprocess = (e) => {
      if (finalizing) return;   // don't send audio during finalization
      if (ws.readyState !== WebSocket.OPEN) return;
      const f32 = e.inputBuffer.getChannelData(0);
      const i16 = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++)
        i16[i] = Math.max(-32768, Math.min(32767, f32[i] * 32768));
      ws.send(i16.buffer);
    };

    source.connect(processor);
    processor.connect(ctx.destination);
    micState = { ctx, source, processor, stream };
  } catch(e) {
    addMsg("Mic error: " + e.message, "system");
  }
}

async function startChat() {
  showSessionControls();
  setIndicator("Starting...");
  await startMic();

  if (ws.readyState === WebSocket.OPEN) {
    ws.send("__start__");
  } else {
    pendingStart = true;
  }
}

connectWS();
</script>
</body>
</html>"""