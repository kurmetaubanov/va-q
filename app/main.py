"""
Voice Scheduling Agent
STT: Deepgram  |  LLM: Groq  |  TTS: ElevenLabs

pip install groq fastapi uvicorn websockets icalendar httpx python-dotenv
uvicorn app.main:app --reload
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

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

DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2&language=en-US&punctuate=true"
    "&interim_results=true&endpointing=400"
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
# Groq tools
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

# ─────────────────────────────────────────────────────────────
# Pure sync functions — run via asyncio.to_thread, never block loop
# ─────────────────────────────────────────────────────────────

def _build_system(state: dict, secondary_intent: dict | None = None) -> str:
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


def groq_initiate() -> str:
    """Opening greeting. Sync — run in thread pool."""
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
    """
    One dialogue turn. Sync — run in thread pool.
    Returns (reply_text, new_state, iso_datetime_or_None).
    """
    messages = history + [{"role": "user", "content": user_msg}]

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": _build_system(state)}] + messages,
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
                iso_datetime = args.get("iso_datetime")
                print(f"[TOOL] resolve_datetime: {iso_datetime}")

    if message.tool_calls and not reply_text:
        tool_results = [
            {"role": "tool", "tool_call_id": tc.id, "content": "ok"}
            for tc in message.tool_calls
        ]
        followup = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _build_system(new_state, secondary_intent)},
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
# Async TTS
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

    audio_q:      asyncio.Queue = asyncio.Queue()
    transcript_q: asyncio.Queue = asyncio.Queue()
    rec_bufs:     dict          = {}   # rec_id -> [fragment, ...]

    rec_id      = 0
    rec_active  = False
    rec_started = 0.0

    dg_ready    = asyncio.Event()   # Deepgram connected and ready
    dg_flushed  = asyncio.Event()   # Deepgram finished flushing finals after CloseStream

    # ── send helpers ─────────────────────────────────────────

    async def tx(msg: str):
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(msg)
        except Exception:
            pass

    async def tx_debug(msg: str):
        print(msg)
        await tx(f"__debug__{msg}")

    async def speak(text: str):
        try:
            mp3 = await elevenlabs_tts(text)
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_bytes(mp3)
        except Exception as e:
            print(f"[TTS ERROR] {e}")

    # ── Deepgram background task ──────────────────────────────

    async def deepgram_task():
        nonlocal rec_id
        while True:
            try:
                async with wsl.connect(
                    DEEPGRAM_URL,
                    additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                    ping_interval=None,
                ) as dg:
                    print("[DG] connected")
                    dg_ready.set()
                    dg_flushed.set()   # no pending flush at start

                    async def sender():
                        while True:
                            try:
                                item = await asyncio.wait_for(audio_q.get(), timeout=5.0)
                            except asyncio.TimeoutError:
                                try:
                                    await dg.send(json.dumps({"type": "KeepAlive"}))
                                    print("[DG] KeepAlive")
                                except Exception as e:
                                    print(f"[DG] KeepAlive fail: {e}")
                                    return
                                continue

                            if item is None:
                                return
                            if isinstance(item, dict):
                                t = item["type"]
                                if t == "finalize":
                                    await dg.send(json.dumps({"type": "Finalize"}))
                                    print("[DG] Finalize sent")
                                elif t == "close":
                                    dg_flushed.clear()   # mark: waiting for finals
                                    await dg.send(json.dumps({"type": "CloseStream"}))
                                    print("[DG] CloseStream sent")
                                    return  # triggers reconnect
                            else:
                                await dg.send(item)
                                print(f"[DG] audio {len(item)}b")

                    async def receiver():
                        last_final_was_empty = False
                        async for raw in dg:
                            try:
                                data       = json.loads(raw)
                                msg_type   = data.get("type", "")
                                alt        = data.get("channel", {}).get("alternatives", [{}])[0]
                                transcript = alt.get("transcript", "")
                                is_final   = data.get("is_final", False)
                                speech_final = data.get("speech_final", False)
                                print(f"[DG←] final={is_final} speech_final={speech_final} '{transcript[:60]}'")

                                if transcript and is_final:
                                    cur = rec_id
                                    rec_bufs.setdefault(cur, []).append(transcript)
                                    print(f"[DG] rec={cur} added: '{transcript}'")
                                    last_final_was_empty = False

                                # Deepgram signals end of stream with empty final + speech_final
                                if is_final and not transcript:
                                    if last_final_was_empty:
                                        # two empty finals = stream fully flushed
                                        dg_flushed.set()
                                        print("[DG] stream flushed")
                                    last_final_was_empty = True

                                if msg_type == "Metadata":
                                    dg_flushed.set()
                                    print("[DG] Metadata received — flushed")

                            except Exception as e:
                                print(f"[DG PARSE] {e}")

                    await asyncio.gather(sender(), receiver())
                    print("[DG] session done — reconnecting")

            except Exception as e:
                print(f"[DG ERROR] {e} — retry 1s")
                await asyncio.sleep(1)
            finally:
                dg_ready.clear()

    # ── LLM processor ────────────────────────────────────────

    async def transcript_processor():
        while True:
            transcript = await transcript_q.get()
            if not transcript.strip():
                continue

            print(f"[TRANSCRIPT] {transcript}")
            await tx(f"__user__{transcript}")

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
                await tx("__mic_ready__")
                continue
            print(f"[REPLY] {reply}")

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
                await speak(reply)
                await tx("__show_start__")
            else:
                session["history"] = hist + [
                    {"role": "user",      "content": transcript},
                    {"role": "assistant", "content": reply},
                ]
                session["state"] = new_state
                await speak(reply)
                # __mic_ready__ is sent by browser after audio.onended

    # ── flush after stop_rec (fire-and-forget task) ───────────

    async def flush_recording(flushed_id: int, elapsed: float):
        await tx_debug(f"[STOP_REC] rec={flushed_id} elapsed={elapsed:.2f}s — waiting for DG flush...")
        await audio_q.put({"type": "finalize"})
        await audio_q.put({"type": "close"})

        # Give Deepgram 2s to return any remaining finals, then wait for flush signal
        await asyncio.sleep(1.0)
        try:
            await asyncio.wait_for(dg_flushed.wait(), timeout=6.0)
        except asyncio.TimeoutError:
            print(f"[STOP_REC] rec={flushed_id} DG flush timeout — collecting anyway")

        fragments = rec_bufs.pop(flushed_id, [])
        full      = " ".join(fragments).strip()
        await tx_debug(f"[STOP_REC] rec={flushed_id} → '{full}'")
        if full:
            await transcript_q.put(full)
        else:
            await tx_debug(f"[STOP_REC] rec={flushed_id} empty — skipped")
            await tx("__mic_ready__")

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
                    session.update(history=[], state=empty_state(), iso_datetime=None)
                    greeting = await asyncio.to_thread(groq_initiate)
                    session["history"] = [{"role": "assistant", "content": greeting}]
                    await tx(f"__agent__{greeting}")
                    await speak(greeting)
                    # __mic_ready__ sent by browser after audio.onended

                elif cmd == "__start_rec__":
                    # Block until DG is ready — no timeout guessing
                    if not dg_ready.is_set():
                        await tx_debug("[START_REC] waiting for Deepgram...")
                        await dg_ready.wait()
                        await tx_debug("[START_REC] Deepgram ready")

                    rec_id     += 1
                    rec_bufs[rec_id] = []
                    rec_active  = True
                    rec_started = asyncio.get_running_loop().time()
                    await tx_debug(f"[START_REC] rec={rec_id}")

                elif cmd == "__stop_rec__":
                    rec_active = False
                    elapsed    = asyncio.get_running_loop().time() - rec_started
                    asyncio.create_task(flush_recording(rec_id, elapsed))

            elif "bytes" in msg:
                chunk = msg["bytes"]
                if not rec_active:
                    continue
                print(f"[WS→DG] rec={rec_id} {len(chunk)}b q={audio_q.qsize()}")
                if dg_ready.is_set():
                    await audio_q.put(chunk)

    except WebSocketDisconnect:
        print("[WS] disconnected")
    except Exception as e:
        print(f"[WS ERROR] {e}")
    finally:
        dg_task.cancel()
        llm_task.cancel()
        await audio_q.put(None)


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
          padding: 16px; height: 380px; overflow-y: auto; margin-bottom: 20px; }
  .user   { color: #222; margin: 6px 0; }
  .agent  { color: #0b5cff; margin: 6px 0; }
  .system { color: #999; font-size: 12px; margin: 4px 0; }
  #startBtn { display: block; width: 100%; padding: 14px; background: #28a745;
              color: #fff; border: none; border-radius: 8px; cursor: pointer;
              font-size: 16px; margin-bottom: 10px; }
  #micBtn   { display: none; width: 100%; padding: 14px; background: #0b5cff;
              color: #fff; border: none; border-radius: 8px; cursor: pointer;
              font-size: 16px; }
  #micBtn.recording { background: #dc3545; }
</style>
</head>
<body>
<div class="container">
  <h2>&#128197; Voice Scheduling Agent</h2>
  <div id="chat"></div>
  <button id="startBtn" onclick="startChat()">START</button>
  <button id="micBtn"   onclick="toggleMic()">&#127908; Tap to record</button>
</div>
<script>
const chat     = document.getElementById("chat");
const micBtn   = document.getElementById("micBtn");
const startBtn = document.getElementById("startBtn");
let ws, mediaRecorder, isRecording = false, recSeq = 0;

function addMsg(text, cls) {
  const d = document.createElement("div");
  d.className = cls; d.textContent = text;
  chat.appendChild(d); chat.scrollTop = chat.scrollHeight;
}

function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen  = () => { addMsg("connected", "system"); };
  ws.onclose = e => { addMsg("disconnected", "system"); };
  ws.onerror = e => { addMsg("connection error", "system"); };

  ws.onmessage = async ({ data }) => {
    if (typeof data === "string") {
      if (data === "__show_start__") {
        startBtn.style.display = "block";
        micBtn.style.display   = "none";
        if (isRecording) stopRecording();
        addMsg("--- done ---", "system");
        return;
      }
      if (data === "__mic_ready__") {
        micBtn.disabled = false;
        if (!isRecording) micBtn.textContent = "🎤 Tap to record";
        return;
      }
      if (data.startsWith("__debug__"))  { addMsg(data.slice(9), "system"); return; }
      if (data === "__ics_ready__") {
        const d = document.createElement("div"); d.className = "system";
        d.innerHTML = '&#128197; <a href="/download-ics" download>Download .ics</a>';
        chat.appendChild(d); chat.scrollTop = chat.scrollHeight; return;
      }
      if (data.startsWith("__agent__")) { addMsg("Agent: " + data.slice(9), "agent"); return; }
      if (data.startsWith("__user__"))  { addMsg("You: "  + data.slice(8),  "user");  return; }
    }
    if (data instanceof ArrayBuffer) {
      micBtn.disabled = true;
      micBtn.classList.remove("recording");
      micBtn.textContent = "🔊 Agent speaking...";
      const audio = new Audio(URL.createObjectURL(new Blob([data], { type: "audio/mpeg" })));
      audio.onended = () => {
        micBtn.disabled = false;
        micBtn.textContent = "🎤 Tap to record";
      };
      audio.play().catch(() => {
        micBtn.disabled = false;
        micBtn.textContent = "🎤 Tap to record";
      });
    }
  };
}

function startChat() {
  startBtn.style.display = "none";
  micBtn.style.display   = "block";
  micBtn.disabled = true;
  micBtn.textContent = "⏳ Loading...";
  ws.send("__start__");
}

function toggleMic() { isRecording ? stopRecording() : startRecording(); }

async function startRecording() {
  try {
    recSeq++;
    const stream   = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
                     ? "audio/webm;codecs=opus" : "audio/webm";
    mediaRecorder  = new MediaRecorder(stream, { mimeType });
    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) ws.send(e.data);
    };
    mediaRecorder.start(250);
    ws.send("__start_rec__");
    isRecording = true;
    micBtn.disabled = false;
    micBtn.textContent = "⏹ Stop recording";
    micBtn.classList.add("recording");
  } catch(e) { addMsg("Mic error: " + e.message, "system"); }
}

function stopRecording() {
  isRecording = false;
  micBtn.disabled = true;
  micBtn.textContent = "⏳ Processing...";
  micBtn.classList.remove("recording");

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.onstop = () => {
      ws.send("__stop_rec__");
      mediaRecorder = null;
      // mic stays disabled until __mic_ready__ from server
    };
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
  } else {
    ws.send("__stop_rec__");
    mediaRecorder = null;
  }
}

connectWS();
</script>
</body>
</html>"""