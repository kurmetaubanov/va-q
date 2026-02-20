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
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from groq import Groq
from starlette.websockets import WebSocketDisconnect, WebSocketState

from .calendar_export import save_ics

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

GROQ_API_KEY       = os.environ["GROQ_API_KEY"]
DEEPGRAM_API_KEY   = os.environ["DEEPGRAM_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE   = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # default: George

MODEL = "llama-3.3-70b-versatile"
TTS_ENABLED = True  # set True to enable ElevenLabs TTS

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

@app.get("/favicon.ico")
def favicon():
    icon_path = Path(__file__).with_name("favicon.ico")
    return FileResponse(icon_path)

@app.get("/debug")
def debug():
    return {k: v for k, v in app_state.items() if k != "history"}

# ─────────────────────────────────────────
# Main WebSocket — browser ↔ server
# ─────────────────────────────────────────

DEEPGRAM_WS_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2"
    "&language=en-US"
    "&punctuate=true"
    "&interim_results=true"
    "&endpointing=400"
)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """
    One WebSocket per browser session.
    Opens its own Deepgram connection — no global queues.
    """
    import websockets as ws_lib

    await ws.accept()

    # Per-session audio queue: browser → Deepgram
    audio_q: asyncio.Queue = asyncio.Queue()
    # Per-session transcript queue: Deepgram → LLM
    transcript_q: asyncio.Queue = asyncio.Queue()
    # Buffer accumulates partial transcripts while recording
    transcript_buffer: list = []
    is_recording: dict = {"active": False}
    recording_ctx = {"id": 0, "started_at": None, "chunks": 0, "bytes": 0}

    async def send_text(msg: str):
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_text(msg)
        except Exception:
            pass

    async def send_debug(msg: str):
        print(msg)
        await send_text(f"__debug__{msg}")

    async def speak(text: str):
        if not TTS_ENABLED:
            return
        try:
            audio = await text_to_speech(text)
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_bytes(audio)
        except Exception as e:
            print(f"[TTS ERROR] {e}")

    async def process_transcript(transcript: str):
        if not transcript.strip():
            return
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

        await send_text(f"__agent__{reply}")

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

    async def deepgram_task():
        """
        Persistent Deepgram connection for the entire browser session.
        Sends KeepAlive every 5s to prevent timeout between recordings.
        """
        while True:
            try:
                async with ws_lib.connect(
                    DEEPGRAM_WS_URL,
                    additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                    ping_interval=None,  # disable websockets ping, we use DG KeepAlive
                ) as dg:
                    print("[DEEPGRAM] session connected")

                    async def dg_sender():
                        while True:
                            try:
                                item = await asyncio.wait_for(audio_q.get(), timeout=5.0)
                                if item is None:
                                    break
                                if isinstance(item, dict) and item.get("type") == "finalize":
                                    await dg.send(json.dumps({"type": "Finalize"}))
                                    print("[Q→DG] Finalize sent to Deepgram")
                                    continue

                                await dg.send(item)
                                print(f"[Q→DG] sent {len(item)} bytes to Deepgram")
                            except asyncio.TimeoutError:
                                # Send KeepAlive — keeps connection alive between recordings
                                try:
                                    await dg.send(json.dumps({"type": "KeepAlive"}))
                                    print("[DG] KeepAlive sent")
                                except Exception as e:
                                    print(f"[DG KEEPALIVE ERROR] {e}")
                                    break
                            except Exception as e:
                                print(f"[DG SENDER ERROR] {e}")
                                break

                    async def dg_receiver():
                        async for raw in dg:
                            try:
                                raw_preview = raw if isinstance(raw, str) else str(raw)
                                raw_preview = raw_preview.replace("\n", " ")[:300]
                                print(f"[DG RAW] rec={recording_ctx['id']} {raw_preview}")

                                data = json.loads(raw)
                                msg_type = data.get("type", "")
                                if msg_type == "KeepAlive":
                                    print("[DG←] KeepAlive ack")
                                    continue
                                alt = data.get("channel", {}).get("alternatives", [{}])[0]
                                transcript = alt.get("transcript", "")
                                is_final = data.get("is_final", False)
                                speech_final = data.get("speech_final", False)
                                print(f"[DG←] type={msg_type} is_final={is_final} speech_final={speech_final} transcript='{transcript[:40]}'")
                                if transcript and is_final:
                                    print(f"[DG] fragment added: '{transcript}'")
                                    transcript_buffer.append(transcript)
                                elif is_final and not transcript:
                                    print("[DG] received final event with empty transcript")
                            except Exception as e:
                                print(f"[DEEPGRAM PARSE ERROR] {e}")

                    await asyncio.gather(dg_sender(), dg_receiver())

            except Exception as e:
                print(f"[DEEPGRAM SESSION ERROR] {e} — reconnecting in 1s")
                await asyncio.sleep(1)

    async def transcript_processor():
        while True:
            transcript = await transcript_q.get()
            await process_transcript(transcript)

    # Start per-session tasks
    dg_task = asyncio.create_task(deepgram_task())
    tr_task = asyncio.create_task(transcript_processor())

    try:
        while True:
            msg = await ws.receive()

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

                elif text == "__start_rec__":
                    recording_ctx["id"] += 1
                    recording_ctx["started_at"] = asyncio.get_running_loop().time()
                    recording_ctx["chunks"] = 0
                    recording_ctx["bytes"] = 0
                    await send_debug(
                        f"[START_REC] rec={recording_ctx['id']} buffer cleared, q size={audio_q.qsize()}, recording started"
                    )
                    transcript_buffer.clear()
                    is_recording["active"] = True

                elif text == "__stop_rec__":
                    is_recording["active"] = False
                    elapsed = None
                    if recording_ctx["started_at"] is not None:
                        elapsed = asyncio.get_running_loop().time() - recording_ctx["started_at"]
                    await send_debug(
                        f"[STOP_REC] rec={recording_ctx['id']} chunks={recording_ctx['chunks']} bytes={recording_ctx['bytes']} elapsed={elapsed:.2f}s waiting for Deepgram to flush..."
                        if elapsed is not None else
                        f"[STOP_REC] rec={recording_ctx['id']} chunks={recording_ctx['chunks']} bytes={recording_ctx['bytes']} waiting for Deepgram to flush..."
                    )
                    # Force Deepgram to flush final transcript for current utterance.
                    await audio_q.put({"type": "finalize"})
                    # Wait for Deepgram to finish processing remaining audio
                    await asyncio.sleep(1.5)
                    full_transcript = " ".join(transcript_buffer).strip()
                    await send_debug(
                        f"[STOP_REC] rec={recording_ctx['id']} buffer had {len(transcript_buffer)} fragments: '{full_transcript}'"
                    )
                    transcript_buffer.clear()
                    if full_transcript:
                        await transcript_q.put(full_transcript)
                    else:
                        await send_debug(f"[STOP_REC] rec={recording_ctx['id']} buffer was empty — nothing sent to LLM")

            elif "bytes" in msg:
                chunk = msg["bytes"]
                if is_recording["active"]:
                    recording_ctx["chunks"] += 1
                    recording_ctx["bytes"] += len(chunk)
                else:
                    print("[WS→Q WARNING] received audio chunk while recording inactive")
                print(
                    f"[WS→Q] rec={recording_ctx['id']} audio chunk {len(chunk)} bytes, q size={audio_q.qsize()}, active={is_recording['active']}"
                )
                await audio_q.put(chunk)

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS ERROR] {e}")
    finally:
        dg_task.cancel()
        tr_task.cancel()
        await audio_q.put(None)  # unblock sender

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
  .container { width: 100%; max-width: 480px; }
  h2 { text-align: center; margin-bottom: 20px; color: #222; }
  #chat { background: #fff; border: 1px solid #ddd; border-radius: 10px;
          padding: 16px; height: 380px; overflow-y: auto; margin-bottom: 20px; }
  .user   { color: #222; margin: 6px 0; }
  .agent  { color: #0b5cff; margin: 6px 0; }
  .system { color: #999; font-size: 12px; margin: 4px 0; }
  #startBtn {
    display: block; width: 100%; padding: 14px;
    background: #28a745; color: #fff; border: none;
    border-radius: 8px; cursor: pointer; font-size: 16px;
    margin-bottom: 10px;
  }
  #micBtn {
    display: none; width: 100%; padding: 14px;
    background: #0b5cff; color: #fff; border: none;
    border-radius: 8px; cursor: pointer; font-size: 16px;
  }
  #micBtn.recording { background: #dc3545; }
</style>
</head>
<body>
<div class="container">
  <h2>&#128197; Voice Scheduling Agent</h2>
  <div id="chat"></div>
  <button id="startBtn" onclick="startChat()">START</button>
  <button id="micBtn" onclick="toggleMic()">&#127908; Tap to record</button>
</div>

<script>
const chat    = document.getElementById("chat");
const micBtn  = document.getElementById("micBtn");
const startBtn = document.getElementById("startBtn");

let ws, mediaRecorder, isRecording = false;
let recordingSeq = 0;

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
    console.log("[WS] connected", new Date().toISOString());
    addMsg("connected", "system");
  };

  ws.onmessage = async ({ data }) => {
    if (typeof data === "string") {
      if (data === "__show_start__") {
        startBtn.style.display = "block";
        micBtn.style.display = "none";
        if (isRecording) stopRecording();
        addMsg("--- conversation finished ---", "system");
        return;
      }
      if (data.startsWith("__debug__")) {
        const debugText = data.slice(9);
        console.debug("[SERVER DEBUG]", debugText);
        addMsg(debugText, "system");
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

    // Binary mp3 → play
    if (data instanceof ArrayBuffer) {
      const blob = new Blob([data], { type: "audio/mpeg" });
      const audio = new Audio(URL.createObjectURL(blob));
      // Disable mic while agent is speaking
      micBtn.disabled = true;
      micBtn.textContent = "🔊 Agent speaking...";
      audio.onended = () => {
        micBtn.disabled = false;
        micBtn.textContent = isRecording ? "⏹ Stop recording" : "🎤 Tap to record";
      };
      audio.play().catch(e => {
        console.warn("Audio play failed:", e);
        micBtn.disabled = false;
        micBtn.textContent = "🎤 Tap to record";
      });
    }
  };

  ws.onerror = (e) => {
    console.error("[WS] error", e);
    addMsg("connection error", "system");
  };
  ws.onclose = (e) => {
    console.warn("[WS] closed", e.code, e.reason);
    addMsg("disconnected", "system");
  };
}

async function startChat() {
  startBtn.style.display = "none";
  micBtn.style.display = "block";
  console.log("[WS] send __start__");
  ws.send("__start__");
}

async function toggleMic() {
  if (!isRecording) {
    await startRecording();
  } else {
    stopRecording();
  }
}

async function startRecording() {
  try {
    recordingSeq += 1;
    console.log(`[MIC][rec=${recordingSeq}] requesting getUserMedia...`);
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    console.log(`[MIC][rec=${recordingSeq}] stream obtained, tracks:`, stream.getTracks().length);

    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "audio/webm";
    console.log(`[MIC][rec=${recordingSeq}] using mimeType:`, mimeType);

    mediaRecorder = new MediaRecorder(stream, { mimeType });

    mediaRecorder.ondataavailable = (e) => {
      console.log(`[MIC][rec=${recordingSeq}] chunk:`, e.data.size, "bytes", "wsState=", ws.readyState);
      if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
        ws.send(e.data);
      }
    };

    mediaRecorder.start(250);
    console.log(`[MIC][rec=${recordingSeq}] MediaRecorder started, state:`, mediaRecorder.state);
    console.log(`[WS] send __start_rec__ rec=${recordingSeq}`);
    ws.send("__start_rec__");
    isRecording = true;
    micBtn.textContent = "⏹ Stop recording";
    micBtn.classList.add("recording");
  } catch (e) {
    console.error("[MIC ERROR]", e);
    addMsg("Microphone error: " + e.message, "system");
  }
}

function stopRecording() {
  console.log(`[MIC][rec=${recordingSeq}] stopRecording called, state:`, mediaRecorder ? mediaRecorder.state : "no recorder");
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.onstop = () => {
      console.log(`[MIC][rec=${recordingSeq}] onstop fired, sending __stop_rec__`);
      console.log(`[WS] send __stop_rec__ rec=${recordingSeq}`);
      ws.send("__stop_rec__");
      mediaRecorder = null;
      // Wait for Deepgram to reconnect before allowing next recording
      micBtn.disabled = true;
      micBtn.textContent = "⏳ Processing...";
      setTimeout(() => {
        if (!isRecording) {
          micBtn.disabled = false;
          micBtn.textContent = "🎤 Tap to record";
        }
      }, 2000);
    };
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => { t.stop(); console.log(`[MIC][rec=${recordingSeq}] track stopped`); });
  } else {
    console.log(`[MIC][rec=${recordingSeq}] recorder inactive, sending __stop_rec__ directly`);
    console.log(`[WS] send __stop_rec__ rec=${recordingSeq}`);
    ws.send("__stop_rec__");
    mediaRecorder = null;
  }
  isRecording = false;
  micBtn.textContent = "🎤 Tap to record";
  micBtn.classList.remove("recording");
}

connectWS();
</script>
</body>
</html>"""

@app.get("/")
def index():
    return HTMLResponse(INDEX_HTML)
