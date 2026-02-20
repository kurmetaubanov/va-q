"""
Scheduling Agent — Groq + function calling
pip install groq fastapi uvicorn
GROQ_API_KEY=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 uvicorn app.main:app --reload
"""

import json
import os

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from groq import Groq
from starlette.websockets import WebSocketDisconnect

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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
                        "description": "Short description of the off-topic request, e.g. 'asked for Elixir LiveView code'"
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
                        "description": (
                            "Single specific date (e.g. 'Monday', '2026-02-24'). "
                            "Null if multiple dates mentioned or ambiguous."
                        )
                    },
                    "time": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": (
                            "Single specific time (e.g. '3pm', '15:00'). "
                            "Null if multiple times mentioned or ambiguous."
                        )
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
    }
]

# ─────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────

def build_system(state, secondary_intent=None):
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
- When all info is collected and confirmed, set schedule_finalized=true and close naturally.
- Keep responses short and conversational.
{drift_instruction}"""

# ─────────────────────────────────────────
# Core dialogue step
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

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": build_system(state)}] + messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.5,
        max_tokens=512,
    )

    choice = response.choices[0]
    message = choice.message

    reply_text = message.content or ""
    new_state = state
    secondary_intent = None

    # ── Process tool calls ──
    if message.tool_calls:
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                print(f"[TOOL ERROR] {name}: {e}")
                continue

            if name == "update_schedule":
                new_state = merge_state(state, args)
                print(f"\n[TOOL] update_schedule: {args}")
                print(f"[STATE] {new_state}\n")

            elif name == "flag_secondary_intent":
                secondary_intent = args
                print(f"\n[TOOL] flag_secondary_intent: {args}\n")

    # ── If model returned tool calls but no text → request reply ──
    # secondary_intent is injected into system prompt so model knows to block the drift
    if message.tool_calls and not reply_text:
        tool_results = [
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": "ok"
            }
            for tc in message.tool_calls
        ]

        followup = client.chat.completions.create(
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

    should_reset = is_finalized(new_state)

    return reply_text, new_state, should_reset


# ─────────────────────────────────────────
# Initiator
# ─────────────────────────────────────────

def initiate():
    response = client.chat.completions.create(
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
# FastAPI + WebSocket
# ─────────────────────────────────────────

app = FastAPI()

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Scheduling Agent</title>
<style>
  body { font-family: Arial; margin: 40px; background: #f5f5f5; }
  #chat {
    border: 1px solid #ddd; padding: 15px;
    height: 380px; overflow-y: auto;
    background: #fff; border-radius: 8px;
    margin-bottom: 12px;
  }
  .user   { color: #222; margin: 6px 0; }
  .agent  { color: #0b5cff; margin: 6px 0; }
  .system { color: #aaa; font-size: 12px; margin: 4px 0; }
  input   { width: 68%; padding: 9px; border-radius: 4px; border: 1px solid #ccc; }
  button  { padding: 9px 16px; border-radius: 4px; border: none;
            background: #0b5cff; color: #fff; cursor: pointer; }
  button:hover { background: #0040cc; }
  #startBtn { background: #28a745; }
</style>
</head>
<body>
<h2>&#128197; Scheduling Agent</h2>
<div id="chat"></div>
<button id="startBtn" onclick="startChat()">START</button>
<br><br>
<input id="msg" placeholder="Type a message..." />
<button onclick="send()">Send</button>

<script>
const chat = document.getElementById("chat");
const startBtn = document.getElementById("startBtn");

function addMsg(text, cls) {
  const d = document.createElement("div");
  d.className = cls;
  d.textContent = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

const ws = new WebSocket(`ws://${location.host}/ws`);

ws.onopen = () => {
  addMsg("connected", "system");
  startBtn.style.display = "inline-block";
};

ws.onmessage = ({ data }) => {
  if (data === "__show_start__") {
    startBtn.style.display = "inline-block";
    addMsg("--- conversation finished ---", "system");
    return;
  }
  addMsg("Agent: " + data, "agent");
};

ws.onerror = () => addMsg("connection error", "system");
ws.onclose = () => addMsg("disconnected", "system");

function startChat() {
  ws.send("__start__");
  startBtn.style.display = "none";
}

function send() {
  const msg = document.getElementById("msg").value.trim();
  if (!msg) return;
  addMsg("You: " + msg, "user");
  ws.send(msg);
  document.getElementById("msg").value = "";
}

document.getElementById("msg")
  .addEventListener("keypress", e => { if (e.key === "Enter") send(); });
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return HTMLResponse(INDEX_HTML)

@app.get("/debug")
def debug():
    return app_state

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()

            if msg == "__start__":
                greeting = initiate()
                app_state["history"] = [{"role": "assistant", "content": greeting}]
                app_state["state"] = empty_state()
                await ws.send_text(greeting)
                continue

            history = app_state["history"]
            state = app_state["state"]

            reply, new_state, should_reset = dialogue_step(history, state, msg)

            new_history = history + [
                {"role": "user", "content": msg},
                {"role": "assistant", "content": reply},
            ]

            if should_reset:
                app_state["history"] = []
                app_state["state"] = empty_state()
                await ws.send_text(reply)
                await ws.send_text("__show_start__")
            else:
                app_state["history"] = new_history
                app_state["state"] = new_state
                await ws.send_text(reply)

    except WebSocketDisconnect:
        print("Client disconnected")