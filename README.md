# Voice Scheduling Agent (Deployed)

A real-time voice assistant that collects scheduling details, confirms them, and creates a calendar event file.

## Deployed URL

1. Open the deployed URL in a browser:
   - https://renewed-causal-phoenix.ngrok-free.app/
2. Click the **START** button to initiate the assistant.
3. The assistant will begin speaking. Wait until the **Tap to record** button appears, which indicates it is ready to listen.
4. **Always** tap **Tap to record** before you speak. If you do not tap record, the assistant will not listen.
5. Speak your request (scheduling-related or unrelated).
6. **Always** tap **Stop recording** immediately after you finish speaking. If you do not stop recording, the assistant will not process your message correctly.
7. Wait for the assistant’s reply, then repeat the same cycle for each new turn: **Tap to record** → speak → **Stop recording**.
8. Continue until all scheduling details are collected and confirmed.
9. After final confirmation, the assistant provides an `.ics` file to download.
10. Import the downloaded file into your preferred calendar app (Google Calendar, Outlook, Mozilla Thunderbird, Apple Calendar, etc.).

## What this agent does

This assistant is intentionally focused, strict, and task-oriented.

- Starts the conversation after user initiation.
- Collects the user’s:
  - name,
  - preferred date,
  - preferred time,
  - optional meeting title.
- Confirms the final booking details.
- Generates a real calendar event as an `.ics` file (downloadable and importable into Google Calendar, Outlook, Apple Calendar, Mozilla Thunderbird, etc.).

## Behavioral constraints (intentional design)

To keep scheduling reliable, the agent follows strict rules:

1. **Out-of-scope requests are not fulfilled.**  
   If the user mixes in unrelated intent (for example, asking for coding help while scheduling), the assistant ignores the secondary request and redirects back to scheduling.

2. **Past date/time validation is enforced.**  
   The assistant checks whether the provided date/time is in the past and rejects it, asking for a future slot.

3. **Only one single appointment is allowed.**  
   The assistant does not allow recurring events and does not accept multiple alternative times in one request (for example: “Friday or Sunday noon, you decide”). It will require exactly one clear date and one clear time.

4. **Functional communication style.**  
   The assistant keeps responses concise, professional, and low-emotion so the flow remains efficient and predictable.

## Run locally (optional)

### Prerequisites

- Python 3.10+
- API keys for:
  - Groq
  - Deepgram
  - ElevenLabs

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn websockets groq httpx python-dotenv icalendar
cp env.example .env
# Fill in API keys in .env
```

### Start

```bash
uvicorn app.main:app --reload
```

Then open:

- http://127.0.0.1:8000

## Calendar integration details

This project creates calendar events using the iCalendar standard:

- On confirmed booking, the assistant builds an event payload from collected state.
- Event fields include:
  - summary (`meeting title` or fallback `Meeting with <name>`),
  - start time (`dtstart` from ISO datetime),
  - end time (`+1 hour`),
  - organizer (`user_name`).
- The event is saved as an `.ics` file and made available to download/import.

Because `.ics` is a universal calendar format, users can import the generated event into Google Calendar or any major calendar provider.

## Logs / evidence of event creation

During runtime, server logs print scheduling tool activity and state updates (including finalized schedule and datetime extraction). This can be used as evidence that event creation flow was executed.