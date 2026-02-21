# Voice Scheduling Agent (Deployed)

A real-time voice assistant that collects scheduling details, confirms them, and creates a calendar event file.

## Deployed URL

1. Open the deployed URL in a browser:
   - https://renewed-causal-phoenix.ngrok-free.app/

2. Click **START** to begin interacting with the assistant.

3. Speak naturally when prompted. The assistant listens automatically — no recording controls are required.

4. You can **interrupt the assistant at any time by speaking** if you don’t want to wait for it to finish.

5. Click **STOP** only if you want to forcefully end the current session.

6. Continue the conversation until all scheduling details are collected and confirmed.

7. After you confirm the booking, the assistant will **automatically finalize the session**.

8. Once finalized, an `.ics` calendar file will appear for download.

9. Import the downloaded file into your preferred calendar app (Google Calendar, Outlook, Apple Calendar, etc.).

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