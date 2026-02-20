"""
calendar_export.py — convert scheduling state to .ics file
pip install icalendar
"""

from datetime import datetime, timedelta
from pathlib import Path

from icalendar import Calendar, Event


def build_ics(state: dict, iso_datetime: str) -> bytes:
    """
    Build .ics file bytes from finalized scheduling state.
    iso_datetime: ISO 8601 string, e.g. '2026-02-21T17:00:00'
    """
    dt = datetime.fromisoformat(iso_datetime)

    cal = Calendar()
    cal.add("prodid", "-//Scheduling Agent//EN")
    cal.add("version", "2.0")

    event = Event()
    event.add("summary", state.get("title") or f"Meeting with {state['user_name']}")
    event.add("dtstart", dt)
    event.add("dtend", dt + timedelta(hours=1))
    event.add("dtstamp", datetime.utcnow())
    event.add("organizer", state["user_name"])

    cal.add_component(event)
    return cal.to_ical()


def save_ics(state: dict, iso_datetime: str, output_dir: str = "/tmp") -> Path:
    """Save .ics to disk and return path."""
    ics_bytes = build_ics(state, iso_datetime)
    name = (state.get("title") or "meeting").replace(" ", "_").lower()
    path = Path(output_dir) / f"{name}.ics"
    path.write_bytes(ics_bytes)
    return path