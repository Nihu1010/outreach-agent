"""Outreach agent: router + 3 tools.

first_touch -> fill_template (plain code, no LLM, so it cannot break the template)
follow_up   -> extract_pain_points (LLM) -> draft_followup (LLM)
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import re

from tracing import span

TEMPLATE_PATH = Path(__file__).parent / "data" / "first_touch_template.txt"
MIN_NOTE_WORDS = 12  # below this, notes are too thin to ground a follow-up


PROVIDERS = {  # name -> (API key env var, model env var, default model)
    "groq": ("GROQ_API_KEY", "GROQ_MODEL", "llama-3.3-70b-versatile"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-haiku-5-5"),
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o-mini"),
}


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "groq").lower()  # groq (free) | anthropic | openai


def key_env() -> str:
    return PROVIDERS[provider()][0]


def model() -> str:
    _, env, default = PROVIDERS[provider()]
    return os.getenv(env, default)


_clients = {}


def llm_json(system: str, user: str, temperature: float) -> dict:
    """One LLM call that must return a JSON object. Works with Groq, Anthropic or OpenAI."""
    p = provider()
    if p not in _clients:
        if p == "groq":
            from openai import OpenAI  # Groq speaks the OpenAI API
            _clients[p] = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
        elif p == "openai":
            from openai import OpenAI
            _clients[p] = OpenAI()  # reads OPENAI_API_KEY
        else:
            from anthropic import Anthropic
            _clients[p] = Anthropic()  # reads ANTHROPIC_API_KEY
    c = _clients[p]
    if p in ("groq", "openai"):
        resp = c.chat.completions.create(
            model=model(), temperature=temperature, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        text = resp.choices[0].message.content
    else:
        resp = c.messages.create(
            model=model(), max_tokens=800, temperature=temperature,
            system=system + "\nRespond with the JSON object only, no other text.",
            messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in resp.content if b.type == "text")
    match = re.search(r"\{.*\}", text, re.S)  # tolerate stray text or code fences around the JSON
    return json.loads(match.group(0) if match else text)


@dataclass
class Result:
    stage: str                      # first_touch | follow_up
    subject: str = ""
    body: str = ""
    pain_points: list = field(default_factory=list)  # [{"pain_point", "source_quote"}]
    message: str = ""               # shown to the user instead of a draft when we refuse to guess


# ---------- Router ----------

def route(call_notes: str) -> str:
    """Rule-based: no call notes means no call has happened yet."""
    return "follow_up" if (call_notes or "").strip() else "first_touch"


# ---------- Tool 1: template fill (no AI) ----------

def fill_template(name: str, role: str, topic: str, sender: str) -> tuple[str, str]:
    greet = f"Dr. {name.split()[-1]}" if name.startswith("Dr.") else name.split()[0]
    text = TEMPLATE_PATH.read_text().format(name=greet, role=role, topic=topic, sender=sender)
    first_line, _, rest = text.partition("\n")
    return first_line.replace("Subject:", "").strip(), rest.strip()


# ---------- Tool 2: extract pain points (LLM) ----------

EXTRACT_PROMPT = """You extract pain points from customer discovery call notes.
Rules:
- Return 1 to 3 pain points the person actually described.
- For each, copy the exact sentence or phrase from the notes that supports it into "source_quote". Copy it word for word.
- Never add a pain point that is not in the notes. If there is none, return an empty list.
Return JSON: {"pain_points": [{"pain_point": "...", "source_quote": "..."}]}"""


def extract_pain_points(notes: str) -> list[dict]:
    items = llm_json(EXTRACT_PROMPT, f"Call notes:\n{notes}", temperature=0).get("pain_points", [])
    # Guardrail: keep only pain points whose quote really appears in the notes.
    norm = " ".join(notes.split()).lower()
    return [p for p in items
            if p.get("pain_point") and " ".join(p.get("source_quote", "").split()).lower() in norm]


# ---------- Tool 3: draft follow-up (LLM) ----------

DRAFT_PROMPT = """You write short follow-up emails after a customer discovery call. The sender is a graduate student researcher.
Rules:
- Under 150 words for the body.
- Thank them and reference at least one specific thing they said, using only the pain points given.
- Do not invent facts, numbers or details that are not in the pain points.
- Exactly one clear ask: a 15-minute follow-up call OR an introduction to a colleague facing the same issue.
- Plain, warm tone. No sales language, no product pitch.
Return JSON: {"subject": "...", "body": "..."}"""


def draft_followup(name: str, role: str, org: str, pain_points: list[dict], sender: str) -> tuple[str, str]:
    points = "\n".join(f"- {p['pain_point']}" for p in pain_points)
    out = llm_json(DRAFT_PROMPT,
                   f"Recipient: {name}, {role} at {org}\nSender name: {sender}\nPain points from our call:\n{points}",
                   temperature=0.4)
    return out.get("subject", "").strip(), out.get("body", "").strip()


# ---------- Orchestration ----------

def run_contact(contact: dict, topic: str, sender: str, stage_override: str | None = None) -> Result:
    notes = str(contact.get("call_notes") or "").strip()
    with span("run_contact", "CHAIN", contact_id=contact.get("contact_id", ""), org=contact.get("org", "")) as s:
        stage = stage_override or route(notes)
        s.set_attribute("stage", stage)

        if stage == "first_touch":
            with span("fill_template", "TOOL"):
                subject, body = fill_template(contact["name"], contact["role"], topic, sender)
            return Result(stage, subject, body)

        if len(notes.split()) < MIN_NOTE_WORDS:
            return Result(stage, message="Not enough detail in the call notes. Add what they said about their work, then try again.")

        with span("extract_pain_points", "TOOL"):
            points = extract_pain_points(notes)
        if not points:
            return Result(stage, message="No clear pain points found in these notes. Add what they said was hard, then try again.")

        with span("draft_followup", "TOOL"):
            subject, body = draft_followup(contact["name"], contact["role"], contact["org"], points, sender)
        return Result(stage, subject, body, points)
