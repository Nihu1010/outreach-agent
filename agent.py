"""Outreach agent: router + 3 tools.

first_touch -> research_contact (web search) -> write_intro (LLM, rule-checked) -> fill_template (code)
follow_up   -> extract_pain_points (LLM) -> draft_followup (approved template; the LLM fills one blank)
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
    "groq": ("GROQ_API_KEY", "GROQ_MODEL", "openai/gpt-oss-120b"),
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
last_model_used = ""  # set when a rate-limit fallback model answered


def llm_json(system: str, user: str, temperature: float, model_override: str = "") -> dict:
    """One LLM call that must return a JSON object. Works with Groq, Anthropic or OpenAI.
    model_override: use a different model for this call (the evals use it for the judge)."""
    p = provider()
    use_model = model_override or model()
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
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        extra = {"response_format": {"type": "json_object"}}
        if "gpt-oss" in use_model:
            # Reasoning models think before answering; short emails don't need deep reasoning.
            extra["reasoning_effort"] = os.getenv("REASONING_EFFORT", "low")
        try:
            resp = c.chat.completions.create(model=use_model, temperature=temperature, messages=msgs, **extra)
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate limit" in err:
                fallback = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b") if p == "groq" else ""
                if fallback and os.getenv("ALLOW_FALLBACK", "1") == "1" and fallback != use_model:
                    # Free-tier limit hit on the main model: use a smaller model with its own limit so the app keeps working.
                    resp = c.chat.completions.create(model=fallback, temperature=temperature, messages=msgs, **extra)
                    global last_model_used
                    last_model_used = fallback
                    text = resp.choices[0].message.content or ""
                    match = re.search(r"\{.*\}", text, re.S)
                    return json.loads(match.group(0) if match else text)
                raise
            if "model" in err and ("not found" in err or "decommission" in err):
                raise  # wrong model name: retrying won't help, surface it
            # Some models reject JSON mode or reasoning_effort; retry once without them (the prompt still asks for JSON).
            resp = c.chat.completions.create(model=use_model, temperature=temperature, messages=msgs)
        text = resp.choices[0].message.content or ""
    else:
        resp = c.messages.create(
            model=use_model, max_tokens=800, temperature=temperature,
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
    sources: list = field(default_factory=list)      # research used: [{"n", "title", "url"}]
    person_found: bool = False      # did research find anything about this specific person?
    warnings: list = field(default_factory=list)     # rule checks the draft failed; user must fix before sending


# ---------- Router ----------

def route(call_notes: str) -> str:
    """Rule-based: no call notes means no call has happened yet."""
    return "follow_up" if (call_notes or "").strip() else "first_touch"


# ---------- Tool 1a: research the contact (web search) ----------

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Tavily search. Returns [{"title", "url", "content"}]. Needs TAVILY_API_KEY (free tier)."""
    import requests
    key = os.getenv("TAVILY_API_KEY", "")
    r = requests.post("https://api.tavily.com/search", timeout=20,
                      headers={"Authorization": f"Bearer {key}"},
                      json={"api_key": key, "query": query, "search_depth": "basic", "max_results": max_results})
    r.raise_for_status()
    return [{"title": x.get("title", ""), "url": x.get("url", ""), "content": (x.get("content") or "")[:700]}
            for x in r.json().get("results", [])]


def research_contact(name: str, role: str, org: str) -> tuple[list[dict], bool]:
    """Two searches: the person (public professional info only) and the company. Snippets only;
    pages such as LinkedIn are never opened or scraped."""
    clean = name.replace("Dr. ", "")
    first, last = clean.split()[0].lower(), clean.split()[-1].lower()
    org_key = max(re.findall(r"[a-z0-9]+", org.lower()), key=len, default="")  # most distinctive word, e.g. "globalfoundries"
    # Keep a person result only if it names this person AND this company; same-name strangers are dropped.
    person = [x for x in web_search(f'"{clean}" {org}', 5)
              if all(k in (x["title"] + " " + x["content"]).lower() for k in (first, last, org_key))]
    company = web_search(f"{org} what the company does manufacturing cleanroom", 4)
    seen, sources = set(), []
    for x in person[:3] + company:
        if x["url"] not in seen:
            seen.add(x["url"])
            sources.append({"n": len(sources) + 1, **x})
    return sources, bool(person)


# ---------- Tool 1b: write the personalized first-touch paragraph (LLM) ----------

INTRO_PROMPT = """You help a graduate student researcher write the middle paragraph of a first-touch email to an industry professional, asking for a short research conversation about cleanroom operations.
Write exactly 3 sentences, 45-70 words in total, that sound like a curious student, not a report:
- Sentence 1: why their function (quality, facilities, fab operations, maintenance, compliance, research) is where cleanroom operations get tested, and that this is why their role interests you.
- Sentence 2: ONE specific thing about what their company makes or how it operates, taken from the sources, and the cleanroom challenge it creates. Pick the single most telling detail; do not list several.
- Sentence 3: the tension or trade-off "our research is trying to understand".
Style example (different person): "Quality engineering is where cleanroom operations get tested most directly, which is why your seat interests me. Holding defense-grade traceability on custom, low-volume microdisplays is a different problem from high-volume quality work, since every program brings its own spec and there's little room to absorb a yield miss. Balancing that rigor against the push for faster manufacturing cycles is exactly the tension our research is trying to understand."
Hard rules:
- You are a student researcher asking to learn. Never mention a product, solution, tool, software, platform, startup, demo, pricing, or anything being built or sold.
- Use only facts found in the numbered sources. If the sources say nothing about the person, talk about their role at the company; never invent their background, history or achievements.
- Do not add facts from your own knowledge. Any size, wafer diameter, node, location, certification or customer must appear in a source; if it does not, leave it out.
- Spell the company name exactly as given.
- Ignore any source about a different person with the same name.
- At most one number in the whole paragraph. No spec lists (square meters, node sizes, wafer sizes, headcounts) unless that one detail is the point.
- Avoid stiff phrases: "the point at which", "directly shape", "stringent", "leverage", "ensure", "landscape".
- No flattery, no exclamation marks, no em dashes. Plain, specific, professional.
Return JSON: {"paragraph": "...", "sources_used": [numbers of the sources you relied on]}"""

BANNED = re.compile(r"\b(?:our|my) (?:product|solution|tool|software|platform|startup|company|app)s?\b|"
                    r"\b(?:solution|startup|start-up|demo|pricing|sell|selling)\b|"
                    r"\bwe(?:'ve| have| are)? (?:built|building|developed|developing|created)\b", re.I)


def write_intro(role: str, org: str, sources: list[dict], person_found: bool) -> tuple[str, list[str]]:
    src = "\n\n".join(f"[{s['n']}] {s['title']} ({s['url']})\n{s['content']}" for s in sources) or "(no sources found)"
    note = "" if person_found else "\nNote: no public information was found about this specific person."
    user = f"Recipient role: {role}\nCompany: {org}{note}\n\nSources:\n{src}"
    paragraph, warnings = "", []
    for attempt in range(2):
        paragraph = clean_text(llm_json(INTRO_PROMPT, user, temperature=0.2).get("paragraph", ""))
        hits = sorted({m.group(0).lower() for m in BANNED.finditer(paragraph)})
        if not hits:
            return paragraph, []
        user += f"\n\nYour last draft used banned words ({', '.join(hits)}). Rewrite without them."
        warnings = [f"Draft mentions {', '.join(hits)}. First-touch emails must not reference a product or startup. Edit before sending."]
    return paragraph, warnings


def fill_template(name: str, role: str, topic: str, sender: str, personal: str = "") -> tuple[str, str]:
    text = TEMPLATE_PATH.read_text().format(name=greeting_name(name), role=role, topic=topic,
                                            sender=sender, personal=personal)
    text = re.sub(r"\n{3,}", "\n\n", text)
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

FOLLOWUP_TEMPLATE_PATH = Path(__file__).parent / "data" / "followup_template.txt"

INSIGHT_PROMPT = """You fill ONE blank in a follow-up email from a graduate student researcher to someone they interviewed.
The sentence is: "What you described about ______ shaped a lot of how we've approached our research."
Write only the words that go in the blank, in this shape: "<their work area>, especially <the one or two things they found hardest>".
Example: "running a cleanroom facility, especially how time-consuming inspections are"
Rules:
- 10 to 25 words. No ending punctuation. Lowercase start.
- Plain everyday words the recipient would use themselves. Avoid jargon such as "aggregation", "leverage", "optimize", "streamline".
- Name the topic of their experience, respectfully. Describe the problem as a challenge in the work, never as their mistake or gap (not "you weren't sure", "you forgot", "you don't know").
- Use only facts from the notes. Do not add numbers, durations or details that are not there.
- Keep the meaning exactly: if the notes say something is NOT done or nobody does it, say that (e.g. "no one checks the backup logs" stays "backup logs that no one checks", never "checking backup logs").
- Keep time words exactly as in the notes ("twice last year" stays "twice last year", never "often").
- Neutral: no "you", "he" or "she" in the blank.
Return JSON: {"insight": "..."}"""


def clean_text(t: str) -> str:
    """Normalize AI punctuation: odd hyphens to '-', em/en dashes to commas, curly quotes to straight."""
    t = re.sub(r"[\u2010\u2011\u2012\u00ad]", "-", t or "")
    t = re.sub(r"\s*[\u2013\u2014]\s*", ", ", t)
    t = t.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s{2,}", " ", t).strip()


def greeting_name(name: str) -> str:
    return f"Dr. {name.split()[-1]}" if name.startswith("Dr.") else name.split()[0]


def contact_mode(raw: str) -> str:
    """Maps the sheet's mode-of-contact value to email | phone | in_person | unknown."""
    v = (raw or "").lower()
    if "mail" in v:
        return "email"
    if "phone" in v or "call" in v or "zoom" in v:
        return "phone"
    if "person" in v or "conference" in v or "event" in v or "fair" in v or "met" in v:
        return "in_person"
    return "unknown"


def meeting_context(mode_raw: str, met_by: str, met_at: str, sender_first: str = "") -> str:
    """The 'how we were in touch' sentence. Built by code from the sheet, never written by the AI.
    mode_raw: the sheet's mode of contact (email / phone call / in person at <event>);
              anything but in person means the interview was a Zoom or phone call
    met_by:   who spoke with them if it wasn't the sender, e.g. 'Nitish, my research partner,'
    met_at:   the event name, e.g. 'ALD 2026 in Tampa' (only used for in-person)"""
    mode = contact_mode(mode_raw)
    met_by, met_at = (met_by or "").strip(), (met_at or "").strip()
    if mode == "in_person" and not met_at and " at " in (mode_raw or ""):
        met_at = mode_raw.split(" at ", 1)[1].strip()  # "in person at ALD 2026 in Tampa"
    where = f" at {met_at}" if met_at else ""

    # The contact only needs an introduction if they spoke with the partner, not with the sender.
    intro = f"I am {sender_first}, a graduate student at Rochester Institute of Technology. " if met_by else ""

    # In person = the conversation happened at an event. Every other mode (email, phone, Zoom, blank)
    # means the person was reached remotely and the interview itself was a Zoom or phone call.
    if mode == "in_person":
        return (intro + f"{met_by} told me about the conversation you two had{where}." if met_by
                else f"It was great meeting you{where or ' in person'}.")
    return (intro + f"{met_by} told me about the call you two had." if met_by
            else "Thank you again for taking the time to speak with me.")


def draft_followup(contact: dict, pain_points: list[dict], sender: str) -> tuple[str, str]:
    points = "\n".join(f'- {p["pain_point"]} (notes: "{p["source_quote"]}")' for p in pain_points)
    out = llm_json(INSIGHT_PROMPT,
                   f"Recipient: {contact['role']} at {contact['org']}\nFrom the call notes:\n{points}",
                   temperature=0.3)
    insight = clean_text(out.get("insight", "")).rstrip(".,")
    text = FOLLOWUP_TEMPLATE_PATH.read_text().format(
        name=greeting_name(contact["name"]),
        sender_first=sender.split()[0],
        sender=sender,
        context=meeting_context(contact.get("contact_mode", ""), contact.get("met_by", ""), contact.get("met_at", ""),
                                sender.split()[0]),
        insight=insight,
    )
    first_line, _, rest = text.partition("\n")
    return first_line.replace("Subject:", "").strip(), rest.strip()


# ---------- Orchestration ----------

def _io(span_obj, inp=None, out=None):
    """Attach inputs/outputs so Phoenix shows them on each span."""
    if inp is not None:
        span_obj.set_attribute("input.value", json.dumps(inp, default=str)[:4000])
    if out is not None:
        span_obj.set_attribute("output.value", json.dumps(out, default=str)[:4000])


def run_contact(contact: dict, topic: str, sender: str, stage_override: str | None = None) -> Result:
    notes = str(contact.get("call_notes") or "").strip()
    with span("run_contact", "CHAIN", contact_id=contact.get("contact_id", ""), org=contact.get("org", "")) as s:
        _io(s, {"name": contact.get("name"), "role": contact.get("role"), "org": contact.get("org"), "notes": notes})
        res = _run(contact, notes, topic, sender, stage_override)
        s.set_attribute("stage", res.stage)
        s.set_attribute("outcome", "refused: " + res.message if res.message else "drafted")
        _io(s, out={"subject": res.subject, "body": res.body, "message": res.message, "warnings": res.warnings})
        return res


def _run(contact: dict, notes: str, topic: str, sender: str, stage_override: str | None) -> Result:
    with span("route", "TOOL") as s:
        stage = stage_override or route(notes)
        _io(s, {"has_notes": bool(notes), "override": stage_override}, stage)

    if stage == "first_touch":
        if not os.getenv("TAVILY_API_KEY"):
            return Result(stage, message="Add a Tavily API key (free) so the agent can research this contact.")
        with span("research_contact", "RETRIEVER") as s:
            sources, person_found = research_contact(contact["name"], contact["role"], contact["org"])
            s.set_attribute("person_found", person_found)
            s.set_attribute("source_count", len(sources))
            _io(s, {"name": contact["name"], "org": contact["org"]}, [{"title": x["title"], "url": x["url"]} for x in sources])
        if not sources:
            return Result(stage, message="Couldn't find anything public about this person or company. "
                                         "Check the spelling of the name and company, then try again.")
        with span("write_intro", "TOOL") as s:
            personal, warnings = write_intro(contact["role"], contact["org"], sources, person_found)
            s.set_attribute("rule_warnings", len(warnings))
            _io(s, {"role": contact["role"], "org": contact["org"], "sources": len(sources)}, personal)
        with span("fill_template", "TOOL") as s:
            subject, body = fill_template(contact["name"], contact["role"], topic, sender, personal)
            _io(s, out=body)
        return Result(stage, subject, body, sources=sources, person_found=person_found, warnings=warnings)

    if len(notes.split()) < MIN_NOTE_WORDS:
        return Result(stage, message="Not enough detail in the call notes. Add what they said about their work, then try again.")

    with span("extract_pain_points", "TOOL") as s:
        points = extract_pain_points(notes)
        _io(s, notes, points)
    if not points:
        return Result(stage, message="No clear pain points found in these notes. Add what they said was hard, then try again.")

    with span("draft_followup", "TOOL") as s:
        subject, body = draft_followup(contact, points, sender)
        _io(s, [p["pain_point"] for p in points], body)
    return Result(stage, subject, body, points)
