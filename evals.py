"""Phase 5 evals. Runs the agent on every sample contact and scores each draft.

Code checks (deterministic):  router accuracy, first-touch compliance, follow-up format,
                              refusal on thin notes, no wrong-person match.
LLM-judge checks:             grounding (every fact traceable to sources/notes), quality.

Run from the app's Evals page, or locally:  python evals.py
"""
import re
from pathlib import Path

import pandas as pd

import agent

ROOT = Path(__file__).parent
SAMPLE = ROOT / "data" / "contacts_sample.csv"
LABELS = ROOT / "data" / "eval_labels.csv"

TARGETS = {  # eval -> minimum pass rate (%) to ship
    "Router accuracy": 95,
    "First-touch compliance": 100,
    "Follow-up format": 100,
    "Refuses thin notes": 100,
    "No wrong-person match": 100,
    "Grounding (judge)": 95,
    "Quality (judge)": 90,
}

# ---------- LLM judges ----------

GROUNDING_FIRST = """You check a cold email paragraph for made-up facts.
Given facts (never count these as claims): the recipient's name, job title and company, which come from the user's contact sheet.
Not factual claims (ignore): general statements about what a job function involves, and the sender's own research interest ("our research is trying to understand...").
Check only specific facts about the company or the person: products, sites, locations, sizes, numbers, dates, customers, certifications, expansions, history.
A claim is supported if a numbered source states it or clearly paraphrases it. Fail if any such claim is missing from the sources, even if it is true in the real world.
Return JSON: {"verdict": "pass" | "fail", "unsupported": ["claim", ...], "reason": "one sentence"}"""

GROUNDING_FOLLOW = """You check one phrase from a follow-up email against the interviewer's call notes.
The phrase describes what the person talked about on the call. It may name their general work area (for example their job function or "cleanroom monitoring") without support.
Fail it if it adds details, numbers, time frames or causes that are not in the notes, or changes their meaning (e.g. "over a weekend" becoming "for days", or something nobody does becoming something they do).
Return JSON: {"verdict": "pass" | "fail", "unsupported": ["detail", ...], "reason": "one sentence"}"""

QUALITY_FIRST = """You review the middle paragraph of a cold email a graduate student will send to an industry professional.
The greeting, the student's introduction and the request for a 15-minute call are added separately by code. Do not fail the paragraph for not asking for a conversation.
Pass only if ALL are true:
1. Reads like a thoughtful person wrote it, not a report or press release: no spec lists, no stacked numbers, no stiff phrases like "the point at which".
2. Names one specific, relevant detail about the recipient's company or role.
3. Researcher tone: curious and respectful. No sales language, no flattery.
4. Clear and grammatical, no typos or misspelled names, about 45-80 words.
Return JSON: {"verdict": "pass" | "fail", "reason": "one sentence naming the main problem, or why it passes"}"""

QUALITY_FOLLOW = """You review one phrase inserted into this sentence of a follow-up email:
"What you described about ______, shaped a lot of how we've approached our research."
Pass only if ALL are true:
1. Reads naturally and grammatically in that sentence.
2. Respectful: describes a challenge in their work, never their mistake or gap.
3. Plain words, no jargon ("aggregation", "leverage", "optimize").
4. Specific enough that the person would recognize their own conversation.
Return JSON: {"verdict": "pass" | "fail", "reason": "one sentence"}"""


_status = None  # progress callback, set by run_evals
WAITS = [20, 40, 60]  # seconds to wait after a per-minute limit before each retry


def with_retry(fn, *args):
    """Call fn; on a per-minute rate limit, wait and retry. On a daily limit, stop the run."""
    import time
    for attempt in range(len(WAITS) + 1):
        try:
            return fn(*args)
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:400]}"
            low = msg.lower()
            if not ("429" in msg or "rate limit" in low):
                raise
            daily = "per day" in low or "(tpd)" in low or "(rpd)" in low
            if daily or attempt == len(WAITS):
                import os
                os.environ["ALLOW_FALLBACK"] = "1"
                raise RuntimeError(("Groq daily limit reached. " if daily else "Groq kept rate-limiting after 3 waits. ")
                                   + "The run was stopped and nothing was scored. Try again later. "
                                   + f"Provider message: {msg}") from e
            if _status:
                _status(f"Per-minute limit hit, waiting {WAITS[attempt]}s before retrying...")
            time.sleep(WAITS[attempt])


def judge_model() -> str:
    """The judge runs on a different model than the agent: its own rate limit, and no grading its own writing."""
    import os
    default = "openai/gpt-oss-20b" if agent.provider() == "groq" else ""
    return os.getenv("JUDGE_MODEL", default)


def judge(prompt: str, content: str) -> dict:
    out = with_retry(agent.llm_json, prompt, content, 0, judge_model())  # evals disable fallback, see run_evals
    out["verdict"] = str(out.get("verdict", "fail")).lower().strip()
    return out


# ---------- Helpers ----------

FIRST_TEMPLATE = (ROOT / "data" / "first_touch_template.txt").read_text()
FIXED_FIRST_LINES = [ln.strip() for ln in FIRST_TEMPLATE.splitlines()[1:]
                     if ln.strip() and "{" not in ln]  # template lines that must appear unchanged


def first_touch_paragraph(body: str) -> str:
    """The AI-written part: the block between the intro line and the ask."""
    blocks = [b.strip() for b in body.split("\n\n") if b.strip()]
    return blocks[2] if len(blocks) > 2 else ""


def followup_insight(body: str) -> str:
    m = re.search(r"What you described about (.+?), shaped a lot", body, re.S)
    return m.group(1).strip() if m else ""


def row(contact, eval_name, passed, detail="", judged=False):
    return {"contact_id": contact["contact_id"], "name": contact["name"], "eval": eval_name,
            "result": "pass" if passed else "fail", "detail": detail, "judge": "LLM" if judged else "code"}


# ---------- Run ----------

QUICK_SET = ["C01", "C03", "C05", "C09", "C12", "C17",   # first touch: 6 companies
             "C02", "C08", "C18", "C26",                 # follow-ups incl. the hardest grounding cases
             "C14"]                                      # thin notes


def run_evals(sender: str = "Niharika Ahirekar", progress=None, quick: bool = False) -> pd.DataFrame:
    import os
    global _status
    _status = (lambda m: progress(None, m)) if progress else None
    os.environ["ALLOW_FALLBACK"] = "0"  # every score in a run must come from the same model
    contacts = pd.read_csv(SAMPLE).fillna("")
    if quick:
        contacts = contacts[contacts["contact_id"].isin(QUICK_SET)]
    labels = pd.read_csv(LABELS).fillna("").set_index("contact_id")
    rows = []

    for i, c in enumerate(contacts.to_dict("records")):
        if progress:
            progress(i / len(contacts), f"{c['name']} ({i + 1}/{len(contacts)})")
        expected_stage = labels.loc[c["contact_id"], "stage"]

        # Router
        predicted = agent.route(c["call_notes"])
        rows.append(row(c, "Router accuracy", predicted == expected_stage, f"expected {expected_stage}, got {predicted}"))

        try:
            res = with_retry(agent.run_contact, c, "cleanroom operations", sender)
        except RuntimeError:
            raise  # rate limit that waiting could not fix: stop the run
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:300]}"
            rows.append(row(c, "Agent error", False, msg))
            continue

        # Thin notes: the agent must refuse rather than guess
        if expected_stage == "follow_up" and len(c["call_notes"].split()) < agent.MIN_NOTE_WORDS:
            rows.append(row(c, "Refuses thin notes", bool(res.message) and not res.body, res.message or "drafted anyway"))
            continue
        if res.message:
            rows.append(row(c, "Grounding (judge)", False, f"no draft: {res.message}"))
            continue

        if res.stage == "first_touch":
            para = first_touch_paragraph(res.body)
            words = len(para.split())
            missing = [ln for ln in FIXED_FIRST_LINES if ln.replace("{sender}", sender) not in res.body]
            banned = sorted({m.group(0) for m in agent.BANNED.finditer(para)})
            ok = not missing and not banned and 35 <= words <= 85
            rows.append(row(c, "First-touch compliance", ok,
                            f"{words} words" + (f"; banned: {banned}" if banned else "") + (f"; template changed" if missing else "")))
            # Every sample person is invented, so any 'person found' is a same-name stranger.
            rows.append(row(c, "No wrong-person match", not res.person_found,
                            "matched a real person" if res.person_found else "none matched"))
            src = "\n\n".join(f"[{s['n']}] {s['title']}\n{s.get('content', '')}" for s in res.sources)
            g = judge(GROUNDING_FIRST, f"Sources:\n{src}\n\nParagraph:\n{para}")
            rows.append(row(c, "Grounding (judge)", g["verdict"] == "pass",
                            g.get("reason", "") + (f" Unsupported: {g.get('unsupported')}" if g.get("unsupported") else ""), True))
            q = judge(QUALITY_FIRST, para)
            rows.append(row(c, "Quality (judge)", q["verdict"] == "pass", q.get("reason", ""), True))
        else:
            insight = followup_insight(res.body)
            words = len(insight.split())
            ok = bool(insight) and 6 <= words <= 30 and res.body.startswith("Hi ")
            rows.append(row(c, "Follow-up format", ok, f"{words}-word phrase"))
            g = judge(GROUNDING_FOLLOW, f"Call notes:\n{c['call_notes']}\n\nPhrase:\n{insight}")
            rows.append(row(c, "Grounding (judge)", g["verdict"] == "pass",
                            g.get("reason", "") + (f" Unsupported: {g.get('unsupported')}" if g.get("unsupported") else ""), True))
            q = judge(QUALITY_FOLLOW, insight)
            rows.append(row(c, "Quality (judge)", q["verdict"] == "pass", q.get("reason", ""), True))

        rows[-1]["draft"] = res.body  # keep the full draft once per contact for review

    if progress:
        progress(1.0, "Done")
    os.environ["ALLOW_FALLBACK"] = "1"
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    s = (df.assign(passed=df["result"].eq("pass"))
           .groupby("eval")["passed"].agg(["sum", "count"]).reset_index())
    s["pass_rate_%"] = (100 * s["sum"] / s["count"]).round(0).astype(int)
    s["target_%"] = s["eval"].map(TARGETS)
    s["status"] = ["not a score: agent crashed" if pd.isna(t) else ("meets target" if r >= t else "below target")
                   for r, t in zip(s["pass_rate_%"], s["target_%"])]
    order = {k: i for i, k in enumerate(list(TARGETS) + ["Agent error"])}
    return (s.rename(columns={"sum": "passed", "count": "cases"})
             .sort_values("eval", key=lambda x: x.map(order)).reset_index(drop=True))


if __name__ == "__main__":
    results = run_evals(progress=lambda p, m: print(f"{p:5.0%}  {m}"))
    print(summarize(results).to_string(index=False))
    results.to_csv(ROOT / "results" / "eval_results.csv", index=False)
