"""Outreach Agent MVP: Streamlit UI."""
import csv
import difflib
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
SAMPLE = ROOT / "data" / "contacts_sample.csv"
LOG = ROOT / "results" / "approvals.csv"
REQUIRED = ["name", "role", "org"]
JUNIOR_WORDS = ("junior", "intern", "student", "assistant", "associate", "technician")

st.set_page_config(page_title="Outreach Agent", page_icon="✉️", layout="centered")

# Secrets on Streamlit Cloud -> environment, before importing modules that read env vars.
for key in ("LLM_PROVIDER", "GROQ_API_KEY", "GROQ_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
            "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "PHOENIX_COLLECTOR_ENDPOINT", "PHOENIX_API_KEY"):
    try:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

from agent import key_env, model, provider, run_contact  # noqa: E402
from tracing import init_tracing  # noqa: E402

KEY_NAME = key_env()
KEY_LABEL = {"groq": "Groq", "anthropic": "Anthropic", "openai": "OpenAI"}[provider()]

tracing_on = init_tracing()
ss = st.session_state
ss.setdefault("result", None)
ss.setdefault("started", None)
ss.setdefault("log", [])

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    sender = st.text_input("Your name (signs the email)", "Niharika Mahesh Ahirekar")
    topic = st.text_input("Research topic (first-touch template)", "cleanroom contamination and equipment risk")
    if not os.getenv(KEY_NAME):
        key = st.text_input(f"{KEY_LABEL} API key", type="password", help="Not stored. Needed for follow-up drafts only.")
        if key:
            os.environ[KEY_NAME] = key
    st.caption(f"Model: {model()} · Tracing: {'on (Phoenix)' if tracing_on else 'off'}")

st.title("Outreach Agent")
st.caption("First-touch emails use the approved template. Follow-ups are drafted from your call notes. You review every email.")


# ---------- 1. Contact sheet ----------
def load_sheet(file) -> pd.DataFrame | None:
    if file is None:
        df = pd.read_csv(SAMPLE)
    elif file.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        st.error(f"Your sheet needs a name, role and org column. Missing: {', '.join(missing)}.")
        return None
    for col in ("contact_id", "email", "seniority", "call_notes"):
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")
    df["contact_id"] = [cid or f"R{i + 1:02d}" for i, cid in enumerate(df["contact_id"].astype(str))]
    return df


upload = st.file_uploader("Contact sheet (.xlsx or .csv)", type=["xlsx", "xls", "csv"],
                          help="Columns: name, role, org (required); email, seniority, call_notes (optional). "
                               "Leave empty to use the 30-contact synthetic sample.")
df = load_sheet(upload)
if df is None:
    st.stop()
if upload is None:
    st.caption("Using the synthetic sample sheet (30 invented contacts).")

labels = [f"{r.name} · {r.role}, {r.org}{'  ✎' if str(r.call_notes).strip() else ''}" for r in df.itertuples()]
idx = st.selectbox("Contact (✎ = has call notes)", range(len(df)), format_func=lambda i: labels[i])
contact = df.iloc[idx].to_dict()

# Warnings (P1 checks, cheap enough to include)
same_org = df[(df["org"] == contact["org"]) & (df.index != idx)]["name"].tolist()
if same_org:
    st.warning(f"{len(same_org)} other contact(s) at {contact['org']} in your sheet: {', '.join(same_org)}. "
               "Make sure you are reaching the right person.")
if any(w in f"{contact['seniority']} {contact['role']}".lower() for w in JUNIOR_WORDS):
    st.warning("This contact looks junior. Consider reaching a decision-maker at this org first.")

# ---------- 2. Notes ----------
notes = st.text_area("Call notes (leave empty for a first-touch email)", value=str(contact["call_notes"]),
                     height=120, key=f"notes_{contact['contact_id']}")
contact["call_notes"] = notes

stage_choice = st.radio("Email type", ["Auto", "First touch", "Follow-up"], horizontal=True,
                        help="Auto: no notes = first touch, notes = follow-up.")
override = {"Auto": None, "First touch": "first_touch", "Follow-up": "follow_up"}[stage_choice]

if st.button("Generate draft", type="primary"):
    if (override == "follow_up" or (override is None and notes.strip())) and not os.getenv(KEY_NAME):
        st.error(f"Add an {KEY_LABEL} API key in the sidebar to draft follow-ups.")
    else:
        try:
            with st.spinner("Drafting..."):
                ss.result = run_contact(contact, topic, sender, override)
                ss.started = time.time()
                ss.contact_id = contact["contact_id"]
        except Exception:
            ss.result = None
            st.error("Couldn't draft right now. Try again.")

# ---------- 3. Review ----------
res = ss.result
if res and ss.get("contact_id") == contact["contact_id"]:
    badge = "First touch · template" if res.stage == "first_touch" else "Follow-up · AI draft"
    st.subheader(badge)

    if res.message:
        st.info(res.message)
        st.stop()

    if res.pain_points:
        with st.expander("Pain points used (check each against the notes)", expanded=True):
            for p in res.pain_points:
                st.markdown(f"- **{p['pain_point']}**  \n  _from notes:_ \"{p['source_quote']}\"")

    subject = st.text_input("Subject", res.subject, key=f"subj_{contact['contact_id']}")
    body = st.text_area("Body", res.body, height=260, key=f"body_{contact['contact_id']}")
    words = len(body.split())
    color = "red" if words > 150 else "green"
    st.markdown(f"Word count: :{color}[{words}] / 150")

    rating = st.feedback("stars", key=f"rate_{contact['contact_id']}")

    if st.button("Approve & copy"):
        final = f"Subject: {subject}\n\n{body}"
        draft = f"Subject: {res.subject}\n\n{res.body}"
        edit_pct = round((1 - difflib.SequenceMatcher(None, draft, final).ratio()) * 100, 1)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "contact_id": contact["contact_id"],
            "stage": res.stage,
            "seconds_on_draft": round(time.time() - (ss.started or time.time()), 1),
            "edit_pct": edit_pct,
            "word_count": words,
            "rating": "" if rating is None else rating + 1,
        }
        ss.log.append(row)
        LOG.parent.mkdir(exist_ok=True)
        new = not LOG.exists()
        with open(LOG, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if new:
                w.writeheader()
            w.writerow(row)
        st.success(f"Approved. You changed {edit_pct}% of the draft. Copy it below.")
        st.code(final, language=None)

# ---------- Session log ----------
if ss.log:
    st.divider()
    st.caption("Approved this session")
    log_df = pd.DataFrame(ss.log)
    st.dataframe(log_df, hide_index=True)
    st.download_button("Download log (CSV)", log_df.to_csv(index=False), "approvals.csv", "text/csv")
