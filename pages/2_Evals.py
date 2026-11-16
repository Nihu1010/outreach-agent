"""Evals page.

Everyone sees the saved results of every scored run (read-only, no API calls).
Running new evals is limited to the project owner (EVAL_PASSWORD), because one run uses most of a day's
free-tier model budget.
"""
import hmac
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HISTORY = ROOT / "data" / "eval_history"

st.set_page_config(page_title="Evals · Outreach Agent", page_icon="📊", layout="wide")

for key in ("LLM_PROVIDER", "GROQ_API_KEY", "GROQ_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "OPENAI_API_KEY",
            "OPENAI_MODEL", "TAVILY_API_KEY", "JUDGE_MODEL", "PHOENIX_PROJECT", "PHOENIX_EVAL_PROJECT",
            "PHOENIX_COLLECTOR_ENDPOINT", "PHOENIX_API_KEY", "EVAL_PASSWORD"):
    try:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

import evals  # noqa: E402

ss = st.session_state

st.title("Evals")
st.caption("Every draft the agent writes is scored by 7 checks: 5 exact code checks and 2 LLM judges. "
           "Below are the saved results of every full run on the 30 sample contacts.")

with st.expander("What each eval checks", expanded=False):
    st.markdown("""
| Eval | Type | Pass rule | Target |
|---|---|---|---|
| Router accuracy | Code | No notes = first touch, notes = follow-up, matches the answer key | 95% |
| First-touch compliance | Code | Fixed template lines unchanged, no product/startup words, paragraph 35-85 words | 100% |
| Follow-up format | Code | AI phrase present, 6-30 words, greeting intact | 100% |
| Refuses thin notes | Code | Notes too short = no draft, asks for more detail | 100% |
| No wrong-person match | Code | Sample people are invented, so research must not claim to find them | 100% |
| Grounding | LLM judge | Every company/person fact is in the sources; every follow-up detail is in the notes | 95% |
| Quality | LLM judge | Reads like a person wrote it, one specific detail, respectful, plain words | 90% |
""")


# ---------- Saved results (public, no API calls) ----------
@st.cache_data
def load_history():
    manifest = json.loads((HISTORY / "manifest.json").read_text())
    runs = []
    for m in manifest:
        df = pd.read_csv(HISTORY / m["file"]).fillna("")
        runs.append({**m, "df": df, "summary": evals.summarize(df)})
    return runs


runs = load_history() if (HISTORY / "manifest.json").exists() else []

if runs:
    table = pd.DataFrame({"eval": list(evals.TARGETS)})
    table["target"] = table["eval"].map(lambda e: f"{evals.TARGETS[e]}%")
    for r in runs:
        rates = dict(zip(r["summary"]["eval"], r["summary"]["pass_rate_%"]))
        table[r["label"]] = table["eval"].map(lambda e: f"{rates[e]}%" if e in rates else "")
    st.subheader("Pass rate by run")
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Judge setup changed after the baseline (fixed prompts, separate judge model), so compare the "
               "code-checked rows directly and read judge rows with that in mind. Round 2 was run twice on the same "
               "code to measure run-to-run variation.")

    with st.expander("Notes on each run"):
        for r in runs:
            st.markdown(f"**{r['label']}** · {r['date']} · build {r['build']}: {r['note']}")

    st.subheader("Explore a run")
    pick = st.selectbox("Run", [r["label"] for r in runs], index=len(runs) - 1)
    run = next(r for r in runs if r["label"] == pick)
    fails = run["df"][run["df"]["result"] == "fail"]
    st.markdown(f"**Failures ({len(fails)})**, with the judge's or check's reason:")
    st.dataframe(fails[["name", "eval", "judge", "detail"]], hide_index=True, width="stretch")
    drafts = run["df"][run["df"]["draft"] != ""][["name", "draft"]]
    with st.expander(f"All {len(drafts)} drafts from this run"):
        st.dataframe(drafts, hide_index=True, width="stretch")
else:
    st.info("No saved eval runs yet.")


# ---------- Run new evals (owner only) ----------
st.divider()
st.subheader("Run new evals")
password = os.getenv("EVAL_PASSWORD", "")
if not password:
    st.caption("Running evals is turned off on this deployment.")
    st.stop()
if not ss.get("eval_unlocked"):
    st.caption("Running evals is limited to the project owner, because one full run uses most of a day's "
               "free-tier model budget. The saved results above are from real runs.")
    entered = st.text_input("Owner password", type="password")
    if entered and hmac.compare_digest(entered, password):
        ss.eval_unlocked = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    st.stop()

from agent import key_env, model  # noqa: E402
from tracing import init_tracing, use_project  # noqa: E402

init_tracing()
st.caption(f"Agent model: {model()}. Judge model: {evals.judge_model() or model()}. "
           "Judges run at temperature 0.")

missing = [k for k in (key_env(), "TAVILY_API_KEY") if not os.getenv(k)]
if missing:
    st.error(f"Add {', '.join(missing)} in the app's Secrets before running evals.")
    st.stop()

version = st.text_input("Version label for this run", "round3-a", help="e.g. round3-a, round3-b")
mode = st.radio("Run size", ["Quick: 11 contacts (for testing a fix)", "Full: 30 contacts (official score)"])
quick = mode.startswith("Quick")
st.caption("Quick run: about 2-3 minutes. Full run: about 8-15 minutes. Keep this tab open until it finishes; "
           "refreshing the page loses the results.")

if st.button("Run evals", type="primary"):
    bar = st.progress(0.0, "Starting...")
    try:
        state = {"p": 0.0}

        def update(p, m):
            if p is not None:
                state["p"] = min(p, 1.0)
            bar.progress(state["p"], m)

        with use_project():  # eval traces go to their own Phoenix project
            df = evals.run_evals(progress=update, quick=quick)
        df.insert(0, "version", version + (" (quick)" if quick else ""))
        df["your_label"] = ""  # fill in pass/fail by hand for the judge check
        ss.eval_df = df
        ss.eval_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        os.environ["ALLOW_FALLBACK"] = "1"
        st.error("The eval run stopped early.")
        st.code(f"{type(e).__name__}: {str(e)[:500]}", language=None)

if "eval_df" in ss:
    df = ss.eval_df
    summary = evals.summarize(df)
    st.subheader(f"Results · {df['version'].iloc[0]} · {ss.eval_time}")
    st.dataframe(summary, hide_index=True, width="stretch")

    fails = df[df["result"] == "fail"]
    st.subheader(f"Failures ({len(fails)})")
    st.dataframe(fails[["name", "eval", "judge", "detail"]], hide_index=True, width="stretch")

    with st.expander("All results and drafts"):
        st.dataframe(df, hide_index=True, width="stretch")

    stamp = ss.eval_time.replace(" ", "_").replace(":", "")
    c1, c2 = st.columns(2)
    c1.download_button("Download full results (CSV)", df.to_csv(index=False),
                       f"evals_{df['version'].iloc[0]}_{stamp}.csv", "text/csv")
    c2.download_button("Download summary (CSV)", summary.to_csv(index=False),
                       f"eval_summary_{df['version'].iloc[0]}_{stamp}.csv", "text/csv")
    st.info("To add this run to the public results: send the full results CSV to be added to "
            "data/eval_history with a manifest entry. Judge check: fill 'your_label' with pass/fail for 10 judge "
            "rows and count agreement with the judge (8 of 10 or more before judge scores count).")
