"""Evals page: run all checks on the 30 sample contacts and compare against targets."""
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="Evals · Outreach Agent", page_icon="📊", layout="wide")

for key in ("LLM_PROVIDER", "GROQ_API_KEY", "GROQ_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "OPENAI_API_KEY",
            "OPENAI_MODEL", "TAVILY_API_KEY", "PHOENIX_COLLECTOR_ENDPOINT", "PHOENIX_API_KEY"):
    try:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

import evals  # noqa: E402
from agent import key_env, model  # noqa: E402
from tracing import init_tracing  # noqa: E402

init_tracing()
ss = st.session_state

st.title("Evals")
st.caption(f"Runs the agent on all 30 sample contacts and scores every draft. Model: {model()}. "
           "Code checks are exact; judge checks use the same model at temperature 0.")

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

missing = [k for k in (key_env(), "TAVILY_API_KEY") if not os.getenv(k)]
if missing:
    st.error(f"Add {', '.join(missing)} in the app's Secrets before running evals.")
    st.stop()

version = st.text_input("Version label for this run", "v1", help="e.g. v1 before a prompt change, v2 after.")
st.caption("One run makes about 100 model calls and 36 web searches (well inside the free tiers) and takes 3-6 minutes.")

if st.button("Run evals", type="primary"):
    bar = st.progress(0.0, "Starting...")
    try:
        df = evals.run_evals(progress=lambda p, m: bar.progress(min(p, 1.0), m))
        df.insert(0, "version", version)
        df["your_label"] = ""  # fill in pass/fail by hand for the judge check
        ss.eval_df = df
        ss.eval_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    except Exception as e:
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
    st.info("Judge check: in the full results CSV, fill 'your_label' with pass/fail for 10 judge rows, "
            "then count how often you agree with the judge. You need 8 of 10 or more before the judge scores count.")
