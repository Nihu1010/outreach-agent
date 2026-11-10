# Outreach Agent

An AI drafting assistant for founders doing customer discovery. First-touch emails are filled into an advisor-approved template with plain code, so they cannot break outreach rules. Follow-ups are drafted by an LLM from call notes, grounded in pain points quoted from those notes. A person reviews every email before it is copied.

All contact data in this repo is synthetic.

## How it works

| Step | What happens | AI? |
| --- | --- | --- |
| Upload | Team's contact sheet (.xlsx/.csv), checked for name, role, org | No |
| Router | No call notes = first touch, notes = follow-up | No |
| `fill_template` | Inserts name, role, topic into the approved template | No |
| `extract_pain_points` | 1-3 pain points, each with a word-for-word quote from the notes; quotes not found in the notes are dropped | Yes |
| `draft_followup` | Under 150 words, one ask, only the extracted pain points | Yes |
| Review | Edit, rate 1-5, approve; time on draft and % edited are logged | No |

## Run locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...   # free; or LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY
streamlit run app.py
```

## Files

- `app.py` Streamlit UI
- `agent.py` router + 3 tools
- `tracing.py` Phoenix tracing (optional)
- `data/contacts_sample.csv` 30 synthetic contacts across 6 personas
- `data/eval_labels.csv` answer key for evals (not used by the app)
- `data/first_touch_template.txt` the approved first-touch template
- `scripts/make_sample_data.py` regenerates the sample data
