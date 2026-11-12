# Outreach Agent

**Live demo:** [outreach-agent-niha.streamlit.app](https://outreach-agent-niha.streamlit.app)

An AI assistant for student founders doing customer discovery. It researches each contact and their company, drafts a personalized first email, and turns call notes into a follow-up. A person reviews every draft before it is sent.

## The problem

I interviewed 8 people running customer discovery outreach through RIT's Kozel program and NSF I-Corps. The same pattern came up:

- Researching one contact (their role, their company's work) and writing a personal email took about **35 minutes**.
- Program targets were high (Kozel: 200 interviews in 6-7 weeks), and only about 10% of outreach got a reply, so volume decided results.
- First emails had to read as student research, with no product or startup mention.
- Follow-ups had to reference the actual conversation, or contacts forgot who they spoke with.

Lead finding was a real pain too, but tools like Apollo already cover it, so it is out of scope.

## How it works

| Step | What happens | Who does it |
| --- | --- | --- |
| Upload | Team's contact sheet (.xlsx or .csv) | User |
| Route | No call notes = first email, notes = follow-up | Code |
| Research | Two web searches: the person, then the company. A person result is kept only if it names both the person and their company | Tavily + code |
| First email | Model writes one paragraph on why this person's role at this company matters to the research, using one detail from the sources | LLM |
| Rule check | Blocks product, startup, demo or "we've built" language; rewrites once, then warns | Code |
| Follow-up | Approved template. The model fills one phrase from the call notes; code writes the opening line from the sheet's mode of contact | LLM + code |
| Review | Sources and pain points shown next to the draft; edit, rate, approve. Time on draft and % edited are logged | User |

**Design principle:** the model only writes the parts that need judgment. Greetings, intros, asks, signatures and anything with a compliance rule are written by code, so they cannot drift.

## What changed through testing

Each fix came from reviewing real drafts:

| Problem found | Fix |
| --- | --- |
| Follow-up changed facts ("over a weekend" became "for days") and pitched a solution | Model now sees exact quotes; facts and time words must match |
| Notes copied word for word, so emails read in third person ("he logs...") | Follow-up became a template with one AI-written phrase |
| Opening line assumed every contact was met at a conference | Opening line built by code from mode of contact (email, phone, in person) |
| Research matched a different person with the same name | Person results must name both the person and their company |
| First email read like a spec sheet (stacked numbers) | One detail per paragraph, at most one number |
| First email locked to a fixed template, which skipped the research that was the real bottleneck | Research-based first email with code-enforced rules |

## Evaluation

The **Evals** page runs the agent on 30 synthetic contacts and scores every draft:

| Eval | Type | Target |
| --- | --- | --- |
| Router accuracy | Code | 95% |
| First-email compliance (no product language, template lines intact) | Code | 100% |
| Follow-up format | Code | 100% |
| Refuses to draft from thin notes | Code | 100% |
| No wrong-person match | Code | 100% |
| Grounding: every fact traceable to sources or notes | LLM judge | 95% |
| Quality: reads like a person wrote it, respectful, plain words | LLM judge | 90% |

The LLM judge is checked against 10 hand-labeled drafts and must agree on at least 8 before its scores count.

## Tech

Python · Streamlit · Groq (`openai/gpt-oss-120b`, free tier) · Tavily web search · Arize Phoenix tracing (optional). Also runs on Anthropic or OpenAI by changing one setting.

## Run locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...
export TAVILY_API_KEY=tvly-...
streamlit run app.py
```

See `.streamlit/secrets.toml.example` for all settings.

## Files

- `app.py`: main page (upload, generate, review)
- `pages/2_Evals.py`: eval runner page
- `agent.py`: router, research, drafting and rule checks
- `evals.py`: the 7 evals and LLM judges
- `tracing.py`: Phoenix tracing
- `data/first_touch_template.txt`, `data/followup_template.txt`: the fixed parts of each email
- `data/contacts_sample.csv`: 30 sample contacts; `data/eval_labels.csv`: answer key for evals
- `scripts/make_sample_data.py`: regenerates the sample data

## Data note

All people in the sample data are invented. First-email contacts are placed at real companies so the research step has public information to find; contacts with call notes are at invented organizations so no made-up incident is tied to a real company. Research uses search snippets only and never scrapes LinkedIn.
