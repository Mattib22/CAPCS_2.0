# CASPER — Your Personal Thinking Companion

CASPER is a Socratic decision-making tool that uses AI to challenge your reasoning, surface cognitive biases, and help you think more clearly about real decisions.

## What it does

1. **Listens** — asks three focused questions to understand your decision and what's driving it
2. **Sparks** — identifies the most likely cognitive bias distorting your thinking and names it
3. **Challenges** — delivers a direct counterattack to stress-test your reasoning
4. **Tracks** — builds a reasoning profile across sessions so you can see your patterns over time

## Stack

- [Streamlit](https://streamlit.io) — UI framework
- [Google Gemini](https://ai.google.dev) (`google-generativeai`) — AI backbone
- [Supabase](https://supabase.com) — user data and session storage

## Setup

**1. Clone and install dependencies**

```bash
pip install -r requirements.txt
```

**2. Set environment variables**

```
GEMINI_API_KEY=your_google_ai_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

**3. Apply the database schema**

Run `supabase_schema_fix.sql` against your Supabase project to create the required tables (`users`, `profiles`, `sessions`, `bias_corrections`).

**4. Run the app**

```bash
streamlit run app.py
```

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Entry point — phase orchestration and all Streamlit UI |
| `ai_prompts.py` | All LLM calls (questions, bias detection, challenges) |
| `database.py` | Supabase read/write helpers |
| `user_model.py` | Onboarding questions, profile formatting, longitudinal context |
| `ui_helpers.py` | Reusable UI components and scroll utilities |
| `config.py` | Constants and CSS styles |

## Privacy

Users are identified by an anonymous hash of their username + PIN — the original credentials are never stored. All data is held on Supabase EU servers and can be deleted at any time from within the app.
