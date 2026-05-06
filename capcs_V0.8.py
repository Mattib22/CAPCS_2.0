"""
CAPCS — Co-Adaptive Predictive Cognitive System
Prototype v0.8

Major architectural change: conversational loop
- Each round has sub-states: present → follow-up on perspective → respond → shifted check
- Full conversation history fed into every AI call
- Confidence only asked when thinking has shifted
- No repeated "which option" forms — flows like a real conversation
- Follow-up questions on new perspective before user responds
- Each round builds on previous answers, biases, and shifts

Setup:
    .streamlit/secrets.toml: GEMINI_API_KEY = "your-key-here"
    Or Streamlit Cloud: App Settings → Secrets

Requirements:
    pip install google-generativeai streamlit
"""

import streamlit as st
import json
import os
from datetime import datetime
from collections import Counter
import google.generativeai as genai

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAPCS — Cognitive Decision Assistant",
    page_icon="⚡",
    layout="centered"
)

# ── STYLES ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #FDFBF7; }
    .phase-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #9A7B3A;
        margin: 16px 0 4px 0;
    }
    .insight-box {
        background: #F5EED8;
        border-left: 3px solid #9A7B3A;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 6px 0 12px 0;
        font-size: 14px;
        line-height: 1.7;
    }
    .info-box {
        background: #EEF2FF;
        border-left: 3px solid #4A6FA5;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 6px 0 12px 0;
        font-size: 13px;
        line-height: 1.7;
    }
    .warning-box {
        background: #FFF4E5;
        border-left: 3px solid #E07B00;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 6px 0 12px 0;
        font-size: 14px;
        line-height: 1.7;
    }
    .perspective-box {
        background: #E8F5E9;
        border-left: 3px solid #2E7D52;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 6px 0 12px 0;
        font-size: 14px;
        line-height: 1.7;
    }
    .question-box {
        background: #F3E5F5;
        border-left: 3px solid #7B1FA2;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 6px 0 12px 0;
        font-size: 15px;
        line-height: 1.7;
        font-style: italic;
    }
    .answer-box {
        background: #F8F9FA;
        border-left: 3px solid #6C757D;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 6px 0 12px 0;
        font-size: 14px;
        line-height: 1.7;
    }
    .confidence-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 80
MAX_ROUNDS = 5
LOG_FILE = "capcs_log.json"
PROFILE_FILE = "capcs_profile.json"

# ── API KEY ────────────────────────────────────────────────────────────────────
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")

API_KEY = get_api_key()

# ── SESSION STATE ──────────────────────────────────────────────────────────────
defaults = {
    "phase": "onboarding",          # onboarding → input → challenge → report
    "sub_state": "present",         # present → followup → respond → shifted → next
    "session_log": [],
    "current_decision": {},
    "onboarding_step": 0,
    "onboarding_answers": {},
    "followup_exchanges": [],        # list of {question, answer} within a round
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── FILE HELPERS ───────────────────────────────────────────────────────────────
def load_file(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {} if path == PROFILE_FILE else []

def save_file(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def save_log(entry):
    history = load_file(LOG_FILE)
    history.append(entry)
    save_file(LOG_FILE, history)

# ── UI HELPERS ─────────────────────────────────────────────────────────────────
def confidence_color(score):
    if score >= 75: return "#d4edda", "#155724"
    if score >= 40: return "#fff3cd", "#856404"
    return "#f8d7da", "#721c24"

def badge(score):
    bg, fg = confidence_color(score)
    st.markdown(
        f'<span class="confidence-badge" style="background:{bg};color:{fg}">'
        f'Confidence: {score}%</span>',
        unsafe_allow_html=True
    )

def label(text):
    st.markdown(f'<div class="phase-label">{text}</div>', unsafe_allow_html=True)

def box(text, style="insight"):
    css_map = {
        "insight": "insight-box", "info": "info-box", "warning": "warning-box",
        "perspective": "perspective-box", "question": "question-box", "answer": "answer-box"
    }
    st.markdown(f'<div class="{css_map.get(style, "insight-box")}">{text}</div>', unsafe_allow_html=True)

# ── PROFILE FORMATTER ──────────────────────────────────────────────────────────
def format_profile(profile):
    if not profile:
        return "No user profile available."
    fields = {
        "education": "Education", "profession": "Profession",
        "decision_style": "Decision-making style", "risk_tolerance": "Risk tolerance",
        "values": "Core values", "passions": "Passions", "hobbies": "Hobbies",
        "social_context": "Social decision style", "known_bias": "Known personal blind spot",
    }
    return "\n".join(f"- {lbl}: {profile[key]}" for key, lbl in fields.items() if profile.get(key))

# ── CONVERSATION HISTORY BUILDER ───────────────────────────────────────────────
def build_history(rounds_log):
    """Build a readable conversation history for the AI prompt."""
    if not rounds_log:
        return "No previous rounds yet."
    lines = []
    for r in rounds_log:
        lines.append(f"--- Round {r['round']} ---")
        if r.get("bias"):
            lines.append(f"Bias identified: {r['bias']}")
        if r.get("perspective"):
            lines.append(f"Perspective offered: {r['perspective']}")
        if r.get("followups"):
            for fq in r["followups"]:
                lines.append(f"User asked about perspective: {fq.get('question', '')}")
                lines.append(f"AI answered: {fq.get('answer', '')}")
        if r.get("answer"):
            lines.append(f"User's answer to challenge: {r['answer']}")
        if r.get("shifted") is not None:
            lines.append(f"Thinking shifted: {'Yes' if r['shifted'] else 'No'}")
        if r.get("how_shifted"):
            lines.append(f"How it shifted: {r['how_shifted']}")
        if r.get("leaning"):
            lines.append(f"Now leaning towards: {r['leaning']}")
        if r.get("reasoning"):
            lines.append(f"Reasoning: {r['reasoning']}")
        if r.get("confidence"):
            lines.append(f"Confidence after round: {r['confidence']}%")
    return "\n".join(lines)

# ── AI CALL HELPER ─────────────────────────────────────────────────────────────
def ask_ai(prompt, max_tokens=800):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.8, "max_output_tokens": max_tokens}
    )
    text = response.text.strip()
    text = text.lstrip("*").rstrip("*").strip()
    text = text.lstrip("#").strip()
    return text

# ── AI CALLS ───────────────────────────────────────────────────────────────────

def get_bias(decision, options, leaning, confidence, profile, history):
    profile_text = format_profile(profile)
    prompt = f"""You are a cognitive scientist analysing someone's decision.

CONVERSATION SO FAR:
{history}

Identify the SINGLE most relevant cognitive bias affecting their thinking RIGHT NOW — different from any already identified above.

Write ONE complete sentence of AT MOST 40 words:
"[Bias name] — because [specific profile evidence], you may be [how this manifests in this specific decision]."

Rules:
- Max 40 words. Write a complete sentence.
- Must cite specific profile data (profession, education, risk tolerance, values, or known blind spot).
- Must NOT repeat a bias already identified in the conversation history above.
- Output only the sentence. No preamble.

USER PROFILE:
{profile_text}

DECISION: {decision}
OPTIONS: {options}
CURRENTLY LEANING: {leaning if leaning else "not specified"}
CONFIDENCE: {confidence}%"""
    return ask_ai(prompt, max_tokens=500)

def get_explanation(bias_text, decision, profile, history):
    profile_text = format_profile(profile)
    prompt = f"""You are explaining a cognitive bias to someone facing a decision.

CONVERSATION SO FAR:
{history}

The bias: {bias_text}

Write AT MOST 100 words covering in order:
1. What this bias is in general (1 sentence)
2. Why it was detected given THIS person's profile (1 sentence, cite the evidence)
3. How it may be distorting this specific decision (1 sentence)
4. One concrete way to counter it right now (1 sentence)

Rules:
- Max 100 words total. Complete sentences only.
- Plain prose, no bullet points, no markdown.
- Output only the paragraph.

USER PROFILE:
{profile_text}

DECISION: {decision}"""
    return ask_ai(prompt, max_tokens=800)

def get_perspective(decision, options, leaning, profile, bias_text, history):
    profile_text = format_profile(profile)
    prompt = f"""You are a Socratic thinking partner helping someone break out of their usual thinking.

CONVERSATION SO FAR:
{history}

Their current bias: {bias_text}
Currently leaning: {leaning if leaning else "unspecified"}

Offer ONE of these (choose the most useful):
(a) A concrete alternative solution NOT in their existing options — named specifically with a brief reason why it helps
(b) A reframe of the decision itself that challenges a hidden assumption in their current leaning

Write AT MOST 80 words in 2-3 complete sentences:
- Sentence 1: The alternative or reframe, stated specifically
- Sentences 2-3: Why this helps — how it addresses the bias or gap in their thinking

Rules:
- Max 80 words. Complete sentences, no truncation.
- Must be DIFFERENT from existing options: {options}
- Must NOT repeat a perspective already offered in the conversation history.
- Never vague ("consider other options" is banned). Be specific and named.
- Calibrated to profile — outside their comfort zone but realistic.
- Output only the perspective. No preamble.

USER PROFILE:
{profile_text}

DECISION: {decision}"""
    return ask_ai(prompt, max_tokens=600)

def get_question(decision, leaning, perspective_text, profile, history):
    profile_text = format_profile(profile)
    prompt = f"""You are a Socratic thinking partner.

CONVERSATION SO FAR:
{history}

User is leaning: {leaning if leaning else "unspecified"}
New perspective just offered: {perspective_text}

Ask ONE Socratic question of AT MOST 35 words that either:
- Probes the assumptions behind their current leaning, OR
- Asks them to engage with the new perspective

Rules:
- Max 35 words. One complete question ending with ?
- Cannot be answered yes/no.
- Specific to this decision.
- Must NOT repeat a question already asked in the conversation history.
- Output only the question.

USER PROFILE:
{profile_text}

DECISION: {decision}"""
    return ask_ai(prompt, max_tokens=400)

def get_followup_answer(perspective_text, user_question, decision, profile, history):
    """Answer the user's follow-up question about the new perspective."""
    profile_text = format_profile(profile)
    prompt = f"""You are a Socratic thinking partner. The user asked a follow-up question about a new perspective that was offered to them.

CONVERSATION SO FAR:
{history}

Perspective offered: {perspective_text}
User's question: {user_question}

Answer their question clearly in AT MOST 80 words. Be specific and helpful.
Your goal is to help them understand the perspective well enough to engage with it seriously.
Do not just repeat the perspective — add new information or clarification.
Complete sentences, plain prose, no markdown.
Output only the answer.

USER PROFILE:
{profile_text}

DECISION: {decision}"""
    return ask_ai(prompt, max_tokens=600)

def get_shift_followup(decision, leaning_before, answer, profile, history):
    """Generate a conversational follow-up when thinking HAS shifted."""
    profile_text = format_profile(profile)
    prompt = f"""You are a Socratic thinking partner. The user just said their thinking has shifted.

CONVERSATION SO FAR:
{history}

Their answer: {answer}
They were leaning towards: {leaning_before}

Write ONE short, warm, conversational question (AT MOST 25 words) asking them:
- How has their thinking changed, and which option are they now considering?

Must feel like a natural follow-up in a conversation, not a form field.
Output only the question.

DECISION: {decision}"""
    return ask_ai(prompt, max_tokens=300)

def get_reasoning_followup(decision, leaning_now, profile, history):
    """Generate a conversational why-this-option question."""
    profile_text = format_profile(profile)
    prompt = f"""You are a Socratic thinking partner.

CONVERSATION SO FAR:
{history}

The user is now leaning towards: {leaning_now}

Write ONE short conversational question (AT MOST 20 words) asking them WHY this option — what's their reasoning?

Must feel natural, not like a form.
Output only the question.

DECISION: {decision}"""
    return ask_ai(prompt, max_tokens=200)

# ── ONBOARDING QUESTIONS ───────────────────────────────────────────────────────
QUESTIONS = [
    {"key": "education", "label": "What is your educational background?",
     "placeholder": "e.g. BSc Psychology, self-taught, high school + vocational training...", "type": "text"},
    {"key": "profession", "label": "What is your current or most recent profession?",
     "placeholder": "e.g. Marketing manager, freelance designer, student...", "type": "text"},
    {"key": "decision_style", "label": "How do you tend to make decisions?",
     "placeholder": "e.g. I deliberate / I trust gut / I consult others first...", "type": "text"},
    {"key": "risk_tolerance", "label": "How comfortable are you with uncertainty and risk?",
     "options": [
         "Very comfortable — I enjoy taking risks",
         "Comfortable — I take calculated risks",
         "Neutral — it depends on the situation",
         "Uncomfortable — I prefer stability",
         "Very uncomfortable — I avoid risk whenever possible"
     ], "type": "select"},
    {"key": "values", "label": "What matters most to you when making an important decision?",
     "placeholder": "e.g. Security, growth, freedom, relationships, money, creativity...", "type": "text"},
    {"key": "passions", "label": "What are you most passionate about in life?",
     "placeholder": "e.g. Technology, art, science, travel, social impact...", "type": "text"},
    {"key": "hobbies", "label": "What do you do in your free time?",
     "placeholder": "e.g. Reading, sport, music, gaming, cooking...", "type": "text"},
    {"key": "social_context", "label": "When facing an important decision, you tend to:",
     "options": [
         "Decide alone — I trust my own judgement",
         "Consult a few trusted people",
         "Discuss widely — the more perspectives the better",
         "It varies depending on the decision"
     ], "type": "select"},
    {"key": "known_bias", "label": "Is there a pattern in your past decisions you'd like to change?",
     "placeholder": "e.g. I overthink / I act too impulsively / I always play it safe...", "type": "text"},
]

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ CAPCS")
    st.markdown("*Co-Adaptive Predictive Cognitive System*")
    st.divider()

    profile = load_file(PROFILE_FILE)
    if profile:
        st.markdown("**Your profile**")
        box(
            f'🎓 {profile.get("education", "—")}<br>'
            f'💼 {profile.get("profession", "—")}<br>'
            f'⚡ Risk: {profile.get("risk_tolerance", "—")}',
            style="info"
        )
        if st.button("✏️ Update profile", use_container_width=True):
            st.session_state.phase = "onboarding"
            st.session_state.onboarding_step = 0
            st.session_state.onboarding_answers = {}
            st.rerun()
    else:
        st.caption("Complete onboarding to personalise your experience.")

    st.divider()
    history_data = load_file(LOG_FILE)
    completed = [h for h in history_data if "confidence_shift" in h]
    if completed:
        st.markdown(f"**Sessions:** {len(completed)}")
        avg = sum(h["confidence_shift"] for h in completed) / len(completed)
        st.markdown(f"**Avg. confidence shift:** {avg:+.1f}%")

        all_biases = []
        for h in completed:
            for r in h.get("rounds_log", []):
                b = r.get("bias", "").split("—")[0].strip()
                if b:
                    all_biases.append(b[:40])
        if all_biases:
            top = Counter(all_biases).most_common(1)[0]
            st.caption(f"Most recurring: {top[0]} ({top[1]}x)")

    st.divider()
    if st.button("🗑 Clear all data", use_container_width=True):
        for f in [LOG_FILE, PROFILE_FILE]:
            if os.path.exists(f):
                os.remove(f)
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("## ⚡ CAPCS")
st.markdown("*A thinking partner, not an answer engine.*")
st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# PHASE 0 — ONBOARDING
# ════════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "onboarding":

    total = len(QUESTIONS)
    step = st.session_state.onboarding_step

    st.progress(step / total)
    st.caption(f"Question {step + 1} of {total}")
    st.markdown("")

    if step == 0:
        st.markdown("### Before we start")
        box("I'd like to learn a little about you. This helps me give you challenges "
            "that are personally relevant. Takes about 2 minutes — only needs to be done once.", style="info")
        st.markdown("")

    q = QUESTIONS[step]
    st.markdown(f"**{q['label']}**")

    if q["type"] == "text":
        answer = st.text_input("answer", label_visibility="collapsed",
            placeholder=q.get("placeholder", ""), key=f"q_{step}",
            value=st.session_state.onboarding_answers.get(q["key"], ""))
    else:
        prev = st.session_state.onboarding_answers.get(q["key"], q["options"][0])
        idx = q["options"].index(prev) if prev in q["options"] else 0
        answer = st.radio("answer", label_visibility="collapsed",
            options=q["options"], key=f"q_{step}", index=idx)

    st.markdown("")
    col1, col2 = st.columns([1, 2])
    with col1:
        if step > 0:
            if st.button("← Back", use_container_width=True):
                st.session_state.onboarding_step -= 1
                st.rerun()
    with col2:
        btn = "Next →" if step < total - 1 else "Start CAPCS →"
        if st.button(btn, type="primary", use_container_width=True):
            if q["type"] == "text" and not answer.strip():
                st.warning("Please add a short answer to continue.")
            else:
                st.session_state.onboarding_answers[q["key"]] = answer
                if step < total - 1:
                    st.session_state.onboarding_step += 1
                    st.rerun()
                else:
                    save_file(PROFILE_FILE, {**st.session_state.onboarding_answers,
                                              "completed_at": datetime.now().isoformat()})
                    st.session_state.phase = "input"
                    st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DECISION INPUT
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "input":

    profile = load_file(PROFILE_FILE)
    if not profile:
        st.session_state.phase = "onboarding"
        st.rerun()

    box("Tell me about the decision you're facing. I just need enough context "
        "to start challenging your thinking.", style="info")

    label("Step 1 — The decision")
    decision = st.text_area("What decision are you working through?",
        placeholder="e.g. I'm thinking of leaving my current job for a startup opportunity...", height=100)

    label("Step 2 — The options")
    st.caption("List the options you're aware of right now.")
    options = st.text_area("What options are you currently considering?",
        placeholder="e.g. Stay in current role / Join the startup / Take 3 months to explore...", height=80)

    label("Step 3 — Your current leaning")
    st.caption("Not a commitment — just where your mind is right now.")
    leaning = st.text_input("Which option are you currently leaning towards?",
        placeholder="e.g. Joining the startup...")

    label("Step 4 — Your confidence")
    confidence = st.slider("Confidence", min_value=0, max_value=100, value=50, step=5,
        label_visibility="collapsed", help="0 = completely unsure, 100 = totally certain")
    badge(confidence)

    st.markdown("")
    if st.button("→ Challenge my thinking", type="primary", use_container_width=True):
        if not decision.strip():
            st.error("Please describe your decision.")
        elif not options.strip():
            st.error("Please list the options you are considering.")
        elif not API_KEY:
            st.error("API key not configured. Add GEMINI_API_KEY to Streamlit secrets.")
        else:
            st.session_state.current_decision = {
                "decision": decision, "options": options, "leaning": leaning,
                "confidence_before": confidence, "confidence_start": confidence,
                "timestamp": datetime.now().isoformat(), "rounds": 0, "rounds_log": []
            }
            st.session_state.sub_state = "present"
            st.session_state.followup_exchanges = []
            st.session_state.phase = "challenge"
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CHALLENGE (conversational loop with sub-states)
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "challenge":

    cd = st.session_state.current_decision
    profile = load_file(PROFILE_FILE)
    round_num = cd.get("rounds", 0) + 1
    sub_state = st.session_state.sub_state
    history_text = build_history(cd.get("rounds_log", []))

    # Progress
    st.progress(round_num / MAX_ROUNDS)
    label(f"Round {round_num} of {MAX_ROUNDS}")
    st.markdown(f"**Decision:** {cd['decision']}")
    st.caption(f"Options: {cd['options']}")
    if cd.get("leaning"):
        st.caption(f"Currently leaning: **{cd['leaning']}**")
    badge(cd["confidence_before"])
    st.divider()

    # ── SUB-STATE: PRESENT ────────────────────────────────────────────────────
    # Generate and show the 4 outputs, then offer follow-up
    if sub_state == "present":

        # Generate outputs if not done yet
        if "bias_text" not in cd:
            try:
                with st.spinner("Identifying bias..."):
                    cd["bias_text"] = get_bias(cd["decision"], cd["options"],
                        cd.get("leaning", ""), cd["confidence_before"], profile, history_text)
                    st.session_state.current_decision = cd

                with st.spinner("Explaining..."):
                    cd["explanation_text"] = get_explanation(cd["bias_text"],
                        cd["decision"], profile, history_text)
                    st.session_state.current_decision = cd

                with st.spinner("Finding a new perspective..."):
                    cd["perspective_text"] = get_perspective(cd["decision"],
                        cd["options"], cd.get("leaning", ""), profile,
                        cd["bias_text"], history_text)
                    st.session_state.current_decision = cd

                with st.spinner("Crafting a question..."):
                    cd["question_text"] = get_question(cd["decision"],
                        cd.get("leaning", ""), cd["perspective_text"], profile, history_text)
                    st.session_state.current_decision = cd

            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        # Show outputs
        label("⚠️ Bias detected")
        box(cd.get("bias_text", "—"), style="warning")

        label("📖 What this is, why it was detected, and how to counter it")
        box(cd.get("explanation_text", "—"), style="insight")

        label("💡 A perspective outside your usual thinking")
        box(cd.get("perspective_text", "—"), style="perspective")

        label("❓ Consider this")
        box(cd.get("question_text", "—"), style="question")

        st.divider()

        # Show any previous follow-up exchanges this round
        if st.session_state.followup_exchanges:
            for exc in st.session_state.followup_exchanges:
                label("You asked")
                box(exc["question"], style="answer")
                label("CAPCS")
                box(exc["answer"], style="info")

        # Offer follow-up or proceed
        box("Do you want to ask a follow-up question about this perspective before responding? "
            "Or are you ready to answer?", style="info")

        followup_q = st.text_input(
            "Your follow-up question (optional):",
            placeholder="e.g. What do you mean by that? How would this work in practice?",
            key=f"followup_input_{round_num}_{len(st.session_state.followup_exchanges)}"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("❓ Ask follow-up", use_container_width=True):
                if followup_q.strip():
                    with st.spinner("Answering..."):
                        ans = get_followup_answer(
                            cd["perspective_text"], followup_q,
                            cd["decision"], profile, history_text
                        )
                    st.session_state.followup_exchanges.append({
                        "question": followup_q, "answer": ans
                    })
                    st.rerun()
                else:
                    st.warning("Please type a question first.")
        with col2:
            if st.button("→ I'm ready to answer", type="primary", use_container_width=True):
                st.session_state.sub_state = "respond"
                st.rerun()

    # ── SUB-STATE: RESPOND ────────────────────────────────────────────────────
    # User answers the Socratic question
    elif sub_state == "respond":

        # Show what was presented (compact)
        label("⚠️ Bias detected")
        box(cd.get("bias_text", "—"), style="warning")
        label("💡 New perspective")
        box(cd.get("perspective_text", "—"), style="perspective")
        label("❓ The question")
        box(cd.get("question_text", "—"), style="question")

        # Show follow-up exchanges if any
        if st.session_state.followup_exchanges:
            for exc in st.session_state.followup_exchanges:
                label("Your follow-up")
                box(exc["question"], style="answer")
                label("CAPCS answered")
                box(exc["answer"], style="info")

        st.divider()
        label("Your answer")
        answer = st.text_area(
            "Answer the question above:",
            placeholder="Your answer...",
            height=120,
            key=f"answer_{round_num}"
        )

        if st.button("→ Continue", type="primary", use_container_width=True):
            if not answer.strip():
                st.warning("Please write an answer before continuing.")
            else:
                cd["user_answer"] = answer
                st.session_state.current_decision = cd

                # Generate shift question
                with st.spinner("..."):
                    cd["shift_question"] = get_shift_followup(
                        cd["decision"], cd.get("leaning", ""), answer, profile, history_text
                    )
                    st.session_state.current_decision = cd

                st.session_state.sub_state = "shifted"
                st.rerun()

    # ── SUB-STATE: SHIFTED ────────────────────────────────────────────────────
    # Ask if thinking shifted; if yes, gather how + new leaning + reasoning
    elif sub_state == "shifted":

        label("❓ The question")
        box(cd.get("question_text", "—"), style="question")
        label("Your answer")
        box(cd.get("user_answer", "—"), style="answer")

        st.divider()

        st.markdown(f"**{cd.get('shift_question', 'Has this shifted your thinking?')}**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, my thinking shifted", use_container_width=True):
                cd["thinking_shifted"] = True
                st.session_state.current_decision = cd
                st.session_state.sub_state = "how_shifted"
                st.rerun()
        with col2:
            if st.button("➡️ Not yet — challenge me again", use_container_width=True):
                # Save round without confidence update, go to next round
                rounds_log = cd.get("rounds_log", [])
                rounds_log.append({
                    "round": round_num,
                    "bias": cd.get("bias_text", ""),
                    "explanation": cd.get("explanation_text", ""),
                    "perspective": cd.get("perspective_text", ""),
                    "question": cd.get("question_text", ""),
                    "followups": st.session_state.followup_exchanges,
                    "answer": cd.get("user_answer", ""),
                    "shifted": False,
                    "leaning": cd.get("leaning", ""),
                    "confidence": cd["confidence_before"],
                    "shift": 0
                })

                if round_num >= MAX_ROUNDS:
                    entry = {
                        "decision": cd["decision"], "options": cd["options"],
                        "confidence_start": cd["confidence_start"],
                        "confidence_final": cd["confidence_before"],
                        "confidence_shift": cd["confidence_before"] - cd["confidence_start"],
                        "final_choice": cd.get("leaning", ""),
                        "rounds_completed": round_num, "rounds_log": rounds_log,
                        "timestamp": cd["timestamp"],
                        "completed_at": datetime.now().isoformat()
                    }
                    save_log(entry)
                    st.session_state.session_log.append(entry)
                    st.session_state.phase = "report"
                else:
                    st.session_state.current_decision = {
                        "decision": cd["decision"], "options": cd["options"],
                        "leaning": cd.get("leaning", ""),
                        "confidence_before": cd["confidence_before"],
                        "confidence_start": cd["confidence_start"],
                        "timestamp": cd["timestamp"],
                        "rounds": round_num, "rounds_log": rounds_log
                    }
                    st.session_state.sub_state = "present"
                    st.session_state.followup_exchanges = []
                st.rerun()

    # ── SUB-STATE: HOW SHIFTED ────────────────────────────────────────────────
    elif sub_state == "how_shifted":

        label("Great — tell me more")

        how_shifted = st.text_area(
            "How has your thinking changed? Which option are you now considering?",
            placeholder="e.g. I'm now considering..., because I realised...",
            height=100,
            key=f"how_shifted_{round_num}"
        )

        if st.button("→ Continue", type="primary", use_container_width=True):
            if not how_shifted.strip():
                st.warning("Please describe how your thinking has changed.")
            else:
                cd["how_shifted"] = how_shifted
                st.session_state.current_decision = cd

                # Extract new leaning from what they wrote to pre-fill
                with st.spinner("..."):
                    cd["reasoning_question"] = get_reasoning_followup(
                        cd["decision"], how_shifted, profile, history_text
                    )
                    st.session_state.current_decision = cd

                st.session_state.sub_state = "reasoning"
                st.rerun()

    # ── SUB-STATE: REASONING ─────────────────────────────────────────────────
    elif sub_state == "reasoning":

        label("How your thinking shifted")
        box(cd.get("how_shifted", "—"), style="answer")

        st.divider()
        st.markdown(f"**{cd.get('reasoning_question', 'Why this option?')}**")

        reasoning = st.text_area(
            "Your reasoning:",
            placeholder="e.g. Because it aligns with my values of... and the risk feels manageable because...",
            height=100,
            key=f"reasoning_{round_num}"
        )

        leaning_now = st.text_input(
            "Which option are you now leaning towards?",
            key=f"leaning_now_{round_num}"
        )

        st.divider()
        label("Updated confidence level")
        st.caption("Now that your thinking has shifted, how confident are you in this direction?")
        confidence_after = st.slider(
            "Confidence", min_value=0, max_value=100,
            value=cd["confidence_before"], step=5,
            key=f"conf_{round_num}", label_visibility="collapsed"
        )
        badge(confidence_after)

        shift = confidence_after - cd["confidence_before"]
        if shift > 0:
            st.caption(f"↑ {shift}% — Confidence growing.")
        elif shift < 0:
            st.caption(f"↓ {abs(shift)}% — You've recalibrated downward.")
        else:
            st.caption("Same confidence level — that's fine.")

        threshold_reached = confidence_after >= CONFIDENCE_THRESHOLD
        max_reached = round_num >= MAX_ROUNDS

        btn = f"✓ I've reached a confident decision ({confidence_after}%)" if threshold_reached else (
              "✓ Complete and see report" if max_reached else "→ Keep challenging me")

        if st.button(btn, type="primary", use_container_width=True):
            rounds_log = cd.get("rounds_log", [])
            rounds_log.append({
                "round": round_num,
                "bias": cd.get("bias_text", ""),
                "explanation": cd.get("explanation_text", ""),
                "perspective": cd.get("perspective_text", ""),
                "question": cd.get("question_text", ""),
                "followups": st.session_state.followup_exchanges,
                "answer": cd.get("user_answer", ""),
                "shifted": True,
                "how_shifted": cd.get("how_shifted", ""),
                "leaning": leaning_now or cd.get("leaning", ""),
                "reasoning": reasoning,
                "confidence": confidence_after,
                "shift": shift
            })

            if threshold_reached or max_reached:
                entry = {
                    "decision": cd["decision"], "options": cd["options"],
                    "confidence_start": cd["confidence_start"],
                    "confidence_final": confidence_after,
                    "confidence_shift": confidence_after - cd["confidence_start"],
                    "final_choice": leaning_now or cd.get("leaning", ""),
                    "rounds_completed": round_num, "rounds_log": rounds_log,
                    "timestamp": cd["timestamp"],
                    "completed_at": datetime.now().isoformat()
                }
                save_log(entry)
                st.session_state.session_log.append(entry)
                st.session_state.phase = "report"
            else:
                # Add perspective to options and start next round
                new_options = cd["options"]
                if cd.get("perspective_text"):
                    first_sentence = cd["perspective_text"].split(".")[0]
                    new_options += f" / {first_sentence}"
                st.session_state.current_decision = {
                    "decision": cd["decision"], "options": new_options,
                    "leaning": leaning_now or cd.get("leaning", ""),
                    "confidence_before": confidence_after,
                    "confidence_start": cd["confidence_start"],
                    "timestamp": cd["timestamp"],
                    "rounds": round_num, "rounds_log": rounds_log
                }
                st.session_state.sub_state = "present"
                st.session_state.followup_exchanges = []
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# PHASE 3 — FINAL REPORT
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "report":

    last = st.session_state.session_log[-1] if st.session_state.session_log else {}

    st.success("Session complete. Here is your decision report.")
    st.divider()

    st.markdown("### 📊 Session Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Starting confidence", f"{last.get('confidence_start', 0)}%")
    with col2:
        st.metric("Final confidence", f"{last.get('confidence_final', 0)}%")
    with col3:
        shift = last.get("confidence_shift", 0)
        st.metric("Total shift", f"{shift:+}%", delta_color="inverse")

    st.markdown(f"**Final decision:** {last.get('final_choice', '—')}")
    st.markdown(f"**Rounds completed:** {last.get('rounds_completed', 0)}")

    st.divider()
    st.markdown("### 🔍 Challenge Rounds")
    for r in last.get("rounds_log", []):
        shifted_label = "✅ Shifted" if r.get("shifted") else "➡️ Not yet"
        with st.expander(
            f"Round {r['round']} — {shifted_label} | "
            f"Confidence: {r.get('confidence', 0)}% ({r.get('shift', 0):+}%)"
        ):
            if r.get("bias"):
                st.markdown(f"⚠️ **Bias:** {r['bias']}")
            if r.get("explanation"):
                st.markdown(f"📖 **Explanation:** {r['explanation']}")
            if r.get("perspective"):
                st.markdown(f"💡 **Perspective:** {r['perspective']}")
            if r.get("question"):
                st.markdown(f"❓ **Question:** {r['question']}")
            if r.get("followups"):
                for fq in r["followups"]:
                    st.markdown(f"↩️ **Follow-up:** {fq.get('question', '')}")
                    st.markdown(f"   *Answer:* {fq.get('answer', '')}")
            if r.get("answer"):
                st.markdown(f"💬 **Your answer:** {r['answer']}")
            if r.get("shifted"):
                st.markdown(f"🔄 **How it shifted:** {r.get('how_shifted', '')}")
                st.markdown(f"💭 **Reasoning:** {r.get('reasoning', '')}")
                st.markdown(f"→ **New leaning:** {r.get('leaning', '—')}")

    # ── Decision matrix ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🗂 Decision Matrix")
    box("Rate each option — including perspectives introduced during the session.", style="info")

    option_list = [o.strip() for o in last.get("options", "").split("/") if o.strip()]

    if option_list:
        criteria = [
            "Alignment with your values",
            "Practical feasibility",
            "Risk (1 = risky, 10 = safe)",
            "Reversibility (1 = permanent, 10 = easy to undo)"
        ]
        matrix_data = {}
        for i, opt in enumerate(option_list):
            st.markdown(f"**{opt}**")
            scores = []
            cols = st.columns(4)
            for j, criterion in enumerate(criteria):
                with cols[j]:
                    scores.append(st.slider(criterion, 1, 10, 5, key=f"m_{i}_{j}"))
            matrix_data[opt] = scores
            st.markdown("")

        if st.button("📊 Calculate matrix", type="primary", use_container_width=True):
            results = {opt: sum(s) for opt, s in matrix_data.items()}
            best = max(results, key=results.get)
            header = "| Option | Values | Feasibility | Risk | Reversibility | **Total** |"
            sep = "|---|---|---|---|---|---|"
            rows = "\n".join(
                f"| {opt} | {s[0]} | {s[1]} | {s[2]} | {s[3]} | **{sum(s)}/40** |"
                for opt, s in matrix_data.items()
            )
            st.markdown(f"{header}\n{sep}\n{rows}")
            box(f"📌 Highest-scoring: <b>{best}</b> ({results[best]}/40)<br>"
                f"<small>Based on your ratings. Your final decision remains yours.</small>",
                style="perspective")

    # ── Longitudinal ─────────────────────────────────────────────────────────
    history_all = load_file(LOG_FILE)
    completed = [h for h in history_all if "confidence_final" in h]
    if len(completed) > 1:
        st.divider()
        st.markdown("### 📈 Your Pattern Over Time")
        avg_shift = sum(h.get("confidence_shift", 0) for h in completed) / len(completed)
        st.markdown(f"**Total sessions:** {len(completed)}")
        st.markdown(f"**Average confidence shift:** {avg_shift:+.1f}%")
        all_biases = []
        for h in completed:
            for r in h.get("rounds_log", []):
                b = r.get("bias", "").split("—")[0].strip()
                if b:
                    all_biases.append(b[:40])
        if all_biases:
            top = Counter(all_biases).most_common(3)
            st.markdown("**Most recurring biases:**")
            for bn, count in top:
                st.markdown(f"- {bn} ({count}x)")

    st.divider()
    if st.button("→ New decision", type="primary", use_container_width=True):
        st.session_state.phase = "input"
        st.session_state.sub_state = "present"
        st.session_state.current_decision = {}
        st.session_state.followup_exchanges = []
        st.rerun()
