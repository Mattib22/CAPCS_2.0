"""
CAPCS — Co-Adaptive Predictive Cognitive System
Prototype v0.2

Architecture:
- Phase 0: Onboarding questionnaire (one-time, builds user context model)
- Phase 1: Decision input — data gathering ONLY (decision, options, leaning, confidence)
- Phase 2: Challenge loop — bias detection, alternative suggestion, Socratic question
           Repeats up to 3 rounds OR until confidence reaches 80%
- Phase 3: Final report — session summary + decision matrix + final confidence

API key stored as Streamlit secret — users never see it.

Setup:
  Create .streamlit/secrets.toml locally with:
    GEMINI_API_KEY = "your-key-here"
  Or on Streamlit Cloud: App Settings → Secrets → add the same line.

Requirements:
  pip install google-generativeai streamlit
"""

import streamlit as st
import json
import os
from datetime import datetime
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
        margin-bottom: 4px;
    }
    .insight-box {
        background: #F5EED8;
        border-left: 3px solid #9A7B3A;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 8px 0;
        font-size: 14px;
        line-height: 1.6;
    }
    .info-box {
        background: #EEF2FF;
        border-left: 3px solid #4A6FA5;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 8px 0;
        font-size: 13px;
        line-height: 1.6;
    }
    .confidence-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        margin: 4px 0;
    }
    .matrix-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin-top: 8px;
    }
    .matrix-table th {
        background: #F5EED8;
        padding: 8px;
        text-align: left;
        border: 1px solid #DDD5C0;
    }
    .matrix-table td {
        padding: 8px;
        border: 1px solid #DDD5C0;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 80
MAX_ROUNDS = 3
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
    "phase": "onboarding",   # onboarding → input → challenge → report
    "session_log": [],
    "current_decision": {},
    "onboarding_step": 0,
    "onboarding_answers": {},
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

def phase_label(text):
    st.markdown(f'<div class="phase-label">{text}</div>', unsafe_allow_html=True)

def insight(text, style="insight"):
    css = "insight-box" if style == "insight" else "info-box"
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)

# ── PROFILE FORMATTER ──────────────────────────────────────────────────────────
def format_profile(profile):
    if not profile:
        return "No user profile available."
    labels = {
        "education": "Education",
        "profession": "Profession",
        "decision_style": "Decision-making style",
        "risk_tolerance": "Risk tolerance",
        "values": "Core values",
        "passions": "Passions",
        "hobbies": "Hobbies",
        "social_context": "Social decision style",
        "known_bias": "Known personal blind spot",
    }
    return "\n".join(
        f"- {label}: {profile[key]}"
        for key, label in labels.items()
        if profile.get(key)
    )

# ── AI PROMPT ──────────────────────────────────────────────────────────────────
def build_prompt(decision, options, leaning, confidence, profile, history_summary):
    return f"""You are CAPCS — a Socratic cognitive partner that helps people make better decisions.

Your response must follow this EXACT format — three lines only, nothing else:

BIAS IDENTIFIED: [name the bias] — [one sentence: why it applies to THIS person given their profile]
ALTERNATIVE TO CONSIDER: [one concrete option they have NOT listed — specific, actionable, one sentence]
QUESTION: [one Socratic question targeting their current leaning — slightly outside their comfort zone]

Critical rules:
- Use the user's personal profile to make all three outputs specific, not generic
- The alternative MUST be different from their listed options
- Never recommend what to decide
- Never be preachy or judgmental
- The question should stretch their thinking, not attack their choice

User profile:
{format_profile(profile)}

Decision: {decision}
Options listed: {options}
Currently leaning towards: {leaning if leaning else "not specified"}
Confidence in current leaning: {confidence}%
{f"Past decision patterns: {history_summary}" if history_summary else ""}"""

def get_challenge(decision, options, leaning, confidence, profile, history_summary=""):
    prompt = build_prompt(decision, options, leaning, confidence, profile, history_summary)
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.8, "max_output_tokens": 300}
    )
    return response.text.strip()

def parse_challenge(text):
    bias, alternative, question = "", "", ""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("BIAS IDENTIFIED:"):
            bias = line.replace("BIAS IDENTIFIED:", "").strip()
        elif line.startswith("ALTERNATIVE TO CONSIDER:"):
            alternative = line.replace("ALTERNATIVE TO CONSIDER:", "").strip()
        elif line.startswith("QUESTION:"):
            question = line.replace("QUESTION:", "").strip()
    return bias, alternative, (question or text)

def history_summary():
    history = load_file(LOG_FILE)
    if not history:
        return ""
    topics = [h.get("decision", "")[:50] for h in history[-5:]]
    shifts = [h.get("confidence_shift", 0) for h in history if "confidence_shift" in h]
    avg = sum(shifts) / len(shifts) if shifts else 0
    return f"Past decisions: {'; '.join(topics)}. Avg confidence shift: {avg:+.0f}%"

# ── ONBOARDING QUESTIONS ───────────────────────────────────────────────────────
QUESTIONS = [
    {
        "key": "education",
        "label": "What is your educational background?",
        "placeholder": "e.g. BSc Psychology, self-taught, high school + vocational training...",
        "type": "text"
    },
    {
        "key": "profession",
        "label": "What is your current or most recent profession?",
        "placeholder": "e.g. Marketing manager, freelance designer, student...",
        "type": "text"
    },
    {
        "key": "decision_style",
        "label": "How do you tend to make decisions?",
        "placeholder": "e.g. I deliberate for a long time / I trust my gut / I always consult others...",
        "type": "text"
    },
    {
        "key": "risk_tolerance",
        "label": "How comfortable are you with uncertainty and risk?",
        "options": [
            "Very comfortable — I enjoy taking risks",
            "Comfortable — I take calculated risks",
            "Neutral — it depends on the situation",
            "Uncomfortable — I prefer stability",
            "Very uncomfortable — I avoid risk whenever possible"
        ],
        "type": "select"
    },
    {
        "key": "values",
        "label": "What matters most to you when making an important decision?",
        "placeholder": "e.g. Security, growth, freedom, relationships, money, creativity...",
        "type": "text"
    },
    {
        "key": "passions",
        "label": "What are you most passionate about in life?",
        "placeholder": "e.g. Technology, art, science, travel, social impact...",
        "type": "text"
    },
    {
        "key": "hobbies",
        "label": "What do you do in your free time?",
        "placeholder": "e.g. Reading, sport, music, gaming, cooking...",
        "type": "text"
    },
    {
        "key": "social_context",
        "label": "When facing an important decision, you tend to:",
        "options": [
            "Decide alone — I trust my own judgement",
            "Consult a few trusted people",
            "Discuss widely — the more perspectives the better",
            "It varies depending on the decision"
        ],
        "type": "select"
    },
    {
        "key": "known_bias",
        "label": "Is there a pattern in your past decisions you'd like to change?",
        "placeholder": "e.g. I overthink and miss opportunities / I act too impulsively / I always play it safe...",
        "type": "text"
    },
]

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ CAPCS")
    st.markdown("*Co-Adaptive Predictive Cognitive System*")
    st.divider()

    profile = load_file(PROFILE_FILE)
    if profile:
        st.markdown("**Your profile**")
        insight(
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
    history = load_file(LOG_FILE)
    completed = [h for h in history if "confidence_shift" in h]
    if completed:
        st.markdown(f"**Sessions:** {len(completed)}")
        avg = sum(h["confidence_shift"] for h in completed) / len(completed)
        st.markdown(f"**Avg. confidence shift:** {avg:+.1f}%")
        st.caption("Negative = more calibrated thinking ✓")
        from collections import Counter
        all_biases = []
        for h in completed:
            for r in h.get("rounds_log", []):
                b = r.get("bias", "").split("—")[0].strip()
                if b:
                    all_biases.append(b)
        if all_biases:
            top = Counter(all_biases).most_common(1)[0]
            st.caption(f"Most common bias: {top[0]} ({top[1]}x)")

    st.divider()
    if st.button("🗑 Clear all data", use_container_width=True):
        for f in [LOG_FILE, PROFILE_FILE]:
            if os.path.exists(f):
                os.remove(f)
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# ── MAIN HEADER ────────────────────────────────────────────────────────────────
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
        insight(
            "I'd like to learn a little about you. This helps me give you challenges "
            "that are personally relevant — not generic questions that could apply to anyone. "
            "It takes about 2 minutes and only needs to be done once.",
            style="info"
        )
        st.markdown("")

    q = QUESTIONS[step]
    st.markdown(f"**{q['label']}**")

    if q["type"] == "text":
        answer = st.text_input(
            label="answer",
            label_visibility="collapsed",
            placeholder=q.get("placeholder", ""),
            key=f"q_{step}",
            value=st.session_state.onboarding_answers.get(q["key"], "")
        )
    else:
        prev = st.session_state.onboarding_answers.get(q["key"], q["options"][0])
        idx = q["options"].index(prev) if prev in q["options"] else 0
        answer = st.radio(
            label="answer",
            label_visibility="collapsed",
            options=q["options"],
            key=f"q_{step}",
            index=idx
        )

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
                    save_file(PROFILE_FILE, {
                        **st.session_state.onboarding_answers,
                        "completed_at": datetime.now().isoformat()
                    })
                    st.session_state.phase = "input"
                    st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DECISION INPUT (data gathering only)
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "input":

    profile = load_file(PROFILE_FILE)
    if not profile:
        st.session_state.phase = "onboarding"
        st.rerun()

    insight(
        "Tell me about the decision you're facing. Don't worry about having all the "
        "answers — I just need enough to start challenging your thinking.",
        style="info"
    )
    st.markdown("")

    phase_label("Step 1 — The decision")
    decision = st.text_area(
        "What decision are you working through?",
        placeholder="e.g. I'm thinking of leaving my current job for a startup opportunity...",
        height=100
    )

    st.markdown("")
    phase_label("Step 2 — The options")
    st.caption("List the options you're aware of right now. Don't worry if the list feels incomplete.")
    options = st.text_area(
        "What options are you currently considering?",
        placeholder="e.g. Stay in current role / Join the startup / Take 3 months to explore...",
        height=80
    )

    st.markdown("")
    phase_label("Step 3 — Your current leaning")
    st.caption("Where is your mind right now? This is not a commitment — just a starting point.")
    leaning = st.text_input(
        "Which option are you currently leaning towards?",
        placeholder="e.g. Joining the startup..."
    )

    st.markdown("")
    phase_label("Step 4 — Your confidence")
    st.caption("How sure are you about that leaning right now?")
    confidence = st.slider(
        "Confidence level",
        min_value=0, max_value=100, value=50, step=5,
        label_visibility="collapsed",
        help="0 = completely unsure, 100 = totally certain"
    )
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
                "decision": decision,
                "options": options,
                "leaning": leaning,
                "confidence_before": confidence,
                "confidence_start": confidence,
                "timestamp": datetime.now().isoformat(),
                "rounds": 0,
                "rounds_log": []
            }
            st.session_state.phase = "challenge"
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CHALLENGE LOOP
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "challenge":

    cd = st.session_state.current_decision
    profile = load_file(PROFILE_FILE)
    round_num = cd.get("rounds", 0) + 1

    # Progress bar
    st.progress(round_num / MAX_ROUNDS)
    phase_label(f"Challenge round {round_num} of {MAX_ROUNDS}")
    st.markdown(f"**{cd['decision']}**")
    st.caption(f"Options: {cd['options']}")
    if cd.get("leaning"):
        st.caption(f"Currently leaning: **{cd['leaning']}**")
    badge(cd["confidence_before"])
    st.divider()

    # Generate challenge if not yet done for this round
    if "challenge_question" not in cd:
        with st.spinner("Analysing your thinking..."):
            try:
                raw = get_challenge(
                    cd["decision"],
                    cd["options"],
                    cd.get("leaning", ""),
                    cd["confidence_before"],
                    profile,
                    history_summary()
                )
                bias, alternative, question = parse_challenge(raw)
                st.session_state.current_decision.update({
                    "challenge_question": question,
                    "bias_identified": bias,
                    "alternative_suggested": alternative,
                })
                cd = st.session_state.current_decision
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

    # Display the three outputs
    if cd.get("bias_identified"):
        phase_label("⚠️ Bias detected")
        insight(cd["bias_identified"])

    if cd.get("alternative_suggested"):
        phase_label("💡 An option you may not have considered")
        insight(cd["alternative_suggested"])

    phase_label("❓ Consider this")
    insight(cd["challenge_question"])

    st.divider()
    phase_label("Your response")

    reflection = st.text_area(
        "How has this changed your thinking? (optional)",
        placeholder="e.g. I hadn't considered that at all, it changes things because...",
        height=80,
        key=f"reflection_{round_num}"
    )

    leaning_updated = st.text_input(
        "Which option are you now leaning towards?",
        value=cd.get("leaning", ""),
        key=f"leaning_{round_num}"
    )

    confidence_after = st.slider(
        "Updated confidence level:",
        min_value=0, max_value=100,
        value=cd["confidence_before"],
        step=5,
        key=f"conf_{round_num}"
    )

    shift = confidence_after - cd["confidence_before"]
    if shift < 0:
        st.caption(f"↓ {abs(shift)}% — You've updated your prior. Good.")
    elif shift > 0:
        st.caption(f"↑ {shift}% — Confidence growing.")
    else:
        st.caption("No change yet — keep thinking.")

    # Determine next action
    threshold_reached = confidence_after >= CONFIDENCE_THRESHOLD
    max_reached = round_num >= MAX_ROUNDS

    if threshold_reached:
        btn = f"✓ I'm confident in my decision ({confidence_after}%)"
    elif max_reached:
        btn = "✓ Complete and see report"
    else:
        btn = "→ Challenge me again"

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("↩ Start over", use_container_width=True):
            st.session_state.phase = "input"
            st.session_state.current_decision = {}
            st.rerun()
    with col2:
        if st.button(btn, type="primary", use_container_width=True):
            rounds_log = cd.get("rounds_log", [])
            rounds_log.append({
                "round": round_num,
                "bias": cd.get("bias_identified", ""),
                "alternative": cd.get("alternative_suggested", ""),
                "question": cd.get("challenge_question", ""),
                "reflection": reflection,
                "leaning": leaning_updated,
                "confidence": confidence_after,
                "shift": shift
            })

            if threshold_reached or max_reached:
                entry = {
                    "decision": cd["decision"],
                    "options": cd["options"],
                    "confidence_start": cd["confidence_start"],
                    "confidence_final": confidence_after,
                    "confidence_shift": confidence_after - cd["confidence_start"],
                    "final_choice": leaning_updated,
                    "rounds_completed": round_num,
                    "rounds_log": rounds_log,
                    "timestamp": cd["timestamp"],
                    "completed_at": datetime.now().isoformat()
                }
                save_log(entry)
                st.session_state.session_log.append(entry)
                st.session_state.phase = "report"
                st.rerun()
            else:
                # Next round — add alternative to options so it appears in next challenge
                new_options = cd["options"]
                if cd.get("alternative_suggested"):
                    new_options += f" / {cd['alternative_suggested']}"
                st.session_state.current_decision = {
                    "decision": cd["decision"],
                    "options": new_options,
                    "leaning": leaning_updated,
                    "confidence_before": confidence_after,
                    "confidence_start": cd["confidence_start"],
                    "timestamp": cd["timestamp"],
                    "rounds": round_num,
                    "rounds_log": rounds_log
                }
                st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# PHASE 3 — FINAL REPORT
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "report":

    last = st.session_state.session_log[-1] if st.session_state.session_log else {}
    profile = load_file(PROFILE_FILE)

    st.success("Session complete. Here is your decision report.")
    st.divider()

    # ── Summary metrics ──────────────────────────────────────────────────────
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

    # ── Round-by-round breakdown ─────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔍 Challenge Rounds")
    for r in last.get("rounds_log", []):
        with st.expander(
            f"Round {r['round']} — Leaning: {r.get('leaning', '—')} | "
            f"Confidence: {r['confidence']}% ({r['shift']:+}%)"
        ):
            if r.get("bias"):
                st.markdown(f"⚠️ **Bias identified:** {r['bias']}")
            if r.get("alternative"):
                st.markdown(f"💡 **Alternative suggested:** {r['alternative']}")
            st.markdown(f"❓ **Challenge question:** {r['question']}")
            if r.get("reflection"):
                st.markdown(f"💭 **Your reflection:** {r['reflection']}")

    # ── Decision matrix ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🗂 Decision Matrix")
    st.caption(
        "A structured view of all options considered — including alternatives suggested "
        "during the challenge. Rate each option against the four criteria."
    )

    # Collect all unique options (original + AI-suggested alternatives)
    all_options_raw = last.get("options", "")
    option_list = [o.strip() for o in all_options_raw.split("/") if o.strip()]

    if option_list:
        criteria = ["Alignment with values", "Feasibility", "Risk (lower = better)", "Reversibility"]
        matrix_data = {}

        for i, opt in enumerate(option_list):
            st.markdown(f"**{opt}**")
            scores = []
            cols = st.columns(4)
            for j, criterion in enumerate(criteria):
                with cols[j]:
                    score = st.slider(
                        criterion,
                        min_value=1, max_value=10, value=5,
                        key=f"matrix_{i}_{j}"
                    )
                    scores.append(score)
            matrix_data[opt] = scores
            st.markdown("---")

        # Build and display matrix table
        if st.button("📊 Calculate matrix scores", use_container_width=True):
            rows = ""
            results = {}
            for opt, scores in matrix_data.items():
                total = sum(scores)
                results[opt] = total
                rows += f"<tr><td><b>{opt}</b></td>"
                for s in scores:
                    rows += f"<td style='text-align:center'>{s}/10</td>"
                rows += f"<td style='text-align:center;font-weight:600'>{total}/40</td></tr>"

            best = max(results, key=results.get)
            st.markdown(f"""
            <table class="matrix-table">
                <tr>
                    <th>Option</th>
                    <th>Values alignment</th>
                    <th>Feasibility</th>
                    <th>Risk</th>
                    <th>Reversibility</th>
                    <th>Total</th>
                </tr>
                {rows}
            </table>
            """, unsafe_allow_html=True)
            st.markdown("")
            insight(
                f"📌 Highest-scoring option: <b>{best}</b> ({results[best]}/40)<br>"
                f"Note: This reflects your own ratings — use it as a reference, "
                f"not a prescription. Your final decision is yours.",
            )

    # ── Longitudinal stats ───────────────────────────────────────────────────
    history = load_file(LOG_FILE)
    completed = [h for h in history if "confidence_final" in h]
    if len(completed) > 1:
        st.divider()
        st.markdown("### 📈 Your Pattern Over Time")
        avg_shift = sum(h.get("confidence_shift", 0) for h in completed) / len(completed)
        st.markdown(f"**Total sessions:** {len(completed)}")
        st.markdown(f"**Average confidence shift:** {avg_shift:+.1f}%")

        from collections import Counter
        all_biases = []
        for h in completed:
            for r in h.get("rounds_log", []):
                b = r.get("bias", "").split("—")[0].strip()
                if b:
                    all_biases.append(b)
        if all_biases:
            top = Counter(all_biases).most_common(3)
            st.markdown("**Most common biases across all sessions:**")
            for bias_name, count in top:
                st.markdown(f"- {bias_name} ({count}x)")

    st.divider()
    if st.button("→ New decision", type="primary", use_container_width=True):
        st.session_state.phase = "input"
        st.session_state.current_decision = {}
        st.rerun()
