"""
CAPCS — Co-Adaptive Predictive Cognitive System
A Socratic decision-making companion grounded in Predictive Processing,
Drift-Diffusion Models, and Bayesian inference.

Architecture:
- 5-round Socratic challenge loop with PP-grounded bias detection
- Personalised confidence threshold (DDM decision boundary) adapted from profile + observed behaviour
- Longitudinal user model: confidence calibration, recurring biases, situational context
- Co-adaptive feedback loop: user corrections directly update AI prompts
- Anti-echo-chamber: cross-session bias deprioritisation, perspective diversity tracking
- Three-path response: shifted / reinforced / still undecided
- Consolidation mode for sustained confidence drops (PP precision reduction)

Setup:
    .streamlit/secrets.toml: GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY
    Streamlit Cloud: App Settings → Secrets

Requirements:
    pip install google-generativeai streamlit supabase
"""

import streamlit as st
import json
import os
import re
import concurrent.futures
from datetime import datetime
from collections import Counter
import google.generativeai as genai
from supabase import create_client

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CAPCS — Cognitive Decision Assistant", page_icon="⚡", layout="centered")

# ── STYLES ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #FDFBF7; }
    /* Disable browser scroll anchoring — this is the primary cause of
       Streamlit pages loading at the bottom instead of the top */
    html, body, .stApp, .main, section.main, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
        overflow-anchor: none !important;
        scroll-behavior: auto !important;
    }
    * { overflow-anchor: none !important; }
    .phase-label { font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#9A7B3A;margin:16px 0 4px 0; }
    .insight-box { background:#F5EED8;border-left:3px solid #9A7B3A;padding:12px 16px;border-radius:0 6px 6px 0;margin:6px 0 12px 0;font-size:14px;line-height:1.7;color:#2A2825 !important; }
    .info-box { background:#EEF2FF;border-left:3px solid #4A6FA5;padding:12px 16px;border-radius:0 6px 6px 0;margin:6px 0 12px 0;font-size:13px;line-height:1.7;color:#1A1A2E !important; }
    .warning-box { background:#FFF4E5;border-left:3px solid #E07B00;padding:12px 16px;border-radius:0 6px 6px 0;margin:6px 0 12px 0;font-size:14px;line-height:1.7;color:#3D2000 !important; }
    .perspective-box { background:#E8F5E9;border-left:3px solid #2E7D52;padding:12px 16px;border-radius:0 6px 6px 0;margin:6px 0 12px 0;font-size:14px;line-height:1.7;color:#1B3A2A !important; }
    .question-box { background:#F3E5F5;border-left:3px solid #7B1FA2;padding:12px 16px;border-radius:0 6px 6px 0;margin:6px 0 12px 0;font-size:15px;line-height:1.7;font-style:italic;color:#2D0A3E !important; }
    .answer-box { background:#F8F9FA;border-left:3px solid #6C757D;padding:12px 16px;border-radius:0 6px 6px 0;margin:6px 0 12px 0;font-size:14px;line-height:1.7;color:#2A2A2A !important; }
    .highlight-box { background:#E3F2FD;border-left:3px solid #1565C0;padding:14px 18px;border-radius:0 6px 6px 0;margin:6px 0 12px 0;font-size:14px;line-height:1.8;color:#0A2A50 !important; }
    .confidence-badge { display:inline-block;padding:4px 14px;border-radius:20px;font-weight:600;font-size:14px;margin:4px 0; }
    /* Force text color inside all boxes on all devices */
    .insight-box *, .info-box *, .warning-box *, .perspective-box *,
    .question-box *, .answer-box *, .highlight-box * { color: inherit !important; }

    /* ── Thinking animation ── */
    @keyframes capcs-pulse {
        0%   { opacity: 0.3; transform: scale(0.8); }
        50%  { opacity: 1;   transform: scale(1.1); }
        100% { opacity: 0.3; transform: scale(0.8); }
    }
    @keyframes capcs-drift {
        0%   { transform: translateY(0px);   opacity: 0.6; }
        50%  { transform: translateY(-6px);  opacity: 1;   }
        100% { transform: translateY(0px);   opacity: 0.6; }
    }
    .capcs-thinking {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px 20px;
        gap: 20px;
    }
    .capcs-dots {
        display: flex;
        gap: 10px;
        align-items: center;
    }
    .capcs-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        animation: capcs-pulse 1.4s ease-in-out infinite;
    }
    .capcs-dot:nth-child(1) { background:#9A7B3A; animation-delay: 0s; }
    .capcs-dot:nth-child(2) { background:#7B1FA2; animation-delay: 0.2s; }
    .capcs-dot:nth-child(3) { background:#2E7D52; animation-delay: 0.4s; }
    .capcs-dot:nth-child(4) { background:#4A6FA5; animation-delay: 0.6s; }
    .capcs-thinking-text {
        font-size: 13px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #9A7B3A;
        font-weight: 600;
        animation: capcs-drift 2s ease-in-out infinite;
    }
    .capcs-icon {
        font-size: 32px;
        animation: capcs-drift 2.5s ease-in-out infinite;
    }

    /* Streamlit spinner override — make it match our theme */
    div[data-testid="stSpinner"] > div {
        border-top-color: #9A7B3A !important;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
MAX_ROUNDS = 5

# Canonical labels — use these everywhere instead of magic strings
UNDECIDED_INITIAL_LABEL = "I'm genuinely undecided"   # shown on decision input
UNDECIDED_MID_SESSION_LABEL = "I'm still undecided"   # shown in round radio
OTHER_LABEL = "Other"

def compute_confidence_threshold(profile: dict, past_sessions: list = None, starting_confidence: int = None) -> int:
    """
    Compute a personalised confidence threshold grounded in DDM theory.

    KEY PRINCIPLE (DDM): The decision boundary is not absolute — it's the
    amount of evidence accumulation needed to commit. Framed relatively:
    threshold = starting_confidence + required_shift

    Where required_shift depends on:
    - Decision style (deliberators need less shift; impulsive need more)
    - Known blind spot (overthinkers need lower bar; impulsive need higher)
    - Observed calibration from past sessions (actual shift patterns)

    This prevents the common failure:
    - User starts at 85% → absolute threshold of 75% is already met → session ends trivially
    - User starts at 20% → absolute threshold of 75% is unreachable → frustration

    The threshold is always set ABOVE the starting confidence by a meaningful margin.
    Clamped so threshold never exceeds 95% or drops below starting_confidence + 10%.
    """
    # Required shift — how much confidence must change for it to count as genuine updating
    # DDM: this is the drift criterion — minimum evidence before committing
    required_shift = 15  # default: 15 percentage points above start

    # ── Profile-based shift adjustments ───────────────────────────────────────
    style = profile.get("decision_style", "").lower()
    if "deliberate for a long time" in style or "need to feel certain" in style:
        required_shift -= 5   # over-deliberators: lower bar — stop them overthinking
    elif "avoid deciding" in style or "wait for things" in style:
        required_shift -= 7   # chronic avoiders: even lower — force a conclusion
    elif "trust my gut" in style or "quickly" in style:
        required_shift += 5   # intuitive: need more evidence before committing
    elif "research extensively" in style or "need data" in style:
        required_shift += 3

    bias = profile.get("known_bias", "").lower()
    if "overthink" in bias or "miss opportunities" in bias:
        required_shift -= 5   # counter analysis paralysis
    elif "impulsive" in bias or "act too impulsively" in bias:
        required_shift += 5   # slow them down
    elif "always play it safe" in bias:
        required_shift -= 3
    elif "need others to agree" in bias or "seek approval" in bias:
        required_shift += 2

    # ── Observed behaviour adjustment ─────────────────────────────────────────
    if past_sessions and len(past_sessions) >= 3:
        completed = [
            h for h in past_sessions
            if h.get("confidence_final") is not None
            and not h.get("undecided_outcome", False)
        ]
        if len(completed) >= 3:
            # Use the user's actual average shift as the empirical calibration signal
            avg_shift = sum(
                abs(h.get("confidence_shift", 0) or 0) for h in completed
            ) / len(completed)

            if avg_shift < 8:
                # User barely moves — lower required shift so sessions can conclude
                required_shift = max(required_shift - 5, 8)
            elif avg_shift > 25:
                # User shifts a lot — raise required shift slightly (easy bar)
                required_shift = min(required_shift + 3, 25)

            # Shift rate: if user almost never shifts, lower the bar further
            total_rounds = sum(h.get("rounds_completed", 0) for h in completed)
            total_shifts = sum(1 for h in completed for r in h.get("rounds_log", []) if r.get("shifted"))
            shift_rate = total_shifts / max(total_rounds, 1)
            if shift_rate < 0.15:
                required_shift = max(required_shift - 3, 8)

    # Clamp required_shift between 8 and 30 percentage points
    required_shift = max(8, min(30, required_shift))

    # ── Compute absolute threshold from starting confidence ────────────────────
    start = starting_confidence if starting_confidence is not None else 50
    threshold = start + required_shift

    # Clamp: never below start+8 (must require some genuine movement)
    # never above 92 (100% confidence is unrealistic for most real decisions)
    threshold = max(start + 8, min(92, threshold))

    return int(threshold)

# ── API KEY ────────────────────────────────────────────────────────────────────
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")

API_KEY = get_api_key()

# ── SUPABASE CLIENT ────────────────────────────────────────────────────────────
import hashlib as _hashlib

def make_user_key(username: str, pin: str) -> str:
    """
    Generate a deterministic anonymous ID from username + PIN.
    sha256(username:pin) — same combo always produces the same key.
    Nothing identifiable is stored — only the hash.
    """
    raw = f"{username.strip().lower()}:{pin.strip()}"
    return _hashlib.sha256(raw.encode()).hexdigest()[:32]


@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

# ── PROFILE FUNCTIONS ──────────────────────────────────────────────────────────
def load_profile_for_user(user_key: str) -> dict:
    if not user_key:
        return {}
    try:
        sb = get_supabase()
        res = sb.table("profiles").select("*").eq("user_key", user_key).execute()
        if res.data:
            return res.data[0]
    except Exception:
        pass
    return {}

def load_profile() -> dict:
    # Check session cache first — avoids Supabase roundtrip immediately after onboarding
    if st.session_state.get("cached_profile"):
        return st.session_state.cached_profile
    return load_profile_for_user(st.session_state.get("user_key", ""))

def save_profile(profile_data: dict):
    user_key = st.session_state.get("user_key", "")
    if not user_key:
        return
    try:
        sb = get_supabase()
        completed_at = profile_data.get("completed_at", datetime.now().isoformat())
        row = {
            "user_key": user_key,
            "age_range": profile_data.get("age_range", ""),
            "education_level": profile_data.get("education_level", ""),
            "education_field": profile_data.get("education_field", ""),
            "values": profile_data.get("values", ""),
            "passions": profile_data.get("passions", ""),
            "current_situation": profile_data.get("current_situation", ""),
            "current_job": profile_data.get("current_job", ""),
            "main_constraint": profile_data.get("main_constraint", ""),
            "who_is_affected": profile_data.get("who_is_affected", ""),
            "decision_style": profile_data.get("decision_style", ""),
            "known_bias": profile_data.get("known_bias", ""),
            "success_criteria": profile_data.get("success_criteria", ""),
            "version": profile_data.get("version", PROFILE_VERSION),
            "completed_at": completed_at,
        }
        # Upsert current profile
        res = sb.table("profiles").upsert(row, on_conflict="user_key").execute()
        # Log to profile_history — every save creates a new row so changes are tracked
        history_row = {**row, "saved_at": datetime.now().isoformat()}
        sb.table("profile_history").insert(history_row).execute()
        return res
    except Exception as e:
        st.warning(f"⚠️ Profile save issue: {e}")

def get_or_create_user_key() -> tuple:
    """
    Retrieve the user's persistent anonymous ID from Supabase by looking up
    their localStorage key. If none exists, generate a new UUID, store it
    in Supabase, and return it.

    Returns (user_key, display_name, is_new_user)
    """
    try:
        sb = get_supabase()
        user_key = st.session_state.get("user_key", "")
        if user_key:
            # Already loaded this session
            return user_key, st.session_state.get("display_name", ""), False

        # Try to get stored key from session state (set by JS below)
        stored_key = st.session_state.get("_stored_user_key", "")
        if stored_key:
            # Look up in Supabase to get display name
            res = sb.table("users").select("*").eq("user_key", stored_key).execute()
            if res.data:
                row = res.data[0]
                return row["user_key"], row.get("display_name", ""), False

        # New user — generate UUID
        import uuid
        new_key = str(uuid.uuid4())
        return new_key, "", True
    except Exception:
        import uuid
        return str(uuid.uuid4()), "", True


def save_user(user_key: str, display_name: str, consent_at: str):
    """
    Update the user's consent timestamp in Supabase.
    The user row is already created at registration — this just fills in consent_given_at.
    Uses upsert as a safety net in case the row doesn't exist.
    """
    try:
        sb = get_supabase()
        sb.table("users").upsert({
            "user_key": user_key,
            "display_name": display_name,
            "consent_given_at": consent_at,
        }, on_conflict="user_key").execute()
    except Exception as e:
        st.warning(f"Could not save consent record: {e}")


    try:
        sb = get_supabase()
        sb.table("profiles").delete().eq("user_key", user_key).execute()
    except Exception:
        pass

# ── SESSION LOG FUNCTIONS ──────────────────────────────────────────────────────
def save_log(entry: dict):
    user_key = st.session_state.get("user_key", "anonymous")
    try:
        sb = get_supabase()
        # Compute session duration in seconds
        session_duration = None
        try:
            start = datetime.fromisoformat(entry.get("timestamp", ""))
            end = datetime.fromisoformat(entry.get("completed_at", ""))
            session_duration = int((end - start).total_seconds())
        except Exception:
            pass

        # Insert session summary row (no rounds_log — stored in rounds table)
        session_row = {
            "user_key": user_key,
            "timestamp": entry.get("timestamp"),
            "completed_at": entry.get("completed_at"),
            "decision": entry.get("decision"),
            "context": entry.get("context", ""),
            "domain": entry.get("domain"),
            "options": entry.get("options"),
            "all_options": entry.get("all_options", []),
            "confidence_start": entry.get("confidence_start"),
            "confidence_final": entry.get("confidence_final"),
            "confidence_shift": entry.get("confidence_shift"),
            "confidence_trajectory": entry.get("confidence_trajectory", []),
            "confidence_threshold": entry.get("confidence_threshold"),
            "final_choice": entry.get("final_choice"),
            "rounds_completed": entry.get("rounds_completed"),
            "round_durations_seconds": entry.get("round_durations_seconds", []),
            "session_duration_seconds": session_duration,
        }
        session_res = sb.table("sessions").insert(session_row).execute()

        # Get the new session's UUID to link rounds AND feedback
        session_id = session_res.data[0]["id"] if session_res.data else None
        # Store session_id back in entry so the report page can use it
        entry["id"] = session_id

        # Insert each round as a separate row in the rounds table
        if session_id:
            for r in entry.get("rounds_log", []):
                round_row = {
                    "session_id": session_id,
                    "user_key": user_key,
                    "round_number": r.get("round"),
                    "timestamp": r.get("timestamp"),
                    "bias": r.get("bias", ""),
                    "explanation": r.get("explanation", ""),
                    "perspective": r.get("perspective", ""),
                    "question": r.get("question", ""),
                    "answer": r.get("answer", ""),
                    "answer_depth": r.get("answer_depth", ""),
                    "answer_emotion": r.get("answer_emotion", ""),
                    "answer_certainty": r.get("answer_certainty", ""),
                    "answer_key_signal": r.get("answer_key_signal", ""),
                    "how_shifted": r.get("how_shifted", ""),
                    "shifted": r.get("shifted", False),
                    "leaning": r.get("leaning", ""),
                    "confidence": r.get("confidence"),
                    "confidence_shift": r.get("shift", 0),
                }
                sb.table("rounds").insert(round_row).execute()

        # Invalidate cache so next load_log fetches fresh data
        st.session_state["_load_log_dirty"] = True

    except Exception as e:
        st.error(f"Could not save session: {e}")

def load_log() -> list:
    """
    Load session history from Supabase. Cached per Streamlit run to avoid
    multiple round-trips per page render.
    """
    user_key = st.session_state.get("user_key", "anonymous")
    # Cache for this render cycle — cleared when new data is saved
    cache_key = f"_load_log_cache_{user_key}"
    if cache_key in st.session_state and st.session_state.get("_load_log_dirty") is not True:
        return st.session_state[cache_key]
    try:
        sb = get_supabase()
        sessions_res = sb.table("sessions").select("*").eq("user_key", user_key).order("completed_at").execute()
        sessions = sessions_res.data or []

        rounds_res = sb.table("rounds").select("*").eq("user_key", user_key).order("round_number").execute()
        rounds_by_session = {}
        for r in (rounds_res.data or []):
            sid = r.get("session_id")
            rounds_by_session.setdefault(sid, []).append(r)

        for s in sessions:
            s["rounds_log"] = rounds_by_session.get(s.get("id"), [])

        st.session_state[cache_key] = sessions
        st.session_state["_load_log_dirty"] = False
        return sessions
    except Exception:
        return []

def delete_log(user_key: str):
    try:
        sb = get_supabase()
        sb.table("sessions").delete().eq("user_key", user_key).execute()
    except Exception:
        pass


# ── STARTING PHASE ─────────────────────────────────────────────────────────────
PROFILE_VERSION = "4"  # Bumped: added passions question

def _starting_phase():
    """Skip onboarding only if this user has a current-version profile."""
    user_key = st.session_state.get("user_key", "")
    if not user_key:
        return "onboarding"
    # Check session cache first
    cached = st.session_state.get("cached_profile", {})
    if cached and cached.get("version") == PROFILE_VERSION:
        return "input"
    # Check Supabase
    try:
        profile = load_profile_for_user(user_key)
        if profile and profile.get("version") == PROFILE_VERSION:
            st.session_state.cached_profile = profile  # cache it
            return "input"
    except Exception:
        pass
    return "onboarding"

# ── SESSION STATE ──────────────────────────────────────────────────────────────
# Note: phase starts as "onboarding" by default — it gets updated to "input"
# when the user enters their name and _starting_phase() is called at that point.
defaults = {
    "phase": "onboarding",
    "sub_state": "present",
    "session_log": [],
    "current_decision": {},
    "onboarding_step": 0,
    "onboarding_answers": {},
    "followup_exchanges": [],
    "all_options": [],
    "cached_profile": {},
    "longitudinal_text": None,
    "observed_profile": None,
    "last_completed_entry": None,
    "previous_phase": "input",
    "show_feedback_page": False,
    "show_followup": False,       # whether follow-up input is expanded in challenge
    "consent_given": False,
    "display_name": "",
    "input_session_counter": 0,
    "generating_round": False,
    "loading_target": None,
    "force_profile_update": False,
    "profile_edit_section": None,
    "confidence_threshold": 75,   # will be recomputed per session
    "_report_analysis": None,
    "_analysis_just_generated": False,
    "_load_log_dirty": False,
    "input_messages": [],
    "input_step": "decision",
    "_input_decision": "",
    "_input_leaning": "",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val
# Note: consent gate is checked AFTER user_key is resolved from localStorage
# to avoid forcing returning users through consent/onboarding on every visit

# ── UI HELPERS ─────────────────────────────────────────────────────────────────
def confidence_color(score):
    if score >= 75: return "#d4edda", "#155724"
    if score >= 40: return "#fff3cd", "#856404"
    return "#f8d7da", "#721c24"

def thinking_animation(message="CAPCS is thinking"):
    """Render a branded animated loading indicator."""
    st.markdown(f"""
    <div class="capcs-thinking">
        <div class="capcs-icon">🧠</div>
        <div class="capcs-dots">
            <div class="capcs-dot"></div>
            <div class="capcs-dot"></div>
            <div class="capcs-dot"></div>
            <div class="capcs-dot"></div>
        </div>
        <div class="capcs-thinking-text">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def navigate_to(phase: str):
    """Navigate to a phase via the loading screen so user always sees feedback."""
    st.session_state.loading_target = phase
    st.session_state.phase = "loading"
    st.rerun()


def scroll_to_top():
    """
    Place an empty container at the top — combined with the CSS overflow-anchor
    fix this ensures the browser doesn't scroll anchor to mid-page elements.
    The primary fix is overflow-anchor:none in the CSS block above.
    """
    st.empty()


def badge(score):
    bg, fg = confidence_color(score)
    st.markdown(f'<span class="confidence-badge" style="background:{bg};color:{fg}">Confidence: {score}%</span>', unsafe_allow_html=True)

def label(text):
    st.markdown(f'<div class="phase-label">{text}</div>', unsafe_allow_html=True)

def box(text, style="insight"):
    # Convert markdown bold (**text**) to HTML bold so it renders correctly
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', str(text))
    # Convert markdown italic (*text*) to HTML italic
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    css = {"insight":"insight-box","info":"info-box","warning":"warning-box",
           "perspective":"perspective-box","question":"question-box",
           "answer":"answer-box","highlight":"highlight-box"}.get(style,"insight-box")
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)

# ── PROFILE ────────────────────────────────────────────────────────────────────
def build_observed_profile(past_sessions: list) -> dict:
    """
    Build an observed profile from past session data.
    Augments the static self-reported profile with what CAPCS has actually seen.
    """
    completed = [h for h in past_sessions if h.get("confidence_final") is not None]
    if not completed:
        return {}
    observed = {}

    # Top 3 recurring biases
    all_biases = []
    for h in completed:
        for r in h.get("rounds_log", []):
            b = r.get("bias", "").split("—")[0].strip()[:60]
            if b: all_biases.append(b)
    if all_biases:
        top = Counter(all_biases).most_common(3)
        observed["observed_recurring_biases"] = ", ".join(f"{b} ({c}x)" for b, c in top)

    # Shift rate
    total_rounds = sum(h.get("rounds_completed", 0) for h in completed)
    total_shifts = sum(1 for h in completed for r in h.get("rounds_log", []) if r.get("shifted"))
    if total_rounds > 0:
        shift_rate = int(100 * total_shifts / total_rounds)
        observed["observed_shift_rate"] = f"{shift_rate}% of rounds result in a thinking shift"

    # Most common domain
    domains = [h.get("domain", "") for h in completed if h.get("domain")]
    if domains:
        observed["observed_main_domain"] = Counter(domains).most_common(1)[0][0]

    # Calibration direction — describe in neutral PP terms (posterior update pattern)
    # We don't have ground truth so we can only describe direction, not correctness
    avg_shift = sum(h.get("confidence_shift", 0) for h in completed) / len(completed)
    if avg_shift < -5:
        observed["observed_calibration"] = f"Posterior typically updates DOWN after reflection (avg {avg_shift:+.1f}%)"
    elif avg_shift > 5:
        observed["observed_calibration"] = f"Posterior typically updates UP after reflection (avg {avg_shift:+.1f}%)"
    else:
        observed["observed_calibration"] = f"Posterior stays near prior after reflection (avg {avg_shift:+.1f}%)"

    # Avg rounds to first shift
    first_shifts = []
    for h in completed:
        for r in h.get("rounds_log", []):
            if r.get("shifted"):
                rn = r.get("round_number") or r.get("round") or 1
                first_shifts.append(rn)
                break
    if first_shifts:
        observed["observed_rounds_to_shift"] = f"{sum(first_shifts)/len(first_shifts):.1f} rounds on average"

    return observed

def format_profile(profile, observed: dict = None, corrections: dict = None) -> str:
    if not profile:
        return "No user profile available."
    fields = {
        "age_range": "Age",
        "education_level": "Education level",
        "education_field": "Field of study / expertise",
        "values": "Core values",
        "passions": "Passions",
        "current_situation": "Current situation",
        "current_job": "Current / most recent job",
        "main_constraint": "Main constraint right now",
        "who_is_affected": "Who is affected by decisions",
        "decision_style": "Decision-making style",
        "known_bias": "Known personal blind spot",
        "success_criteria": "What makes a decision feel right",
    }
    lines = [f"- {lbl}: {profile[key]}" for key, lbl in fields.items() if profile.get(key)]

    # Append observed profile — what CAPCS has actually seen across past sessions
    if observed:
        observed_labels = {
            "observed_recurring_biases": "Observed recurring biases (past sessions)",
            "observed_shift_rate": "Observed thinking shift rate",
            "observed_main_domain": "Most common decision domain",
            "observed_calibration": "Confidence calibration pattern",
            "observed_rounds_to_shift": "Typical rounds needed to shift",
        }
        obs_lines = [f"- {lbl}: {observed[key]}" for key, lbl in observed_labels.items() if observed.get(key)]
        if obs_lines:
            lines.append("--- OBSERVED BEHAVIOUR FROM PAST SESSIONS ---")
            lines.extend(obs_lines)

    # Bias corrections — always injected here so they reach AI even on first session
    if corrections:
        disputed = [b for b, c in corrections.items() if c.get("verdict") == "inaccurate"]
        partial = [b for b, c in corrections.items() if c.get("verdict") == "partial"]
        confirmed = [b for b, c in corrections.items() if c.get("verdict") == "accurate"]
        if disputed or partial or confirmed:
            lines.append("--- USER BIAS CORRECTIONS ---")
        if disputed:
            lines.append(
                f"The user said these bias detections felt inaccurate in past sessions — "
                f"only use them if you have very strong evidence they apply here: {', '.join(disputed)}"
            )
        if partial:
            lines.append(
                f"The user said these were only partially accurate — use with care "
                f"and only when clearly relevant: {', '.join(partial)}"
            )
        if confirmed:
            lines.append(
                f"The user confirmed these are genuine patterns — prioritise when contextually relevant: {', '.join(confirmed)}"
            )

    return "\n".join(lines)

# ── CONVERSATION HISTORY ───────────────────────────────────────────────────────
def build_history(rounds_log):
    if not rounds_log:
        return "No previous rounds yet."
    lines = []
    for r in rounds_log:
        lines.append(f"--- Round {r.get('round') or r.get('round_number','?')} ---")
        if r.get("bias"): lines.append(f"Bias identified: {r['bias']}")
        if r.get("perspective"): lines.append(f"Perspective offered: {r['perspective']}")
        for fq in r.get("followups", []):
            lines.append(f"User asked: {fq.get('question','')}")
            lines.append(f"AI answered: {fq.get('answer','')}")
        if r.get("answer"): lines.append(f"User's answer: {r['answer']}")
        lines.append(f"Thinking shifted: {'Yes' if r.get('shifted') else 'No'}")
        if r.get("how_shifted"): lines.append(f"How shifted: {r['how_shifted']}")
        if r.get("leaning"): lines.append(f"Leaning after: {r['leaning']}")
        if r.get("reasoning"): lines.append(f"Reasoning: {r['reasoning']}")
        if r.get("confidence"): lines.append(f"Confidence: {r['confidence']}%")
    return "\n".join(lines)

def build_longitudinal_context(history_sessions: list) -> str:
    """
    Summarise the user's past sessions into a concise context string
    injected into every AI call. Makes the system genuinely adaptive
    across sessions — not just within a single session.
    """
    if not history_sessions:
        return ""
    completed = [h for h in history_sessions if h.get("confidence_final") is not None]
    if not completed:
        return ""

    lines = ["=== THIS USER'S HISTORY ACROSS ALL PAST SESSIONS (use to personalise) ==="]

    # Calibration pattern
    avg_shift = sum(h.get("confidence_shift", 0) for h in completed) / len(completed)
    avg_start = sum(h.get("confidence_start", 0) for h in completed) / len(completed)
    lines.append(f"Sessions completed: {len(completed)}")
    lines.append(f"Average starting confidence: {avg_start:.0f}%")
    if avg_shift < -5:
        lines.append(f"Calibration pattern: posterior typically updates DOWN after reflection (avg {avg_shift:+.1f}%)")
    elif avg_shift > 5:
        lines.append(f"Calibration pattern: posterior typically updates UP after reflection (avg {avg_shift:+.1f}%)")
    else:
        lines.append(f"Calibration pattern: posterior stays near prior (avg {avg_shift:+.1f}%)")

    # Recurring biases — enforce diversity programmatically, not just via prompt
    all_biases = []
    recent_biases = []  # biases from last 2 sessions
    for i, h in enumerate(completed):
        for r in h.get("rounds_log", []):
            b = r.get("bias", "").split("—")[0].strip()[:60]
            if b:
                all_biases.append(b)
                if i >= len(completed) - 2:
                    recent_biases.append(b)

    if all_biases:
        top_biases = Counter(all_biases).most_common(3)
        lines.append(f"Most recurring biases across past sessions: {', '.join(f'{b} ({c}x)' for b, c in top_biases)}")
        # Hard deprioritisation: if a bias appeared in the last 2 sessions, explicitly flag it
        if recent_biases:
            recent_top = Counter(recent_biases).most_common(2)
            recent_names = [b for b, _ in recent_top]
            lines.append(
                f"RECENTLY USED BIASES (last 2 sessions): {', '.join(recent_names)}. "
                f"AVOID these unless they are clearly the dominant bias this round — the user has already been challenged on them. "
                f"Actively seek a DIFFERENT bias angle."
            )

    # Shift pattern
    total_rounds = sum(h.get("rounds_completed", 0) for h in completed)
    total_shifts = sum(1 for h in completed for r in h.get("rounds_log", []) if r.get("shifted"))
    shift_rate = int(100 * total_shifts / max(total_rounds, 1))
    lines.append(f"Thinking shift rate: {shift_rate}% of rounds")
    if shift_rate < 20:
        lines.append("Behaviour note: this user rarely shifts — use stronger, more direct challenges.")
    elif shift_rate > 60:
        lines.append("Behaviour note: this user shifts easily — ensure perspectives are well-grounded.")

    # Domain pattern
    domains = [h.get("domain", "") for h in completed if h.get("domain")]
    if domains:
        top_domains = Counter(domains).most_common(2)
        lines.append(f"Most common decision domains: {', '.join(d[0] for d in top_domains)}")

    # Context history — recurring situational themes from past sessions
    contexts = [h.get("context", "").strip() for h in completed if h.get("context", "").strip()]
    if contexts:
        recent_contexts = contexts[-3:]
        contexts_summary = " | ".join(c[:120] for c in recent_contexts)
        lines.append(
            f"Recent situational contexts (use to recognise continuity): {contexts_summary}"
        )

    # Answer quality patterns from past sessions — how this user typically responds
    all_depths = []
    all_emotions = []
    all_certainties = []
    all_key_signals = []
    for h in completed:
        for r in h.get("rounds_log", []):
            if r.get("answer_depth"): all_depths.append(r["answer_depth"])
            if r.get("answer_emotion"): all_emotions.append(r["answer_emotion"])
            if r.get("answer_certainty"): all_certainties.append(r["answer_certainty"])
            if r.get("answer_key_signal"): all_key_signals.append(r["answer_key_signal"])

    if all_depths:
        top_depth = Counter(all_depths).most_common(1)[0]
        top_emotion = Counter(all_emotions).most_common(1)[0] if all_emotions else None
        top_certainty = Counter(all_certainties).most_common(1)[0] if all_certainties else None

        depth_note = {
            "surface": "tends to give short, factual answers — push for deeper reflection",
            "reflective": "tends to reflect well — can handle more direct challenges",
            "avoidant": "tends to deflect or change subject — name this pattern directly"
        }.get(top_depth[0], "")

        emotion_note = {
            "anxious": "frequently expresses anxiety — reduce pressure, use grounding questions",
            "conflicted": "frequently conflicted — help identify the core tension",
            "avoidant": "frequently avoids the real issue — probe beneath the surface answer",
            "excited": "frequently excited — ensure enthusiasm isn't masking risk",
            "determined": "frequently determined — challenge assumptions directly"
        }.get(top_emotion[0] if top_emotion else "", "")

        certainty_note = {
            "hedging": "frequently hedges — gently challenge the hedging language directly",
            "low": "frequently uncertain — consolidation questions work better than new challenges",
            "high": "frequently confident — probe whether confidence is warranted"
        }.get(top_certainty[0] if top_certainty else "", "")

        sig_parts = []
        if depth_note: sig_parts.append(f"Answer depth: {top_depth[0]} ({depth_note})")
        if emotion_note: sig_parts.append(f"Emotional tone: {top_emotion[0]} ({emotion_note})")
        if certainty_note: sig_parts.append(f"Certainty pattern: {top_certainty[0]} ({certainty_note})")

        if sig_parts:
            lines.append("OBSERVED REASONING STYLE (calibrate challenge intensity accordingly):")
            lines.extend(f"  - {s}" for s in sig_parts)

        # Include a sample of the most psychologically significant signals
        if all_key_signals:
            recent_signals = all_key_signals[-5:]  # last 5 key signals
            lines.append(
                f"Recent key signals from user's answers: {' | '.join(s[:60] for s in recent_signals)}"
            )

    # Rounds to shift
    first_shifts = []
    for h in completed:
        for r in h.get("rounds_log", []):
            if r.get("shifted"):
                rn = r.get("round_number") or r.get("round") or 1
                first_shifts.append(rn)
                break
    if first_shifts:
        avg_rts = sum(first_shifts) / len(first_shifts)
        lines.append(f"Typically shifts around round {avg_rts:.1f}")

    # Fix 3: Perspective diversity tracking — collect all past perspectives so they are never repeated
    past_perspectives = []
    for h in completed:
        for r in h.get("rounds_log", []):
            p = r.get("perspective", "")
            if p:
                for line in p.split("\n"):
                    if line.strip().startswith("OPTION:"):
                        opt = line.replace("OPTION:", "").strip()[:80]
                        if opt and opt not in past_perspectives:
                            past_perspectives.append(opt)
    if past_perspectives:
        lines.append(
            f"PERSPECTIVES ALREADY OFFERED IN PAST SESSIONS (never repeat these, "
            f"generate genuinely new ones): {' | '.join(past_perspectives[:10])}"
        )

    lines.append("=== END OF USER HISTORY ===")
    return "\n".join(lines)


def ask_ai(prompt, max_tokens=1200):
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.8, "max_output_tokens": max_tokens}
        )
        # response.text raises ValueError if the response was blocked or empty
        try:
            text = response.text.strip()
        except ValueError:
            # Safety filter triggered or empty response — return safe fallback
            return ""
        text = text.lstrip("*").rstrip("*").strip()
        for prefix in ["BIAS:","PERSPECTIVE:","EXPLANATION:","QUESTION:",
                       "Bias:","Perspective:","Explanation:","Question:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text
    except Exception:
        return ""

# ── NEW CONVERSATIONAL AI FUNCTIONS ────────────────────────────────────────────

def get_opening_question(decision, options, confidence, profile_str, context, longitudinal=""):
    """
    Turn 1 only. Pure listening. One warm observation + one open question.
    No bias named. No labels. Forms hypothesis internally but says nothing.
    """
    long_section = f"\n{longitudinal}" if longitudinal else ""
    confidence_label = (
        "completely lost" if confidence < 20 else
        "quite uncertain" if confidence < 40 else
        "roughly 50/50" if confidence < 60 else
        "leaning one way" if confidence < 80 else
        "fairly settled"
    )
    prompt = f"""You are a thinking partner helping someone work through a real decision.

Your job on this first turn is ONLY to listen and ask one question.

Read everything carefully. Form a hypothesis about what might really be going on — but say NOTHING about biases, patterns, or judgements yet.

Write one short paragraph (max 60 words):
- First: one warm, specific observation about what they described — not a compliment, an observation that shows you heard them. Something they said that is worth noticing.
- Last: one genuinely open question — not leading, not rhetorical. A question you do not already know the answer to.

Rules:
- No bias names, no labels, no diagnoses, no "I notice you said"
- Do not reference their confidence level directly
- Warm, direct, second person
- End with exactly one question mark

DECISION: {decision}
OPTIONS: {options}
CONFIDENCE: {confidence}% ({confidence_label})
CONTEXT: {context}
PROFILE:
{profile_str}{long_section}"""
    return ask_ai(prompt, 512)


def get_challenge_response(decision, options, leaning, confidence, profile_str,
                           history, last_answer, context, longitudinal="",
                           emotion="neutral", turn_num=2,
                           is_undecided=False, confidence_dropped=False,
                           sustained_drop=False):
    """
    Turn 2+. The spark turn.
    One conversational message: reflects → names bias naturally → introduces
    new perspective → ends with challenge question on the NEW perspective.
    Also returns extractable structured fields in a hidden block for Supabase.
    """
    long_section = f"\n{longitudinal}" if longitudinal else ""

    tone_note = {
        "anxious": "The user sounds anxious — begin with more acknowledgement before introducing any challenge. Be gentle.",
        "conflicted": "The user sounds conflicted — name the conflict explicitly before anything else.",
        "determined": "The user sounds determined — be more direct. Challenge with confidence.",
        "avoidant": "The user seems to be deflecting — bring them back to the core question gently.",
        "excited": "The user sounds excited — match their energy but introduce a grounding question.",
    }.get(emotion, "")

    undecided_note = "The user is genuinely undecided — do not push toward any option. Help them discover what they actually want." if is_undecided else ""

    drop_note = "The user's confidence has been dropping. Shift to simplifying — identify what is making this feel more complex than it needs to be." if sustained_drop else (
        "The user's confidence dropped last turn — this turn should simplify, not add complexity." if confidence_dropped else ""
    )

    prompt = f"""You are a thinking partner helping someone work through a real decision. You have been listening for {turn_num} turns. Now you have enough to name something.

{tone_note}
{undecided_note}
{drop_note}

Write ONE conversational message (max 120 words) that does three things in sequence:

1. REFLECT — one sentence that builds directly on what the person just said. Use their exact words or phrases. Prove you heard them specifically, not generically.

2. NAME — introduce the cognitive bias pattern naturally, mid-flow, as a revelation not a label. Do not announce it. Weave it in: "what you're describing is actually a well-known pattern called [X] — it shows up when..." Make the name feel like a moment of recognition.

3. CHALLENGE — introduce one new concrete perspective or option they have not considered, embedded naturally. Then end with ONE question that challenges them to engage with THIS NEW perspective — not their original position.

Tone: warm, direct, curious. Like a thoughtful friend who also knows cognitive science.
Never say "I notice" or "it seems like" — just speak.
Final sentence must end with a question mark.

---

After the message, on a new line, output this structured block EXACTLY (it will be hidden from the user — it is for the system only):

---EXTRACT---
BIAS: [bias name only — max 6 words]
EXPLANATION: [plain English: what this bias is and why it appeared here — max 40 words, complete sentence]
PERSPECTIVE: [the new option or angle introduced — max 8 words]
QUESTION: [copy the exact question from above]
---END---

CONVERSATION HISTORY:
{history}

USER'S LAST ANSWER: {last_answer}

DECISION: {decision}
OPTIONS: {options}
LEANING: {leaning or 'genuinely undecided'}
CONFIDENCE: {confidence}%
CONTEXT: {context}
PROFILE:
{profile_str}{long_section}"""
    return ask_ai(prompt, 2048)


def extract_challenge_fields(full_response: str) -> dict:
    """
    Parse the ---EXTRACT--- block from get_challenge_response output.
    Returns dict with bias, explanation, perspective, question.
    Falls back to empty strings on parse failure.
    """
    default = {"bias_text": "", "explanation_text": "", "perspective_text": "", "question_text": ""}
    try:
        if "---EXTRACT---" not in full_response:
            return default
        extract_block = full_response.split("---EXTRACT---")[1].split("---END---")[0].strip()
        result = dict(default)
        for line in extract_block.split("\n"):
            line = line.strip()
            if line.startswith("BIAS:"):
                result["bias_text"] = line.replace("BIAS:", "").strip()
            elif line.startswith("EXPLANATION:"):
                result["explanation_text"] = line.replace("EXPLANATION:", "").strip()
            elif line.startswith("PERSPECTIVE:"):
                result["perspective_text"] = line.replace("PERSPECTIVE:", "").strip()
            elif line.startswith("QUESTION:"):
                result["question_text"] = line.replace("QUESTION:", "").strip()
        return result
    except Exception:
        return default


def get_conversation_message(full_response: str) -> str:
    """
    Extract the conversational message from get_challenge_response output.
    Strips the ---EXTRACT--- block — only returns what the user sees.
    """
    if "---EXTRACT---" in full_response:
        return full_response.split("---EXTRACT---")[0].strip()
    return full_response.strip()


# ── EXISTING AI CALLS (preserved for fallback and report generation) ──────────
def get_bias(decision, options, leaning, confidence, profile_str, history, last_answer="", longitudinal="", confidence_dropped=False, is_undecided=False):
    answer_section = f"\nUSER'S MOST RECENT ANSWER (reason from this first):\n{last_answer}" if last_answer else ""
    long_section = f"\n{longitudinal}" if longitudinal else ""

    undecided_note = "The user is genuinely undecided. Focus on what is BLOCKING a decision rather than which option to pick." if is_undecided else ""

    if confidence_dropped:
        drop_note = """IMPORTANT: The user's confidence has DROPPED since the last round. This means the challenges may be increasing confusion rather than clarity.
Shift your approach: identify a bias that is making the decision feel MORE complex than it needs to be (e.g. analysis paralysis, perfectionism, catastrophising).
The goal this round is to SIMPLIFY, not to introduce more complexity."""
    else:
        drop_note = ""

    prompt = f"""You are a cognitive scientist. Identify the SINGLE most relevant cognitive bias active RIGHT NOW in this decision.

{undecided_note}
{drop_note}

Write ONE complete sentence max 40 words:
"[Bias name] — because [specific profile evidence + what they just said if available], you may be [how this manifests in this decision]."

Rules:
- Max 40 words, complete sentence, output only the sentence
- Cite profile data AND the latest answer if available
- Do NOT repeat a bias already used in THIS SESSION (see history below)
- If the same bias has appeared across previous sessions, that is fine — recurring patterns are meaningful and should be detected again if genuinely present
- Choose the bias that best explains THIS specific moment in the reasoning, not the most dramatic one

CONVERSATION HISTORY (this session — avoid repeating these biases):
{history}{answer_section}{long_section}

USER PROFILE:
{profile_str}

DECISION: {decision}
OPTIONS: {options}
LEANING: {leaning or 'genuinely undecided'}
CONFIDENCE: {confidence}%"""
    return ask_ai(prompt, 2048)

def get_explanation(bias_text, decision, profile_str, history, last_answer="", longitudinal=""):
    answer_section = f"\nUSER'S MOST RECENT ANSWER: {last_answer}" if last_answer else ""
    long_section = f"\n{longitudinal}" if longitudinal else ""
    prompt = f"""Explain this bias in 2-3 short sentences only:
1. What this bias is doing in this specific decision (one sentence)
2. Why it was detected for this person — cite ONE thing from their profile or answer (one sentence)
3. One concrete counter-action (one sentence)

No preamble. No bullets. Plain prose. Final sentence must be complete. Max 60 words total.

BIAS: {bias_text}
DECISION: {decision}
USER PROFILE: {profile_str}
{answer_section}"""
    return ask_ai(prompt, 1024)

def get_perspective(decision, options, leaning, profile_str, bias_text, history, last_answer="", longitudinal="", is_undecided=False):
    answer_section = f"\nUSER'S MOST RECENT ANSWER (calibrate to what they said):\n{last_answer}" if last_answer else ""
    long_section = f"\n{longitudinal}" if longitudinal else ""

    if is_undecided:
        undecided_note = "The user is genuinely undecided — do NOT push them toward any option. Instead offer a reframe or a concrete experiment that would help them discover what they actually want."
    else:
        undecided_note = ""

    prompt = f"""You are a Socratic thinking partner. Offer ONE genuinely useful perspective that challenges the user's current thinking.

{undecided_note}

Your perspective must:
- Be a CONCRETE, SPECIFIC named option or reframe (3-8 words)
- Be meaningfully different from the existing options — not a minor variation
- Represent a genuinely different way of approaching the situation, not just a new label
- Counter the identified bias directly
- Be relevant to this specific person's profile and situation

Bad examples (too vague, too obvious, too similar to existing):
- "Consider all your options carefully"
- "Beach trip this weekend" (when existing option is "go to beach")
- "Think about what you really want"

Good examples (specific, different, challenging):
- "Commit to 30-day trial of one option" (forces experiential learning over analysis)
- "Consult someone who chose differently" (breaks echo chamber)
- "Write the decision off for 2 weeks" (tests urgency assumption)
- "Design the hybrid version of both" (challenges false dichotomy)

Format EXACTLY — two lines only:
OPTION: [3-8 word specific named perspective]
WHY: [one sentence connecting this to their bias, their answer, and their specific situation]

CONVERSATION HISTORY:
{history}{answer_section}{long_section}

USER PROFILE:
{profile_str}

DECISION: {decision}
EXISTING OPTIONS (must be genuinely different from these): {options}
CURRENTLY LEANING: {leaning or 'genuinely undecided'}
BIAS IDENTIFIED: {bias_text}

Output only the two lines. Nothing else."""
    return ask_ai(prompt, 2048)

def get_question(decision, leaning, perspective_text, profile_str, history, last_answer="", longitudinal=""):
    answer_section = f"\nUSER'S MOST RECENT ANSWER (engage directly with this):\n{last_answer}" if last_answer else ""
    long_section = f"\n{longitudinal}" if longitudinal else ""
    prompt = f"""You are a Socratic thinking partner. Ask ONE short, plain question that builds on what the user just said.

Critical rules:
- Max 20 words. Short is better than long.
- Plain conversational language — no academic or psychological jargon whatsoever
- Match the register of the decision: if it's a small everyday decision, ask a simple direct question; only go deeper for major life decisions
- Must engage with something specific from their answer
- Cannot be answered yes/no
- Must NOT repeat a question from history
- Output ONLY the question, nothing else

Examples of good questions (simple, direct):
- "What would you regret more — trying and failing, or not trying?"
- "What's actually stopping you from deciding right now?"
- "If a friend had this choice, what would you tell them?"
- "What would change if you had to decide today?"

CONVERSATION HISTORY:
{history}{answer_section}{long_section}

DECISION: {decision}
LEANING: {leaning or 'unspecified'}
PERSPECTIVE JUST OFFERED: {perspective_text}"""
    return ask_ai(prompt, 1024)

def analyse_answer_quality(answer: str, question: str) -> dict:
    """
    Analyse the user's free-text answer for psychological signals.
    Returns a dict with depth, emotion, certainty, key_signal, new_option.
    """
    default = {"depth": "surface", "emotion": "neutral", "certainty": "unclear", "key_signal": "", "new_option": ""}

    if not answer or len(answer.strip()) < 10:
        return default

    prompt = f"""Analyse this decision-making response. Return ONLY a JSON object, nothing else.

Question: {question}
Answer: {answer}

Return exactly this JSON structure with no extra text, replacing each bracketed value with your assessment:
{{"depth": "[surface|reflective|avoidant]", "emotion": "[anxious|excited|conflicted|neutral|determined]", "certainty": "[high|medium|low|hedging]", "key_signal": "[3-6 words capturing the key psychological signal]", "new_option": "[new concrete option if clearly mentioned, else empty string]"}}

Definitions:
- depth: surface=factual or short answer, reflective=shows self-awareness or reasoning, avoidant=deflects or changes subject
- emotion: the single dominant emotional tone
- certainty: high=confident statements, medium=mixed, low=genuinely uncertain, hedging=lots of maybe/perhaps/not sure
- key_signal: the most psychologically revealing phrase or pattern in the answer
- new_option: only if the user clearly introduces a brand new concrete possibility not previously mentioned"""

    try:
        import json as _json, re as _re
        result = ask_ai(prompt, 400)
        if not result:
            return default

        # Strategy 1: try parsing the whole response directly
        try:
            stripped = result.strip()
            # Remove markdown code fences if present
            if stripped.startswith("```"):
                stripped = _re.sub(r"```(?:json)?", "", stripped).strip().rstrip("`").strip()
            parsed = _json.loads(stripped)
            if "depth" in parsed:
                return parsed
        except Exception:
            pass

        # Strategy 2: find the outermost { ... } block (handles nested content)
        # Use a stack-based approach rather than regex to handle nesting
        start = result.find("{")
        if start == -1:
            return default
        depth = 0
        end = -1
        for i, ch in enumerate(result[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            return default

        try:
            parsed = _json.loads(result[start:end])
            if "depth" in parsed:
                return parsed
        except Exception:
            pass

        return default
    except Exception:
        return default

def get_consolidation_question(decision, leaning, rounds_log, profile_str, history) -> str:
    """
    Called when confidence drops for 2+ consecutive rounds.
    In PP terms: sustained prediction error divergence means the challenges are
    increasing uncertainty faster than the user can integrate. We must reduce
    precision on new challenges and consolidate the existing generative model.
    In Bayesian terms: identify the sufficient statistic — the single piece of
    information that would most resolve the posterior.
    """
    prompt = f"""The user's confidence has been dropping across multiple rounds — the Socratic challenges are destabilising rather than helping. Shift approach completely.

Do NOT introduce a new challenge or bias. Instead:
- Acknowledge that it's okay to feel uncertain
- Ask ONE grounding question that helps them identify what they already know for certain about this decision
- The question should consolidate, not challenge

Max 35 words, ends with ?, output only the question.

HISTORY:
{history}

DECISION: {decision}
CURRENT LEANING: {leaning or 'undecided'}
USER PROFILE:
{profile_str}"""
    return ask_ai(prompt, 200)


def split_options(text: str) -> list:
    """
    Split user-typed options by common separators: / , vs, or
    Handles natural language inputs like "beach or mountain" or "stay, leave, postpone".
    """
    if not text or not text.strip():
        return []
    cleaned = re.sub(r'\s*\bor\b\s*|\s*\bvs\.?\b\s*|\s*/\s*|\s*,\s*', '|||', text, flags=re.IGNORECASE)
    parts = [p.strip() for p in cleaned.split('|||') if p.strip()]
    seen = set()
    out = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def infer_options(decision_text: str, context: str = "") -> list:
    """
    Auto-infer plausible decision options from a decision description.
    Returns 2-4 short option labels the user can edit.
    """
    if not decision_text.strip():
        return []
    context_section = f"\nContext: {context}" if context.strip() else ""
    prompt = f"""The user is facing this decision: "{decision_text.strip()}"{context_section}

Infer the 2-4 most likely options they are considering. Be concrete and short (3-6 words each).
If the decision is binary (yes/no), include both sides plus possibly a "postpone" or "explore further" option.

Output ONLY the option names, one per line, no numbering, no explanation. Example:
Stay in current job
Take the new offer
Negotiate counter-offer"""
    try:
        result = ask_ai(prompt, 200).strip()
        opts = [line.strip(" -•*").strip() for line in result.split("\n") if line.strip()]
        # Filter to reasonable lengths
        return [o for o in opts if 2 <= len(o.split()) <= 8][:4]
    except Exception:
        return []


    """
    Split user-typed options by common separators: / , vs, or
    Handles natural language inputs like "beach or mountain" or "stay, leave, postpone".
    """
    if not text or not text.strip():
        return []
    # Replace separators with a single delimiter, then split
    # Word boundaries ensure "or" inside a word (e.g. "morning") isn't matched
    cleaned = re.sub(r'\s*\bor\b\s*|\s*\bvs\.?\b\s*|\s*/\s*|\s*,\s*', '|||', text, flags=re.IGNORECASE)
    parts = [p.strip() for p in cleaned.split('|||') if p.strip()]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def save_session_feedback(session_id: str, feedback: dict):
    """
    Save post-session self-report feedback to Supabase.
    Linked to the session by session_id for longitudinal correlation.
    """
    try:
        sb = get_supabase()
        sb.table("session_feedback").insert({
            "session_id": session_id,
            "user_key": st.session_state.get("user_key", ""),
            "cognitive_clarity": feedback.get("cognitive_clarity"),
            "metacognitive_awareness": feedback.get("metacognitive_awareness"),
            "bias_relevance": feedback.get("bias_relevance"),
            "perspective_novelty": feedback.get("perspective_novelty"),
            "overall_usefulness": feedback.get("overall_usefulness"),
            "open_response": feedback.get("open_response", ""),
            "created_at": datetime.now().isoformat()
        }).execute()
        return True
    except Exception as e:
        st.warning(f"Could not save feedback: {e}")
        return False

def feedback_already_submitted(session_id: str) -> bool:
    """Check if feedback was already submitted for this session."""
    try:
        sb = get_supabase()
        res = sb.table("session_feedback").select("session_id").eq("session_id", session_id).execute()
        return len(res.data or []) > 0
    except Exception:
        return False


def save_bias_correction(user_key: str, bias_name: str, verdict: str, note: str):
    """
    Save a user's correction/annotation of a detected bias to Supabase.
    verdict: 'accurate' | 'inaccurate' | 'partial'
    """
    # Strip any Markdown bold formatting that may have leaked from display
    import re as _re
    bias_name = _re.sub(r'\*+', '', bias_name).strip()
    try:
        sb = get_supabase()
        sb.table("bias_corrections").insert({
            "user_key": user_key,
            "bias_name": bias_name,
            "verdict": verdict,
            "note": note,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        st.warning(f"Could not save correction: {e}")

def load_bias_corrections(user_key: str) -> dict:
    """Load all user corrections keyed by bias name."""
    try:
        sb = get_supabase()
        res = sb.table("bias_corrections").select("*").eq("user_key", user_key).execute()
        corrections = {}
        for row in (res.data or []):
            corrections[row["bias_name"]] = {
                "verdict": row["verdict"],
                "note": row["note"]
            }
        return corrections
    except Exception:
        return {}


def get_followup_answer(perspective_text, user_question, decision, profile_str, history):
    """Answer a user's follow-up question about a perspective."""
    prompt = f"""The user has asked a follow-up question about a perspective offered to them. Answer it directly and honestly.

Rules:
- Max 80 words, complete sentences
- Stay grounded in their specific situation — do not give generic advice
- Add genuinely new information or reasoning that wasn't already in the perspective
- Do NOT introduce new biases, new challenges, or new options — just answer the question
- Warm, direct tone — second person
- Final sentence must be complete

CONVERSATION HISTORY:
{history}

PERSPECTIVE OFFERED: {perspective_text}
USER'S QUESTION: {user_question}
DECISION: {decision}
USER PROFILE:
{profile_str}"""
    return ask_ai(prompt, 2048)

def get_decision_domain(decision, profile):
    """Classify the decision into a domain for Phase 3 longitudinal analysis."""
    prompt = f"""Classify this decision into ONE of these domains:
career, education, relationships, finance, location, identity, health, lifestyle, other

Decision: {decision}

Output only the single domain word, lowercase, nothing else."""
    return ask_ai(prompt, 50)

def get_session_recommendation(final_choice, profile, rounds_log, decision, all_options=None):
    """Generate a personalised recommendation based on the full session."""
    biases = [r.get("bias","") for r in rounds_log if r.get("bias")]
    shifts = [r for r in rounds_log if r.get("shifted")]
    perspectives = [r.get("perspective","") for r in rounds_log if r.get("perspective")]
    all_opts = all_options or []

    shift_summary = ""
    for r in shifts:
        rn = r.get("round") or r.get("round_number", "?")
        shift_summary += f"- Round {rn}: shifted to '{r.get('leaning','')}' because: {r.get('how_shifted','')}\n"

    # Slim profile — only fields relevant to the recommendation
    fields = {
        "values": "Values",
        "passions": "Passions",
        "current_situation": "Current situation",
        "current_job": "Job",
        "main_constraint": "Main constraint",
        "decision_style": "Decision style",
        "known_bias": "Known blind spot",
        "success_criteria": "What makes a decision feel right",
    }
    slim_profile = "\n".join(
        f"- {lbl}: {profile[key]}" for key, lbl in fields.items() if profile and profile.get(key)
    ) or "No profile available."

    prompt = f"""You are analysing the results of a Socratic decision-making session. Write a genuinely personalised analysis — not generic advice about "trusting your instincts" or "following your values". Every sentence must be grounded in the specific data below.

USER PROFILE:
{slim_profile}

DECISION: {decision[:500]}

OPTIONS CONSIDERED: {', '.join(all_opts) if all_opts else 'see below'}

FINAL CHOICE: {final_choice}

BIASES IDENTIFIED:
{chr(10).join(f'- {b[:120]}' for b in biases) if biases else 'None identified'}

PERSPECTIVES OFFERED:
{chr(10).join(f'- {p[:120]}' for p in perspectives) if perspectives else 'None'}

THINKING SHIFTS:
{shift_summary if shift_summary else 'No shifts recorded'}

Write a clear, warm analysis covering:
1. Which option makes most sense for this person and why — cite their specific values, situation, and constraint
2. How the specific biases identified may have been limiting their thinking
3. Whether their final choice genuinely aligns with who they are based on their profile
4. One concrete, specific next step — not vague encouragement

CRITICAL FORMATTING:
- Max 200 words
- Second person ("you", "your")
- Plain prose — no bullets, no headers
- Every sentence must be complete — especially the last one
- Do not start with "Based on our session" or "It seems" — start with a direct statement"""
    return ask_ai(prompt, 4096)

def get_bias_analysis(bias_name, count, profile, all_contexts):
    """Short bias insight for Reasoning Profile — 2-3 sentences."""
    fields = {"decision_style": "Decision style", "known_bias": "Known blind spot", "values": "Values"}
    slim_profile = "\n".join(
        f"- {lbl}: {profile[key]}" for key, lbl in fields.items() if profile and profile.get(key)
    ) or "No profile available."

    prompt = f"""In 2-3 short sentences:
1. Why this person tends toward this bias (cite one thing from their profile)
2. What it costs them in decisions
3. One practical counter-action

Bias: {bias_name} (detected {count}x in: {str(all_contexts)[:80]})
Profile: {slim_profile}

Max 50 words. Plain prose. Final sentence complete."""
    return ask_ai(prompt, 512)

# ── ONBOARDING QUESTIONS ───────────────────────────────────────────────────────
QUESTIONS = [
    # ── LAYER 1: STABLE IDENTITY ──────────────────────────────────────────────
    {
        "key": "age_range",
        "label": "How old are you?",
        "section": "About you",
        "options": ["18–24", "25–34", "35–44", "45–54", "55+"],
        "type": "select"
    },
    {
        "key": "education_level",
        "label": "What is your highest level of education?",
        "section": "About you",
        "options": [
            "High school / Secondary school",
            "Vocational / Trade qualification",
            "Bachelor's degree",
            "Master's degree",
            "PhD / Doctorate",
            "Self-taught / No formal qualification"
        ],
        "type": "select"
    },
    {
        "key": "education_field",
        "label": "What field did you study or specialise in?",
        "section": "About you",
        "placeholder": "e.g. Cognitive science, Engineering, Business, Arts, Medicine...",
        "type": "text"
    },
    {
        "key": "values",
        "label": "Which of these matter most to you? Pick up to 3.",
        "section": "About you",
        "options": [
            "Security & stability",
            "Growth & learning",
            "Freedom & independence",
            "Family & relationships",
            "Financial success",
            "Social impact",
            "Creativity & expression",
            "Adventure & new experiences",
            "Status & recognition",
            "Health & wellbeing"
        ],
        "type": "multiselect"
    },
    {
        "key": "passions",
        "label": "What are you most passionate about in life?",
        "section": "About you",
        "placeholder": "e.g. Technology, travel, music, science, helping others, building things...",
        "type": "text"
    },
    # ── LAYER 2: CURRENT CONTEXT ──────────────────────────────────────────────
    {
        "key": "current_situation",
        "label": "What best describes your current situation?",
        "section": "Your situation right now",
        "options": [
            "Employed full-time",
            "Employed part-time",
            "Freelance / Self-employed",
            "Student",
            "Between jobs / Job searching",
            "Travelling / Career break",
            "Running my own business",
            "Other"
        ],
        "type": "select"
    },
    {
        "key": "current_job",
        "label": "What is your current or most recent job / role?",
        "section": "Your situation right now",
        "placeholder": "e.g. Marketing manager, Waiter, Software developer, Student, N/A...",
        "type": "text"
    },
    {
        "key": "main_constraint",
        "label": "What is your biggest constraint right now?",
        "section": "Your situation right now",
        "options": [
            "Money — limited budget or income",
            "Time — too many commitments",
            "Location — tied to a specific place",
            "Family or relationship obligations",
            "Visa or legal status",
            "Lack of experience or qualifications",
            "Health",
            "None — I'm relatively free to choose"
        ],
        "type": "select"
    },
    {
        "key": "who_is_affected",
        "label": "Who else is affected by your decisions?",
        "section": "Your situation right now",
        "options": [
            "Just me — I decide independently",
            "Partner or spouse",
            "Children",
            "Parents or close family",
            "Team or colleagues",
            "Multiple of the above"
        ],
        "type": "select"
    },
    # ── LAYER 3: DECISION-MAKING STYLE ───────────────────────────────────────
    {
        "key": "decision_style",
        "label": "How do you usually make important decisions?",
        "section": "How you decide",
        "options": [
            "I deliberate for a long time — I need to feel certain",
            "I trust my gut — I decide quickly and intuitively",
            "I research extensively — I need data and evidence",
            "I consult others — I value outside perspectives",
            "I tend to avoid deciding — I wait for things to resolve"
        ],
        "type": "select"
    },
    {
        "key": "known_bias",
        "label": "Which of these sounds most like you?",
        "section": "How you decide",
        "options": [
            "I overthink and miss opportunities",
            "I act too impulsively and regret it",
            "I always play it safe even when I shouldn't",
            "I need others to agree before I commit",
            "I ignore my emotions and focus only on logic",
            "I ignore practical realities and follow my feelings",
            "I'm not sure — I'd like to find out"
        ],
        "type": "select"
    },
    {
        "key": "success_criteria",
        "label": "When you look back on a decision, what makes you feel it was the right one?",
        "section": "How you decide",
        "options": [
            "It led to the outcome I wanted",
            "It aligned with my values, regardless of the outcome",
            "Others approved of it or it didn't let anyone down",
            "I felt confident and certain when I made it",
            "It opened up new opportunities I hadn't expected",
            "I have no regrets, even if it didn't work out perfectly"
        ],
        "type": "select"
    },
]


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ CAPCS")
    st.markdown("*A thinking partner, not an answer engine.*")

    # ── Home button ────────────────────────────────────────────────────────────
    if st.button("🏠 New session", key="sidebar_new_session", use_container_width=True, type="primary"):
        st.session_state.phase = "input"
        st.session_state.sub_state = "present"
        st.session_state.current_decision = {}
        st.session_state.followup_exchanges = []
        st.session_state.all_options = []
        st.session_state.session_log = []
        st.session_state.pop("_report_analysis", None)
        # Increment counter to reset all input widget state (context, decision, options)
        st.session_state.input_session_counter = st.session_state.get("input_session_counter", 0) + 1
        st.rerun()

    st.divider()

    profile = load_profile()
    user_key = st.session_state.get("user_key", "")

    if profile:
        if profile.get("version") != PROFILE_VERSION:
            st.warning("⚠️ Your profile needs updating — new questions available.")
        display = st.session_state.get("display_name") or user_key
        st.markdown(f"**👤 {display}**")
        box(
            f'🎓 {profile.get("education_level","—")} — {profile.get("education_field","—")}<br>'
            f'💼 {profile.get("current_job","—")}<br>'
            f'📍 {profile.get("current_situation","—")}',
            style="info"
        )
        if st.button("✏️ Update profile", key="sidebar_update_profile", use_container_width=True):
            st.session_state.phase = "profile_edit"
            st.session_state.force_profile_update = False  # not needed for edit phase
            st.rerun()
    else:
        if user_key:
            st.caption(f"👤 {user_key}")
        st.caption("Complete onboarding to personalise your experience.")

    st.divider()
    hist = load_log()
    completed = [h for h in hist if "confidence_shift" in h]
    if completed:
        st.markdown(f"**Sessions:** {len(completed)}")
        avg = sum(h["confidence_shift"] for h in completed) / len(completed)
        st.markdown(f"**Avg. shift:** {avg:+.1f}%")
        st.markdown("")
        if st.button("📊 My reasoning profile", key="sidebar_reasoning_profile", use_container_width=True):
            st.session_state.previous_phase = st.session_state.get("phase", "input")
            navigate_to("reasoning_profile")
    st.divider()
    if st.button("🗑 Clear my data", key="sidebar_clear_data", use_container_width=True):
        user_key = st.session_state.get("user_key", "anonymous")
        delete_log(user_key)
        delete_profile(user_key)
        # Also delete user record (consent, display name)
        try:
            sb = get_supabase()
            sb.table("users").delete().eq("user_key", user_key).execute()
            sb.table("bias_corrections").delete().eq("user_key", user_key).execute()
        except Exception:
            pass
        for k, v in defaults.items():
            st.session_state[k] = v
        st.session_state.cached_profile = {}
        # Clear localStorage key via JS
        st.markdown("""
        <script>localStorage.removeItem('capcs_user_key');</script>
        """, unsafe_allow_html=True)
        st.rerun()

    st.divider()
    with st.expander("📄 Privacy & Data", expanded=False):
        st.markdown("""
**What we collect**
Anonymised decision text, answers, confidence levels, and session feedback. Your display name is never stored in our database.

**How it's used**
To personalise your experience and for independent research into decision-making and cognitive bias. Data is never sold or shared with third parties.

**Your rights**
- Delete all your data: use **Clear my data** above
- Data is retained for 12 months from your last session
- This tool is a research prototype, not professional advice

**Storage**
Data stored on Supabase (EU servers). Your anonymous ID is stored locally in your browser.

**Contact**
Independent researcher · mrborgini95@outlook.com
        """)


# st.empty() at the very top — gives browser a reference point at position 0
# Do NOT use st.write('#') — markdown headers cause scroll anchoring issues
st.empty()
st.markdown(
    "<p style='font-size:24px;font-weight:700;margin:0;padding:0'>⚡ CAPCS</p>"
    "<p style='font-size:14px;color:#888;margin:0;padding:0'><em>A thinking partner, not an answer engine.</em></p>",
    unsafe_allow_html=True
)
st.divider()

# ── USER IDENTIFICATION ────────────────────────────────────────────────────────
# Persistent anonymous user_key stored in browser localStorage.
# On first visit: generate UUID, write to localStorage, save to Supabase.
# On return visits: read UUID from localStorage, restore session.

# ── USER IDENTIFICATION ────────────────────────────────────────────────────────
# Persistent anonymous user_key stored in browser localStorage.
# IMPORTANT: We do NOT use window.location.replace to pass the key back —
# that causes a full page reload which wipes Streamlit session state mid-session.
# Instead we use a hidden form that posts to Streamlit's query params without reload.

# Read user_key from query params (set on first load by the JS below)
qp = st.query_params
stored_uk = qp.get("uk", "")
if stored_uk and not st.session_state.get("user_key"):
    st.session_state.user_key = stored_uk
    st.session_state.cached_profile = {}
    st.session_state.session_log = []
    st.session_state.current_decision = {}
    st.session_state.longitudinal_text = None
    st.session_state.observed_profile = None
    st.session_state.last_completed_entry = None
    try:
        sb = get_supabase()
        res = sb.table("users").select("display_name, consent_given_at").eq("user_key", stored_uk).execute()
        if res.data:
            row = res.data[0]
            st.session_state.display_name = row.get("display_name", "")
            # User exists in DB — only mark consent given if they completed it
            if row.get("consent_given_at"):
                st.session_state.consent_given = True
                starting = _starting_phase()
                if st.session_state.get("phase") in ("onboarding", "consent"):
                    st.session_state.phase = starting
            # If no consent_given_at, user registered but didn't finish consent
            # — let them proceed to consent page
    except Exception:
        pass

# Only inject the localStorage JS if user is not already identified in session state.
# This avoids the page reload (window.location.replace) when mid-session.
if not st.session_state.get("user_key"):
    # On first load only: read localStorage and append ?uk= to URL once
    # Uses pushState instead of replace so it doesn't wipe session state
    st.markdown("""
<script>
(function() {
    const stored = localStorage.getItem('capcs_user_key');
    if (stored && !new URL(window.location.href).searchParams.get('uk')) {
        const url = new URL(window.location.href);
        url.searchParams.set('uk', stored);
        // Use pushState — does NOT trigger a page reload, just updates the URL
        // Streamlit will pick up the new query param on the next rerun
        window.history.pushState({}, '', url.toString());
        // Trigger a Streamlit rerun by dispatching a popstate event
        window.dispatchEvent(new Event('popstate'));
    }
})();
</script>
""", unsafe_allow_html=True)

# Consent gate — fires only if user_key is already known (new user who just registered)
# or if no user_key yet (will be handled by the identification block below)
if st.session_state.get("user_key") and not st.session_state.get("consent_given"):
    if st.session_state.get("phase") != "consent":
        st.session_state.phase = "consent"

if not st.session_state.get("user_key"):

    st.markdown("### What is CAPCS?")
    box(
        "<b>CAPCS is a Socratic decision-making partner.</b> When you're facing an important decision "
        "and feeling uncertain, CAPCS helps you think more clearly — not by telling you what to do, "
        "but by identifying the cognitive biases that may be distorting your thinking, offering "
        "perspectives you might not have considered, and asking questions that challenge your assumptions.<br><br>"
        "<b>What is a cognitive bias?</b> A cognitive bias is a systematic pattern in how we think that "
        "can lead us to make decisions that don't fully reflect reality. They are not flaws in intelligence — "
        "they are shortcuts the brain uses to process information quickly. Common examples include "
        "<i>confirmation bias</i> (seeking information that confirms what we already believe), "
        "<i>sunk cost fallacy</i> (continuing something because of past investment rather than future value), "
        "and <i>overconfidence</i> (overestimating how certain we are). Most of the time these patterns "
        "operate below conscious awareness — which is exactly why an external challenge can help.<br><br>"
        "The goal is not to reach a specific conclusion. It's to help you arrive at a decision you can "
        "genuinely stand behind — whether that's a new direction or a more confident version of your original one.<br><br>"
        "<b>How it works:</b> You describe your decision and the options you're considering. "
        "CAPCS then runs you through up to 5 challenge rounds, each one building on your answers. "
        "At the end, you get a personalised report with a best-option analysis and a breakdown of your thinking patterns.",
        style="info"
    )
    st.divider()

    tab_new, tab_return = st.tabs(["✨ New user", "🔄 Returning user"])

    with tab_new:
        st.markdown("**Create your access credentials.** Your username + PIN are never stored — they are used only to generate a unique anonymous ID that lets you pick up where you left off.")
        st.markdown("")
        new_username = st.text_input(
            "Choose a username",
            placeholder="e.g. alex, starfish42, j_t — anything memorable",
            key="new_username"
        )
        new_pin = st.text_input(
            "Choose a PIN (4+ digits or characters)",
            placeholder="e.g. 4721 or any short code you'll remember",
            type="password",
            key="new_pin"
        )
        new_pin_confirm = st.text_input(
            "Confirm PIN",
            placeholder="Same PIN again",
            type="password",
            key="new_pin_confirm"
        )
        st.caption("⚠️ Write these down — if you forget your credentials, your session history cannot be recovered.")

        if st.button("→ Create my profile", key="create_profile_btn", type="primary", use_container_width=True):
            if not new_username.strip():
                st.error("Please choose a username.")
            elif len(new_pin.strip()) < 4:
                st.error("PIN must be at least 4 characters.")
            elif new_pin != new_pin_confirm:
                st.error("PINs don't match — please try again.")
            else:
                new_key = make_user_key(new_username, new_pin)
                display_name = new_username.strip()
                try:
                    sb = get_supabase()
                    existing = sb.table("users").select("user_key").eq("user_key", new_key).execute()
                    if existing.data:
                        st.warning("This username + PIN combination already exists. Use the **Returning user** tab to log in, or choose a different combination.")
                    else:
                        # Save user to Supabase immediately at registration
                        try:
                            sb.table("users").insert({
                                "user_key": new_key,
                                "display_name": display_name,
                                "consent_given_at": None,
                            }).execute()
                        except Exception:
                            try:
                                sb.table("users").update({
                                    "display_name": display_name,
                                }).eq("user_key", new_key).execute()
                            except Exception:
                                pass
                        st.session_state.user_key = new_key
                        st.session_state.display_name = display_name
                        st.session_state.phase = "consent"
                        st.markdown(
                            f"<script>localStorage.setItem('capcs_user_key', '{new_key}');</script>",
                            unsafe_allow_html=True
                        )
                        st.rerun()
                except Exception as e:
                    st.error(f"Could not create account: {e}. Please check your connection and try again.")

    with tab_return:
        st.markdown("**Welcome back.** Enter the same username and PIN you used before to recover your session history.")
        st.markdown("")
        ret_username = st.text_input(
            "Username",
            placeholder="Your username",
            key="ret_username"
        )
        ret_pin = st.text_input(
            "PIN",
            placeholder="Your PIN",
            type="password",
            key="ret_pin"
        )
        if st.button("→ Recover my session", key="recover_session_btn", type="primary", use_container_width=True):
            if not ret_username.strip() or not ret_pin.strip():
                st.error("Please enter both your username and PIN.")
            else:
                recovered_key = make_user_key(ret_username, ret_pin)
                try:
                    sb = get_supabase()
                    res = sb.table("users").select("display_name, consent_given_at").eq("user_key", recovered_key).execute()
                    if res.data:
                        row = res.data[0]
                        st.session_state.user_key = recovered_key
                        st.session_state.display_name = row.get("display_name", ret_username.strip())
                        # Only mark consent given if they actually completed it
                        if row.get("consent_given_at"):
                            st.session_state.consent_given = True
                            st.session_state.phase = _starting_phase()
                        else:
                            # Registered but never completed consent — send to consent page
                            st.session_state.phase = "consent"
                        st.markdown(f"<script>localStorage.setItem('capcs_user_key', '{recovered_key}');</script>", unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.error("No session found for this username + PIN combination. Please check your credentials or create a new profile.")
                except Exception as e:
                    st.error(f"Could not connect to retrieve your session: {e}")

    st.stop()


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 0 — ONBOARDING
# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════
# CONSENT PAGE — shown once before onboarding
# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════
# LOADING PHASE — brief branded transition shown between all page navigations
# ════════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "loading":
    target = st.session_state.get("loading_target", "input")
    st.markdown(
        """
        <div style="
            position:fixed;top:0;left:0;width:100vw;height:100vh;
            background:#FDFBF7;z-index:9999;
            display:flex;flex-direction:column;
            align-items:center;justify-content:center;gap:20px;
        ">
            <div style="font-size:36px">⚡</div>
            <div style="display:flex;gap:10px">
                <div style="width:8px;height:8px;border-radius:50%;background:#9A7B3A;
                     animation:capcs-pulse 1.4s ease-in-out 0s infinite"></div>
                <div style="width:8px;height:8px;border-radius:50%;background:#7B1FA2;
                     animation:capcs-pulse 1.4s ease-in-out 0.2s infinite"></div>
                <div style="width:8px;height:8px;border-radius:50%;background:#2E7D52;
                     animation:capcs-pulse 1.4s ease-in-out 0.4s infinite"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    import time as _time
    _time.sleep(0.3)
    st.session_state.phase = target
    st.session_state.loading_target = None
    st.rerun()

elif st.session_state.phase == "consent":
    # Guard: skip if user already consented (returning user)
    if st.session_state.get("consent_given"):
        st.session_state.phase = _starting_phase()
        st.rerun()
    scroll_to_top()
    st.markdown("<p style='font-size:22px;font-weight:700;margin:0'>⚡ Welcome to CAPCS</p>", unsafe_allow_html=True)
    st.markdown("*A thinking partner, not an answer engine.*")
    st.divider()

    box(
        "<b>Before you start, please read this.</b><br><br>"
        "CAPCS is a Socratic decision-making tool that uses AI to challenge your thinking. "
        "It is a research prototype built as part of an independent project in cognitive science.",
        style="info"
    )

    st.markdown("### What data CAPCS collects")
    st.markdown("""
CAPCS collects the following data when you use it:
- The decisions you describe and the context you provide
- Your answers to Socratic challenges
- Your confidence levels and how they change during a session
- The cognitive biases detected in your reasoning
- Your post-session feedback (if you choose to submit it)
- An anonymous ID generated automatically at first use — your name is never stored
""")

    st.markdown("### How your data is used")
    st.markdown("""
Your data is used for:
- Personalising your experience within CAPCS (the system learns your reasoning patterns)
- Improving the tool based on how people use it
- Potential anonymised research and publication in cognitive science contexts

**Your data will never be shared with third parties and will never be used commercially.**
""")

    st.markdown("### Your rights")
    st.markdown("""
- You can delete all your data at any time using the **Clear my data** button in the sidebar
- You can stop using CAPCS at any time with no consequence
- Your display name is shown only to you and is never saved to the database
- Data is retained for 12 months from your last session, then deleted
- This tool is not a substitute for professional advice of any kind
- For any data requests, contact: mrborgini95@outlook.com
""")

    st.divider()

    agree_research = st.checkbox(
        "✅ I agree that my anonymised data may be used for research purposes.",
        value=False, key="consent_research"
    )
    agree_gdpr = st.checkbox(
        "✅ I understand that my data is stored securely, I can delete it at any time, "
        "and I am 18 years of age or older.",
        value=False, key="consent_gdpr"
    )

    both_agreed = agree_research and agree_gdpr
    st.markdown("")
    if st.button(
        "→ I agree — continue to CAPCS",
        key="consent_agree_btn",
        type="primary",
        use_container_width=True,
        disabled=not both_agreed
    ):
        consent_at = datetime.now().isoformat()
        st.session_state.consent_given = True
        # Save consent record to Supabase
        save_user(
            st.session_state.get("user_key", ""),
            st.session_state.get("display_name", ""),
            consent_at
        )
        st.session_state.phase = "onboarding"
        st.rerun()

    if not both_agreed:
        st.caption("Please tick both boxes to continue.")


# ════════════════════════════════════════════════════════════════════════════════
# PROFILE EDIT PHASE — shows current profile with per-section edit buttons
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "profile_edit":
    scroll_to_top()
    profile = load_profile()

    # Section labels and their question keys
    SECTIONS = {
        "About you": ["age_range", "education_level", "education_field", "values", "passions"],
        "Your situation right now": ["current_situation", "current_job", "main_constraint", "who_is_affected"],
        "How you decide": ["decision_style", "known_bias", "success_criteria"],
    }

    edit_section = st.session_state.get("profile_edit_section")

    # ── OVERVIEW — show all sections with Edit buttons ─────────────────────────
    if not edit_section:
        st.markdown("<p style='font-size:22px;font-weight:700;margin:0'>👤 Your Profile</p>", unsafe_allow_html=True)
        st.caption("Click Edit next to any section to update it. Other sections stay unchanged.")
        st.divider()

        field_labels = {q["key"]: q["label"] for q in QUESTIONS}

        for section_name, keys in SECTIONS.items():
            col_title, col_btn = st.columns([4, 1])
            with col_title:
                st.markdown(f"**{section_name}**")
            with col_btn:
                if st.button("✏️ Edit", key=f"edit_{section_name}", use_container_width=True):
                    st.session_state.profile_edit_section = section_name
                    st.rerun()

            for key in keys:
                val = profile.get(key, "—")
                if isinstance(val, list):
                    val = ", ".join(val)
                label = field_labels.get(key, key)
                # Short label — strip question mark and trim
                short = label.rstrip("?").split(".")[-1].strip()[:50]
                st.markdown(f"<small style='color:#888'>{short}</small><br><span style='font-size:15px'>{val or '—'}</span>", unsafe_allow_html=True)
                st.markdown("")

            st.divider()

        if st.button("← Back", key="profile_edit_back", use_container_width=True):
            st.session_state.phase = st.session_state.get("previous_phase", "input")
            st.rerun()

    # ── EDIT — show only the selected section's questions ─────────────────────
    else:
        section_keys = SECTIONS.get(edit_section, [])
        section_questions = [q for q in QUESTIONS if q["key"] in section_keys]

        st.markdown(f"<p style='font-size:22px;font-weight:700;margin:0'>✏️ Edit — {edit_section}</p>", unsafe_allow_html=True)
        st.caption("Update any of these answers. Leave unchanged to keep your current answer.")
        st.divider()

        updated = {}
        for q in section_questions:
            key = q["key"]
            current = profile.get(key, "")
            label = q["label"]

            if q["type"] == "select":
                opts = q["options"]
                # Find current selection index
                try:
                    idx = opts.index(current) if current in opts else 0
                except ValueError:
                    idx = 0
                updated[key] = st.selectbox(label, opts, index=idx, key=f"edit_q_{key}")

            elif q["type"] == "multiselect":
                opts = q["options"]
                current_list = current if isinstance(current, list) else ([current] if current else [])
                valid_defaults = [v for v in current_list if v in opts]
                updated[key] = st.multiselect(label, opts, default=valid_defaults, key=f"edit_q_{key}")

            elif q["type"] == "text":
                updated[key] = st.text_input(
                    label,
                    value=current or "",
                    placeholder=q.get("placeholder", ""),
                    key=f"edit_q_{key}"
                )

        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Cancel", key="profile_edit_cancel", use_container_width=True):
                st.session_state.profile_edit_section = None
                st.rerun()
        with col2:
            if st.button("💾 Save changes", key="profile_edit_save", type="primary", use_container_width=True):
                # Merge updated fields into existing profile
                merged = {**profile, **updated}
                merged["version"] = PROFILE_VERSION
                merged["completed_at"] = datetime.now().isoformat()
                save_profile(merged)
                st.session_state.cached_profile = merged
                st.session_state.profile_edit_section = None
                st.success("✅ Profile updated.")
                st.rerun()


elif st.session_state.phase == "onboarding":
    # Hard consent guard — onboarding must never be reached without consent
    if not st.session_state.get("consent_given"):
        st.session_state.phase = "consent"
        st.rerun()

    # Guard: if user already has a complete profile, skip straight to input
    # UNLESS the user explicitly requested to update their profile
    if not st.session_state.get("force_profile_update"):
        _existing_profile = load_profile()
        if _existing_profile and _existing_profile.get("version") == PROFILE_VERSION:
            st.session_state.cached_profile = _existing_profile
            st.session_state.phase = "input"
            st.rerun()

    total = len(QUESTIONS)
    step = st.session_state.onboarding_step
    st.progress(step / total)
    st.caption(f"Question {step + 1} of {total}")
    st.markdown("")

    if step == 0:
        st.markdown("### Before we start")
        box("I'd like to learn about you — who you are, your situation right now, and how you make decisions. Takes about 3 minutes — only needs to be done once.", style="info")
        st.markdown("")

    q = QUESTIONS[step]

    # Show section header when it changes
    if step == 0 or q.get("section") != QUESTIONS[step - 1].get("section"):
        st.markdown(f"##### {q.get('section','')}")

    st.markdown(f"**{q['label']}**")

    if q["type"] == "text":
        answer = st.text_input("answer", label_visibility="collapsed",
            placeholder=q.get("placeholder",""), key=f"q_{step}",
            value=st.session_state.onboarding_answers.get(q["key"],""))

    elif q["type"] == "select":
        prev = st.session_state.onboarding_answers.get(q["key"], q["options"][0])
        idx = q["options"].index(prev) if prev in q["options"] else 0
        answer = st.radio("answer", label_visibility="collapsed",
            options=q["options"], key=f"q_{step}", index=idx)

    elif q["type"] == "multiselect":
        prev_raw = st.session_state.onboarding_answers.get(q["key"], "")
        prev_list = [x.strip() for x in prev_raw.split(",")] if prev_raw else []
        selected = []
        st.caption("Select up to 3")
        cols = st.columns(2)
        for i, opt in enumerate(q["options"]):
            with cols[i % 2]:
                checked = st.checkbox(opt, value=(opt in prev_list), key=f"q_{step}_opt_{i}")
                if checked:
                    selected.append(opt)
        if len(selected) > 3:
            st.warning("Please select up to 3 options.")
            selected = selected[:3]
        answer = ", ".join(selected)

    st.markdown("")
    col1, col2 = st.columns([1, 2])
    with col1:
        if step > 0:
            if st.button("← Back", key="onboarding_back", use_container_width=True):
                st.session_state.onboarding_step -= 1; st.rerun()
    with col2:
        btn = "Next →" if step < total - 1 else "Start CAPCS →"
        if st.button(btn, key=f"onboarding_next_{step}", type="primary", use_container_width=True):
            if q["type"] == "text" and not answer.strip():
                st.warning("Please add a short answer to continue.")
            elif q["type"] == "multiselect" and not answer.strip():
                st.warning("Please select at least one option.")
            else:
                st.session_state.onboarding_answers[q["key"]] = answer
                if step < total - 1:
                    st.session_state.onboarding_step += 1; st.rerun()
                else:
                    profile_to_save = {
                        **st.session_state.onboarding_answers,
                        "version": PROFILE_VERSION,
                        "completed_at": datetime.now().isoformat()
                    }
                    save_profile(profile_to_save)
                    # Cache in session state so input phase doesn't need Supabase immediately
                    st.session_state.cached_profile = profile_to_save
                    # Clear the force flag so the guard works again next time
                    st.session_state.force_profile_update = False
                    st.session_state.phase = "input"
                    st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DECISION INPUT
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "input":
    if not st.session_state.get("consent_given"):
        st.session_state.phase = "consent"; st.rerun()
    scroll_to_top()
    profile = load_profile()
    if not profile:
        profile = st.session_state.get("onboarding_answers", {})
    if not profile:
        st.session_state.phase = "onboarding"; st.rerun()

    # Resume any paused session
    paused_cd = st.session_state.get("current_decision", {})
    if paused_cd.get("decision") and paused_cd.get("rounds_log") is not None and not st.session_state.get("session_log"):
        box(
            f"<b>You have a paused session:</b><br>"
            f"<i>{paused_cd.get('decision_short') or paused_cd.get('decision','')[:120]}</i>",
            style="info"
        )
        col_resume, col_discard = st.columns(2)
        with col_resume:
            if st.button("▶ Resume session", key="resume_session_btn", type="primary", use_container_width=True):
                st.session_state.phase = "generating"; st.rerun()
        with col_discard:
            if st.button("🗑 Start fresh", key="discard_session_btn", use_container_width=True):
                st.session_state.current_decision = {}
                st.session_state.followup_exchanges = []
                st.session_state.all_options = []
                st.session_state.input_session_counter = st.session_state.get("input_session_counter", 0) + 1
                st.rerun()
        st.divider()

    sc = st.session_state.get("input_session_counter", 0)

    # ── Context (pre-filled from last session if exists) ──────────────────────
    past_sessions_for_hint = load_log()
    past_contexts = [
        h.get("context", "").strip()
        for h in past_sessions_for_hint
        if h.get("context") and str(h.get("context", "")).strip()
    ]

    if past_contexts:
        last_context = past_contexts[-1]
        context_changed = st.checkbox(
            "My situation has changed since last session",
            value=False, key=f"context_changed_toggle_{sc}"
        )
        if context_changed:
            context = st.text_area(
                "What's your current situation?",
                placeholder="e.g. I've just graduated and I'm travelling in Australia...",
                height=80, key=f"input_context_new_{sc}"
            )
        else:
            context = last_context
            st.caption(f"📍 *{last_context}*")
    else:
        context = st.text_area(
            "What's your current situation?",
            placeholder="e.g. I've just graduated and I'm travelling in Australia, feeling unsettled about what comes next...",
            height=80, key=f"input_context_{sc}"
        )

    st.divider()

    # ── Chatbot-style conversation display ────────────────────────────────────
    # Show any messages already exchanged in this input phase
    input_messages = st.session_state.get("input_messages", [])
    for msg in input_messages:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role, avatar="🧠" if role == "assistant" else "👤"):
            st.markdown(content)

    # ── Decision input — chat style ───────────────────────────────────────────
    input_step = st.session_state.get("input_step", "decision")

    if input_step == "decision":
        if not input_messages:
            # First message from CAPCS
            capcs_opener = "What decision are you working through?"
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(capcs_opener)

        user_input = st.chat_input(
            "Describe your decision...",
            key=f"chat_decision_{sc}"
        )
        if user_input and user_input.strip():
            st.session_state.input_messages = [
                {"role": "assistant", "content": "What decision are you working through?"},
                {"role": "user", "content": user_input.strip()},
                {"role": "assistant", "content": "What option are you leaning towards — and why? (If you're genuinely torn, just say that.)"}
            ]
            st.session_state._input_decision = user_input.strip()
            st.session_state.input_step = "leaning"
            st.rerun()

    elif input_step == "leaning":
        user_input = st.chat_input(
            "Your leaning and why...",
            key=f"chat_leaning_{sc}"
        )
        if user_input and user_input.strip():
            messages = st.session_state.get("input_messages", [])
            messages.append({"role": "user", "content": user_input.strip()})
            messages.append({"role": "assistant", "content": "And how clear do you feel going into this? Move the slider below — 0 is completely lost, 100 is basically decided."})
            st.session_state.input_messages = messages
            st.session_state._input_leaning = user_input.strip()
            st.session_state.input_step = "confidence"
            st.rerun()

    elif input_step == "confidence":
        confidence = st.slider(
            "Clarity level",
            0, 100, 35, 5,
            label_visibility="collapsed",
            key=f"confidence_slider_{sc}"
        )
        badge(confidence)
        st.markdown("")
        if st.button("→ Let's think this through", key="challenge_thinking_btn", type="primary", use_container_width=True):
            if not context.strip():
                st.error("Add some context first — what's your current situation?")
            elif not API_KEY:
                st.error("API key not configured.")
            else:
                decision_raw = st.session_state.get("_input_decision", "")
                leaning_raw = st.session_state.get("_input_leaning", "")

                # Extract options from decision text and leaning
                detected = split_options(decision_raw)
                detected = [o for o in detected if o.strip() and len(o) < 80]
                options = " / ".join(detected) if detected else leaning_raw
                is_undecided = any(w in leaning_raw.lower() for w in
                    ["undecided", "not sure", "don't know", "torn", "unsure", "no idea"])

                full_decision = f"{decision_raw}\n\n[CONTEXT]: {context.strip()}"
                past_sessions = load_log()
                st.session_state.confidence_threshold = compute_confidence_threshold(
                    profile, past_sessions, starting_confidence=confidence
                )
                st.session_state.observed_profile = build_observed_profile(past_sessions)
                st.session_state.longitudinal_text = None
                st.session_state.all_options = detected or [leaning_raw]
                st.session_state.current_decision = {
                    "decision": full_decision,
                    "decision_short": decision_raw,
                    "context": context.strip(),
                    "options": options,
                    "leaning": leaning_raw,
                    "is_undecided": is_undecided,
                    "confidence_before": confidence,
                    "confidence_start": confidence,
                    "timestamp": datetime.now().isoformat(),
                    "rounds": 0,
                    "rounds_log": []
                }
                st.session_state.sub_state = "present"
                st.session_state.followup_exchanges = []
                st.session_state.input_step = "decision"
                st.session_state.input_messages = []
                st.session_state.phase = "generating"; st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# GENERATING PHASE — dedicated loading page while AI generates the next challenge
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "generating":
    if not st.session_state.get("consent_given"):
        st.session_state.phase = "consent"; st.rerun()
    scroll_to_top()
    cd = st.session_state.current_decision
    profile = load_profile()
    observed_profile = st.session_state.get("observed_profile", {})
    user_key_corr = st.session_state.get("user_key", "")
    bias_corrections = load_bias_corrections(user_key_corr) if user_key_corr else {}
    enriched_profile_str = format_profile(profile, observed_profile, bias_corrections)
    history_text = build_history(cd.get("rounds_log", []))
    if not st.session_state.get("longitudinal_text"):
        past_sessions = load_log()
        past_completed = [h for h in past_sessions if h.get("completed_at")]
        st.session_state.longitudinal_text = build_longitudinal_context(past_completed)
    longitudinal_text = st.session_state.longitudinal_text or ""

    rounds_log_so_far = cd.get("rounds_log", [])
    confidence_dropped = False
    sustained_drop = False
    if rounds_log_so_far:
        last_shift = rounds_log_so_far[-1].get("shift", 0) or 0
        if last_shift < 0:
            confidence_dropped = True
        if len(rounds_log_so_far) >= 2:
            prev_shift = rounds_log_so_far[-2].get("shift", 0) or 0
            if last_shift < 0 and prev_shift < 0:
                sustained_drop = True
    is_undecided = cd.get("is_undecided", False)
    round_num = cd.get("rounds", 0) + 1

    # ── Full-viewport loading overlay — visible regardless of scroll position ──
    # Uses fixed positioning so it covers the whole screen no matter where
    # the user was on the previous page
    round_num = cd.get("rounds", 0) + 1
    steps = [
        (0.15, "1/4 — Analysing bias"),
        (0.40, "2/4 — Explaining what this means for you"),
        (0.65, "3/4 — Generating a perspective"),
        (0.85, "4/4 — Crafting your challenge"),
    ]
    step_messages = [
        "Analysing your thinking pattern",
        "Explaining what this means for you",
        "Finding a perspective outside your thinking",
        "Crafting your Socratic challenge",
    ]

    # Render a fixed overlay that covers the entire screen
    overlay_ph = st.empty()
    progress_ph = st.empty()

    def render_overlay(message, step_label):
        overlay_ph.markdown(
            f"""
            <div style="
                position:fixed;top:0;left:0;width:100vw;height:100vh;
                background:#FDFBF7;z-index:9999;
                display:flex;flex-direction:column;
                align-items:center;justify-content:center;gap:24px;
            ">
                <p style="font-size:13px;font-weight:600;letter-spacing:2px;
                   text-transform:uppercase;color:#9A7B3A;margin:0">
                    Round {round_num} of {MAX_ROUNDS}
                </p>
                <div style="font-size:40px">🧠</div>
                <div style="display:flex;gap:10px;align-items:center">
                    <div style="width:10px;height:10px;border-radius:50%;background:#9A7B3A;
                         animation:capcs-pulse 1.4s ease-in-out 0s infinite"></div>
                    <div style="width:10px;height:10px;border-radius:50%;background:#7B1FA2;
                         animation:capcs-pulse 1.4s ease-in-out 0.2s infinite"></div>
                    <div style="width:10px;height:10px;border-radius:50%;background:#2E7D52;
                         animation:capcs-pulse 1.4s ease-in-out 0.4s infinite"></div>
                    <div style="width:10px;height:10px;border-radius:50%;background:#4A6FA5;
                         animation:capcs-pulse 1.4s ease-in-out 0.6s infinite"></div>
                </div>
                <p style="font-size:13px;letter-spacing:1.5px;text-transform:uppercase;
                   color:#9A7B3A;font-weight:600;margin:0">{message}</p>
                <p style="font-size:12px;color:#aaa;margin:0">{step_label}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    try:
        turn_num = round_num  # turn_num 1 = opening, 2+ = challenge with bias
        last_answer = cd.get("last_answer", "")
        emotion = cd.get("answer_signals", {}).get("emotion", "neutral")
        context = cd.get("context", "")

        if turn_num == 1:
            # ── Turn 1: Pure listening — one observation + one question ──────
            render_overlay("Thinking about what you've shared", "Getting ready...")

            opening = get_opening_question(
                cd["decision"], cd["options"],
                cd["confidence_before"], enriched_profile_str,
                context, longitudinal_text
            )

            cd["conversation_message"] = opening
            cd["bias_text"] = ""
            cd["explanation_text"] = ""
            cd["perspective_option"] = ""
            cd["perspective_why"] = ""
            cd["perspective_text"] = ""
            cd["question_text"] = opening  # the whole message is the question on turn 1
            st.session_state.current_decision = cd

        else:
            # ── Turn 2+: The spark — bias named, perspective offered ─────────
            render_overlay(step_messages[0], steps[0][1])

            full_response = get_challenge_response(
                cd["decision"], cd["options"], cd.get("leaning", ""),
                cd["confidence_before"], enriched_profile_str,
                history_text, last_answer, context, longitudinal_text,
                emotion=emotion, turn_num=turn_num,
                is_undecided=is_undecided,
                confidence_dropped=confidence_dropped,
                sustained_drop=sustained_drop
            )

            render_overlay(step_messages[2], steps[2][1])

            # Extract user-facing message and Supabase fields
            conversation_message = get_conversation_message(full_response)
            fields = extract_challenge_fields(full_response)

            render_overlay(step_messages[3], steps[3][1])

            option_name = fields["perspective_text"]

            # Similarity check on perspective option
            existing_opts = [o.lower() for o in split_options(cd.get("options", ""))]
            existing_opts += [o.lower() for o in st.session_state.all_options]

            def _too_similar(gen, existing):
                for e in existing:
                    gen_words = set(gen.split())
                    ex_words = set(e.split())
                    if not gen_words or not ex_words: continue
                    if len(gen_words & ex_words) / min(len(gen_words), len(ex_words)) > 0.6:
                        return True
                return False

            cd["conversation_message"] = conversation_message
            cd["bias_text"] = fields["bias_text"]
            cd["explanation_text"] = fields["explanation_text"]
            cd["perspective_option"] = option_name
            cd["perspective_why"] = ""
            cd["perspective_text"] = option_name
            cd["question_text"] = fields["question_text"]
            st.session_state.current_decision = cd

            if option_name and not _too_similar(option_name.lower(), existing_opts):
                if option_name not in st.session_state.all_options:
                    st.session_state.all_options.append(option_name)

        thinking_ph.empty() if 'thinking_ph' in dir() else None
        overlay_ph.empty()
        progress_ph.empty()

        st.session_state.phase = "challenge"
        st.rerun()

    except Exception as e:
        st.error(f"API error: {e}")
        st.stop()


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CHALLENGE LOOP
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "challenge":
    if not st.session_state.get("consent_given"):
        st.session_state.phase = "consent"; st.rerun()
    # Scroll to top whenever a new round renders
    scroll_to_top()
    cd = st.session_state.current_decision
    profile = load_profile()
    observed_profile = st.session_state.get("observed_profile", {})
    # Load bias corrections — always passed to format_profile so they work from session 1
    user_key_corr = st.session_state.get("user_key", "")
    bias_corrections = load_bias_corrections(user_key_corr) if user_key_corr else {}
    # Enrich the profile string with observed behaviour and corrections
    enriched_profile_str = format_profile(profile, observed_profile, bias_corrections)
    round_num = cd.get("rounds", 0) + 1
    sub_state = st.session_state.sub_state
    history_text = build_history(cd.get("rounds_log", []))
    # Build longitudinal context from past sessions (cached to avoid repeated DB calls)
    if not st.session_state.get("longitudinal_text"):
        past_sessions = load_log()
        past_completed = [h for h in past_sessions if h.get("completed_at")]
        st.session_state.longitudinal_text = build_longitudinal_context(past_completed)
    longitudinal_text = st.session_state.longitudinal_text

    # Detect confidence drop pattern
    # Within a round: confidence_before → confidence_after (stored as "confidence")
    # A drop means the round REDUCED the user's confidence.
    # Sustained drop = 2+ consecutive rounds where confidence decreased within the round.
    rounds_log_so_far = cd.get("rounds_log", [])
    confidence_dropped = False
    sustained_drop = False
    if len(rounds_log_so_far) >= 1:
        last_shift = rounds_log_so_far[-1].get("shift", 0) or 0
        if last_shift < 0:
            confidence_dropped = True
        # 2+ consecutive drops
        if len(rounds_log_so_far) >= 2:
            prev_shift = rounds_log_so_far[-2].get("shift", 0) or 0
            if last_shift < 0 and prev_shift < 0:
                sustained_drop = True

    CONFIDENCE_THRESHOLD = st.session_state.get("confidence_threshold", 75)
    is_undecided = cd.get("is_undecided", False)

    # Subtle abandon link — preserves session_state so user can resume by re-entering input
    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("← Pause session", key="pause_session_btn", help="Pause and view profile or settings — your progress will be kept"):
            st.session_state.phase = "input"
            st.rerun()

    st.progress(round_num / MAX_ROUNDS)
    label(f"Round {round_num} of {MAX_ROUNDS}")
    display_decision = cd.get("decision_short") or cd["decision"].split("\n\n[CONTEXT]")[0].strip()
    st.markdown(f"**Decision:** {display_decision}")
    if cd.get("options") and cd["options"] != cd["decision"]:
        st.caption(f"Options: {cd['options']}")
    if is_undecided:
        box("You've told us you're genuinely undecided — CAPCS will focus on what's blocking clarity rather than pushing you toward an option.", style="info")
    elif cd.get("leaning"):
        st.caption(f"Currently leaning: **{cd['leaning']}**")
    badge(cd["confidence_before"])
    if sustained_drop:
        box("⚠️ Your confidence has been dropping across multiple rounds. CAPCS is shifting focus — instead of new challenges, we'll help you identify what you already know.", style="warning")
    elif confidence_dropped:
        st.caption("⚠️ Your confidence dropped last round — this round will focus on simplifying your thinking.")
    st.divider()

    # ── PRESENT ──────────────────────────────────────────────────────────────
    if sub_state == "present":
        if "conversation_message" not in cd and "bias_text" not in cd:
            st.session_state.phase = "generating"
            st.rerun()

        # Build chat history from rounds log — show full conversation
        for r in cd.get("rounds_log", []):
            # CAPCS message
            capcs_msg = r.get("conversation_message") or r.get("question", "")
            if capcs_msg:
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(capcs_msg)
                # "What I noticed" expander after CAPCS messages that have bias
                bias_n = r.get("bias","").split("—")[0].strip()[:60]
                if bias_n and r.get("explanation"):
                    with st.expander("💡 What I noticed in your thinking", expanded=False):
                        st.markdown(f"**{bias_n}**")
                        st.markdown(r.get("explanation",""))
            # User's answer
            if r.get("answer"):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(r["answer"])

        # Current CAPCS message
        conversation_msg = cd.get("conversation_message") or cd.get("question_text", "")
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(conversation_msg)

        # "What I noticed" — collapsed, Turn 2+ only
        bias_name_short = cd.get("bias_text", "").split("—")[0].strip()[:60]
        if bias_name_short and cd.get("explanation_text"):
            with st.expander("💡 What I noticed in your thinking", expanded=False):
                st.markdown(f"**{bias_name_short}**")
                st.markdown(cd.get("explanation_text", ""))
                st.divider()
                st.caption("Does this resonate?")
                uk = st.session_state.get("user_key", "")
                existing_corrections = load_bias_corrections(uk) if uk else {}
                existing_corr = existing_corrections.get(bias_name_short, {})
                if existing_corr:
                    verdict_label = {
                        "accurate": "✅ You confirmed this",
                        "inaccurate": "❌ You said this didn't fit",
                        "partial": "🔶 You said this partially fits"
                    }.get(existing_corr.get("verdict", ""), "")
                    st.caption(verdict_label)
                else:
                    kb = f"bias_ack_{bias_name_short[:15].replace(' ','_')}_{round_num}"
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("✅ Resonates", key=f"{kb}_yes", use_container_width=True):
                            save_bias_correction(uk, bias_name_short, "accurate", "")
                            st.rerun()
                    with col2:
                        if st.button("🔶 Partially", key=f"{kb}_partial", use_container_width=True):
                            save_bias_correction(uk, bias_name_short, "partial", "")
                            st.rerun()
                    with col3:
                        if st.button("❌ Doesn't fit", key=f"{kb}_no", use_container_width=True):
                            save_bias_correction(uk, bias_name_short, "inaccurate", "")
                            st.rerun()

        # Previous follow-up exchanges this round
        if st.session_state.followup_exchanges:
            for exc in st.session_state.followup_exchanges:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(exc["question"])
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(exc["answer"])

        # ── Chat input for user response ──────────────────────────────────────
        inline_answer = st.chat_input(
            "Respond, push back, or think out loud...",
            key=f"chat_answer_{round_num}"
        )

        if inline_answer and inline_answer.strip():
            cd["user_answer"] = inline_answer.strip()
            with st.spinner("Reading your answer..."):
                signals = analyse_answer_quality(inline_answer, cd.get("question_text", ""))
                cd["answer_signals"] = signals
            st.session_state.current_decision = cd
            st.session_state.sub_state = "shifted"
            st.session_state.show_followup = False
            st.rerun()

    # ── SHIFTED ───────────────────────────────────────────────────────────────
    elif sub_state == "shifted":
        # Show the conversation so far
        for r in cd.get("rounds_log", []):
            capcs_msg = r.get("conversation_message") or r.get("question", "")
            if capcs_msg:
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(capcs_msg)
            if r.get("answer"):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(r["answer"])

        # Show current CAPCS message and user's answer
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(cd.get("conversation_message") or cd.get("question_text", ""))
        with st.chat_message("user", avatar="👤"):
            st.markdown(cd.get("user_answer", ""))

        # CAPCS asks a follow-through question
        previous_leaning = cd.get("leaning", "")
        is_still_undecided = cd.get("is_undecided", False)

        if is_still_undecided:
            followthrough = "What specifically is keeping you from deciding? Try to be concrete — not 'I'm not sure' but what would need to be true for you to feel clear."
        else:
            followthrough = f"Has this shifted how you're thinking, or are you still leaning towards {previous_leaning}? And how clear do you feel now?"

        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(followthrough)

        followup_answer = st.chat_input(
            "How has your thinking shifted (or not)?",
            key=f"chat_shifted_{round_num}"
        )

        if followup_answer and followup_answer.strip():
            # Detect shift from user's words
            shift_words = ["changed", "shifted", "different", "now think", "leaning towards",
                          "actually", "realised", "realize", "maybe", "perhaps", "instead"]
            hold_words = ["still", "same", "unchanged", "confirmed", "reinforced", "more sure",
                         "more confident", "definitely", "certain"]
            lower = followup_answer.lower()
            thinking_shifted = any(w in lower for w in shift_words) and not any(w in lower for w in hold_words)
            is_still_undecided_new = any(w in lower for w in ["still unsure", "still not sure", "still undecided", "still unclear", "don't know"])

            leaning_now = previous_leaning  # will be updated if user mentions a new option

            # Extract any new option from the follow-up using answer signals
            signals = cd.get("answer_signals", {})
            new_opt = signals.get("new_option", "")
            if new_opt and len(new_opt) > 2:
                leaning_now = new_opt
                thinking_shifted = True

            # Confidence slider
            st.markdown("")
            confidence_label_text = "How clear do you feel now? (0 = completely lost, 100 = decided)"
            st.caption(confidence_label_text)
            confidence_after = st.slider(
                "Confidence now", 0, 100,
                cd.get("confidence_before", 50), 5,
                label_visibility="collapsed",
                key=f"conf_after_{round_num}"
            )
            badge(confidence_after)
            shift = confidence_after - cd.get("confidence_before", 50)

            # ── Threshold and continuation logic ──────────────────────────────
            CONFIDENCE_THRESHOLD = st.session_state.get("confidence_threshold", 75)
            max_reached = round_num >= MAX_ROUNDS
            threshold_reached = confidence_after >= CONFIDENCE_THRESHOLD and not is_still_undecided_new

            if is_still_undecided_new and max_reached:
                btn = "✓ Complete session — still deciding"
            elif threshold_reached and not max_reached:
                st.divider()
                box(
                    f"<b>Your clarity has reached {confidence_after}% — above your threshold.</b><br>"
                    "Do you feel ready to decide, or would you like to keep exploring?",
                    style="insight"
                )
                col_done, col_more = st.columns(2)
                with col_done:
                    btn = f"✓ I feel clear enough ({confidence_after}%)"
                with col_more:
                    if st.button("→ Keep exploring", key=f"keep_challenging_{round_num}", use_container_width=True):
                        new_threshold = min(CONFIDENCE_THRESHOLD + 10, 92)
                        st.session_state.confidence_threshold = new_threshold
                        rounds_log = cd.get("rounds_log", [])
                        rounds_log.append({
                            "round": round_num, "round_number": round_num,
                            "timestamp": datetime.now().isoformat(),
                            "bias": cd.get("bias_text",""),
                            "explanation": cd.get("explanation_text",""),
                            "perspective": cd.get("perspective_text",""),
                            "question": cd.get("question_text",""),
                            "conversation_message": cd.get("conversation_message",""),
                            "followups": st.session_state.followup_exchanges,
                            "answer": cd.get("user_answer",""),
                            "answer_depth": cd.get("answer_signals",{}).get("depth",""),
                            "answer_emotion": cd.get("answer_signals",{}).get("emotion",""),
                            "answer_certainty": cd.get("answer_signals",{}).get("certainty",""),
                            "answer_key_signal": cd.get("answer_signals",{}).get("key_signal",""),
                            "shifted": thinking_shifted,
                            "still_undecided": is_still_undecided_new,
                            "how_shifted": followup_answer,
                            "leaning": leaning_now,
                            "confidence": confidence_after,
                            "shift": shift,
                            "confidence_shift": shift,
                        })
                        st.session_state.current_decision = {
                            "decision": cd["decision"], "options": cd["options"],
                            "leaning": leaning_now,
                            "is_undecided": is_still_undecided_new,
                            "confidence_before": confidence_after,
                            "confidence_start": cd["confidence_start"],
                            "timestamp": cd["timestamp"], "rounds": round_num,
                            "rounds_log": rounds_log,
                            "last_answer": followup_answer,
                            "context": cd.get("context",""),
                            "decision_short": cd.get("decision_short",""),
                        }
                        st.session_state.sub_state = "present"
                        st.session_state.followup_exchanges = []
                        st.session_state.phase = "generating"
                        st.rerun()
                btn = f"✓ I feel clear enough ({confidence_after}%)"
            elif max_reached:
                btn = "✓ Complete and see report"
            else:
                btn = "→ Continue"

            if st.button(btn, key=f"challenge_continue_btn_{round_num}", type="primary", use_container_width=True):
                if not followup_answer.strip():
                    st.warning("Say something about how your thinking has shifted.")
                else:
                    rounds_log = cd.get("rounds_log", [])
                    rounds_log.append({
                        "round": round_num, "round_number": round_num,
                        "timestamp": datetime.now().isoformat(),
                        "bias": cd.get("bias_text",""),
                        "explanation": cd.get("explanation_text",""),
                        "perspective": cd.get("perspective_text",""),
                        "question": cd.get("question_text",""),
                        "conversation_message": cd.get("conversation_message",""),
                        "followups": st.session_state.followup_exchanges,
                        "answer": cd.get("user_answer",""),
                        "answer_depth": cd.get("answer_signals",{}).get("depth",""),
                        "answer_emotion": cd.get("answer_signals",{}).get("emotion",""),
                        "answer_certainty": cd.get("answer_signals",{}).get("certainty",""),
                        "answer_key_signal": cd.get("answer_signals",{}).get("key_signal",""),
                        "shifted": thinking_shifted,
                        "still_undecided": is_still_undecided_new,
                        "how_shifted": followup_answer,
                        "leaning": leaning_now,
                        "confidence": confidence_after,
                        "shift": shift,
                        "confidence_shift": shift,
                    })
                    new_options = cd["options"]
                    if leaning_now and leaning_now not in new_options:
                        new_options += f" / {leaning_now}"
                    enriched_answer = followup_answer

                    # Session end conditions
                    if max_reached or (threshold_reached and btn.startswith("✓")):
                        # Save session
                        entry = {
                            "user_key": st.session_state.get("user_key",""),
                            "decision": cd.get("decision_short",""),
                            "context": cd.get("context",""),
                            "options": cd.get("options",""),
                            "all_options": st.session_state.all_options,
                            "final_choice": leaning_now or cd.get("leaning",""),
                            "confidence_start": cd.get("confidence_start", confidence_after),
                            "confidence_final": confidence_after,
                            "confidence_shift": confidence_after - cd.get("confidence_start", confidence_after),
                            "confidence_threshold": CONFIDENCE_THRESHOLD,
                            "confidence_trajectory": [r.get("confidence", 0) for r in rounds_log],
                            "rounds_completed": round_num,
                            "rounds_log": rounds_log,
                            "undecided_outcome": is_still_undecided_new,
                            "domain": classify_domain(cd.get("decision_short","")),
                            "timestamp": cd.get("timestamp",""),
                            "completed_at": datetime.now().isoformat(),
                        }
                        save_log(entry)
                        st.session_state.session_log.append(entry)
                        st.session_state.last_completed_entry = entry
                        st.session_state.show_feedback_page = True
                        st.session_state.phase = "feedback"
                    else:
                        # Continue to next round
                        st.session_state.current_decision = {
                            "decision": cd["decision"], "options": new_options,
                            "leaning": leaning_now,
                            "is_undecided": is_still_undecided_new,
                            "confidence_before": confidence_after,
                            "confidence_start": cd["confidence_start"],
                            "timestamp": cd["timestamp"], "rounds": round_num,
                            "rounds_log": rounds_log,
                            "last_answer": enriched_answer,
                            "context": cd.get("context",""),
                            "decision_short": cd.get("decision_short",""),
                        }
                        st.session_state.sub_state = "present"
                        st.session_state.followup_exchanges = []
                        st.session_state.show_followup = False
                        st.session_state.phase = "generating"
                    st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# FEEDBACK PAGE — shown immediately after session ends, before the report
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "feedback":
    if not st.session_state.get("consent_given"):
        st.session_state.phase = "consent"; st.rerun()
    scroll_to_top()
    last = st.session_state.get("last_completed_entry", {})
    session_id = last.get("id", "")

    # Fallback: if session_id missing, try to get it from Supabase
    # (can happen if save_log ran but the id wasn't stored back into entry)
    if not session_id and st.session_state.get("user_key"):
        try:
            sb = get_supabase()
            res = sb.table("sessions").select("id").eq(
                "user_key", st.session_state.user_key
            ).order("completed_at", desc=True).limit(1).execute()
            if res.data:
                session_id = res.data[0]["id"]
                if last:
                    last["id"] = session_id
                    st.session_state.last_completed_entry = last
        except Exception:
            pass

    st.success("Session complete. Quick question before you see your report.")
    st.divider()

    # Check if already submitted
    already_submitted = feedback_already_submitted(session_id) if session_id else False

    if already_submitted:
        st.info("✅ Feedback already submitted. Here's your report.")
        st.session_state.phase = "report"
        st.rerun()
    else:
        box(
            "How did this session feel? Five quick questions — takes under 60 seconds. "
            "Your answers help validate whether CAPCS actually helps people think more clearly.",
            style="info"
        )
        st.markdown("")

        clarity = st.select_slider(
            "This session helped me think more clearly about my decision.",
            options=["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
            value="Neutral", key="fb_clarity"
        )
        meta = st.select_slider(
            "I became aware of something about my thinking I hadn't noticed before.",
            options=["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
            value="Neutral", key="fb_meta"
        )
        bias_rel = st.select_slider(
            "The bias detected felt relevant to my actual situation.",
            options=["Not at all", "Slightly", "Moderately", "Mostly", "Very much"],
            value="Moderately", key="fb_bias"
        )
        novelty = st.select_slider(
            "The perspective offered was genuinely different from how I was thinking.",
            options=["Not at all", "Slightly", "Moderately", "Mostly", "Very much"],
            value="Moderately", key="fb_novelty"
        )
        useful = st.select_slider(
            "I would use CAPCS again for a future decision.",
            options=["Definitely not", "Probably not", "Maybe", "Probably yes", "Definitely yes"],
            value="Maybe", key="fb_useful"
        )
        open_resp = st.text_area(
            "Anything else? (optional)",
            placeholder="e.g. The second question made me realise I was avoiding the real issue...",
            height=80, key="fb_open"
        )

        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit and see report", key="feedback_submit_btn", type="primary", use_container_width=True):
                feedback = {
                    "cognitive_clarity": clarity,
                    "metacognitive_awareness": meta,
                    "bias_relevance": bias_rel,
                    "perspective_novelty": novelty,
                    "overall_usefulness": useful,
                    "open_response": open_resp.strip()
                }
                save_session_feedback(session_id, feedback)
                navigate_to("report")
        with col2:
            if st.button("Skip to report →", key="feedback_skip_btn", use_container_width=True):
                navigate_to("report")


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 3 — FINAL REPORT
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "report":
    if not st.session_state.get("consent_given"):
        st.session_state.phase = "consent"; st.rerun()
    # Primary: in-memory session_log
    # Fallback 1: last_completed_entry (survives navigation)
    # Fallback 2: load latest from Supabase
    if st.session_state.session_log:
        last = st.session_state.session_log[-1]
    elif st.session_state.get("last_completed_entry"):
        last = st.session_state.last_completed_entry
        st.session_state.session_log = [last]
    else:
        history = load_log()
        completed = [h for h in history if h.get("completed_at")]
        last = completed[-1] if completed else {}
    profile = load_profile()

    scroll_to_top()

    st.success("Session complete.")

    if last.get("undecided_outcome"):
        box(
            "<b>You completed this session still undecided — and that's useful information.</b><br>"
            "Pay attention to which questions felt hardest — that's usually where the real uncertainty lives.",
            style="warning"
        )

    # ── Three tabs — feedback is now its own page before the report ───────────
    tab_summary, tab_analysis, tab_rounds, tab_profile = st.tabs([
        "📊 Summary", "🎯 Analysis", "🔍 Rounds", "📊 Reasoning Profile"
    ])

    # ── TAB 1: Summary ─────────────────────────────────────────────────────────
    with tab_summary:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Starting confidence", f"{last.get('confidence_start',0)}%")
        with col2: st.metric("Final confidence", f"{last.get('confidence_final',0)}%")
        with col3:
            s = last.get("confidence_shift", 0)
            st.metric("Total shift", f"{s:+}%", delta_color="inverse")

        st.markdown(f"**Final decision:** {last.get('final_choice','—')}")
        st.markdown(f"**Rounds completed:** {last.get('rounds_completed',0)}")

        # Longitudinal summary card
        history_all = load_log()
        completed_all = [h for h in history_all if "confidence_final" in h]
        if len(completed_all) > 1:
            st.divider()
            avg_shift_all = sum(h.get("confidence_shift",0) for h in completed_all) / len(completed_all)
            all_biases = []
            for h in completed_all:
                for r in h.get("rounds_log",[]):
                    b = r.get("bias","").split("—")[0].strip()[:50]
                    if b: all_biases.append(b)
            top_bias = Counter(all_biases).most_common(1)[0][0] if all_biases else "none yet"
            domains = [h.get("domain","") for h in completed_all if h.get("domain")]
            top_domain = Counter(domains).most_common(1)[0][0] if domains else "various"
            mid = len(completed_all) // 2
            early_avg = sum(h.get("confidence_shift",0) for h in completed_all[:mid]) / max(mid,1)
            late_avg = sum(h.get("confidence_shift",0) for h in completed_all[mid:]) / max(len(completed_all)-mid,1)
            if late_avg < early_avg - 2:
                trend = "📈 Improving — your confidence is becoming better calibrated over time."
            elif late_avg > early_avg + 2:
                trend = "📉 Shifting up — you may be starting sessions with less certainty."
            else:
                trend = "➡️ Stable — your calibration pattern is consistent."
            box(
                f"<b>Across your {len(completed_all)} sessions:</b> average confidence shift is "
                f"<b>{avg_shift_all:+.1f}%</b>. Most recurring bias: <b>{top_bias}</b> "
                f"in <b>{top_domain}</b> decisions. {trend}",
                style="info"
            )

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 View reasoning profile", key="report_view_profile_btn", use_container_width=True):
                st.session_state.previous_phase = "report"
                navigate_to("reasoning_profile")
        with col2:
            if st.button("→ New decision", key="report_new_decision_btn", type="primary", use_container_width=True):
                st.session_state.phase = "input"
                st.session_state.sub_state = "present"
                st.session_state.current_decision = {}
                st.session_state.followup_exchanges = []
                st.session_state.all_options = []
                st.session_state.session_log = []
                st.session_state.pop("_report_analysis", None)
                st.session_state.input_session_counter = st.session_state.get("input_session_counter", 0) + 1
                st.rerun()

    # ── TAB 2: Analysis ────────────────────────────────────────────────────────
    with tab_analysis:
        final_conf = last.get("confidence_final", 0)
        threshold = last.get("confidence_threshold", 75)
        is_undecided_analysis = last.get("rounds_log") and all(
            not r.get("shifted") for r in last.get("rounds_log", [])
        ) and final_conf < threshold

        if is_undecided_analysis:
            box("You remained undecided after all rounds. Here's an analysis of what may be blocking clarity.", style="info")
            if st.button("🔍 Analyse what's blocking me", key="analyse_blocking_btn", type="primary", use_container_width=True):
                with st.spinner("Mapping what's blocking clarity..."):
                    prompt = f"""The user went through {last.get('rounds_completed',0)} rounds but remained undecided.

USER PROFILE: {format_profile(profile)}
DECISION: {last.get('decision','')}
OPTIONS: {', '.join(last.get('all_options',[]))}
FINAL CONFIDENCE: {final_conf}%
BIASES: {chr(10).join(f"- {r.get('bias','')}" for r in last.get('rounds_log',[]) if r.get('bias'))}

Write a warm honest analysis (max 200 words, second person, plain prose, no bullets, final sentence complete):
1. The real blocker
2. Whether more time or info is needed
3. One concrete action toward clarity
4. What this pattern reveals about them"""
                    stuck_analysis = ask_ai(prompt, 2048)
                box(stuck_analysis, style="highlight")
        else:
            box(
                "Based on your profile, the biases identified, the perspectives explored, "
                "and your answers — here is an analysis of which option makes the most sense for you.",
                style="info"
            )
            if st.session_state.get("_analysis_just_generated"):
                st.session_state["_analysis_just_generated"] = False

            if st.session_state.get("_report_analysis"):
                box(st.session_state["_report_analysis"], style="highlight")
            else:
                if st.button("🔍 Generate analysis", key="generate_analysis_btn", type="primary", use_container_width=True):
                    with st.spinner("Synthesising everything from your session..."):
                        explanation = get_session_recommendation(
                            last.get("final_choice", ""),
                            profile,
                            last.get("rounds_log", []),
                            last.get("decision", ""),
                            last.get("all_options", [])
                        )
                    st.session_state["_report_analysis"] = explanation
                    st.session_state["_analysis_just_generated"] = True
                    st.rerun()

    # ── TAB 3: Rounds ──────────────────────────────────────────────────────────
    with tab_rounds:
        rounds_log_display = last.get("rounds_log", [])
        if not rounds_log_display:
            st.caption("No rounds recorded.")
        for r in rounds_log_display:
            round_num_display = r.get("round") or r.get("round_number","?")
            confidence_val = r.get("confidence", 0) or 0
            shift_val = r.get("shift") or r.get("confidence_shift", 0) or 0
            shifted_label = "✅ Shifted" if r.get("shifted") else "➡️ Held position"
            with st.expander(f"Round {round_num_display} — {shifted_label} | Confidence: {confidence_val}% ({shift_val:+}%)"):
                if r.get("bias"): st.markdown(f"⚠️ **Bias:** {r['bias']}")
                if r.get("explanation"): st.markdown(f"📖 **Explanation:** {r['explanation']}")
                if r.get("perspective"): st.markdown(f"💡 **Perspective:** {r['perspective']}")
                if r.get("question"): st.markdown(f"❓ **Question:** {r['question']}")
                for fq in r.get("followups",[]):
                    st.markdown(f"↩️ *Follow-up:* {fq.get('question','')}")
                    st.markdown(f"   *Answer:* {fq.get('answer','')}")
                if r.get("answer"): st.markdown(f"💬 **Your answer:** {r['answer']}")
                if r.get("shifted"):
                    st.markdown(f"🔄 **How it shifted:** {r.get('how_shifted','')}")
                    st.markdown(f"→ **New leaning:** {r.get('leaning','—')}")

    # ── TAB 4: Reasoning Profile ───────────────────────────────────────────────
    with tab_profile:
        history_all = load_log()
        completed_rp = [h for h in history_all if "confidence_final" in h]

        if len(completed_rp) < 2:
            box("Complete at least 2 sessions to see your longitudinal reasoning profile.", style="info")
            if st.button("← Go to profile page", key="report_go_profile_btn", use_container_width=True):
                st.session_state.previous_phase = "report"
                navigate_to("reasoning_profile")
        else:
            # Calibration summary
            avg_shift = sum(h.get("confidence_shift",0) for h in completed_rp) / len(completed_rp)
            avg_start = sum(h.get("confidence_start",0) for h in completed_rp) / len(completed_rp)
            avg_final = sum(h.get("confidence_final",0) for h in completed_rp) / len(completed_rp)
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Sessions", len(completed_rp))
            with col2: st.metric("Avg. shift", f"{avg_shift:+.1f}%")
            with col3: st.metric("Avg. final confidence", f"{avg_final:.0f}%")

            # Top biases
            bias_counts = {}
            for h in completed_rp:
                for r in h.get("rounds_log",[]):
                    b = r.get("bias","").split("—")[0].strip()[:60]
                    if b: bias_counts[b] = bias_counts.get(b,0) + 1
            if bias_counts:
                st.markdown("**Most recurring biases:**")
                for b, c in sorted(bias_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                    st.markdown(f"- {b} ({c}×)")

            st.divider()
            if st.button("📊 Open full reasoning profile", key="report_open_profile_btn", use_container_width=True):
                st.session_state.previous_phase = "report"
                navigate_to("reasoning_profile")

# ════════════════════════════════════════════════════════════════════════════════
# PHASE 4 — REASONING PROFILE (longitudinal view)
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "reasoning_profile":
    if not st.session_state.get("consent_given"):
        st.session_state.phase = "consent"; st.rerun()
    profile = load_profile()
    history_all = load_log()
    completed = [h for h in history_all if "confidence_final" in h]

    scroll_to_top()
    st.markdown("<p style='font-size:22px;font-weight:700;margin:0'>📊 Your Reasoning Profile</p>", unsafe_allow_html=True)
    st.markdown("*How your thinking patterns look across all your sessions.*")
    st.divider()

    if len(completed) < 2:
        box("Complete at least 2 sessions to see your longitudinal reasoning profile.", style="info")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Calibration", "🧠 Biases", "🔄 Patterns", "👤 Profile"])

        with tab1:
            # ── Section 1: Calibration Card ───────────────────────────────────────
            st.markdown("### 🎯 Confidence Calibration")

            avg_shift = sum(h.get("confidence_shift",0) for h in completed) / len(completed)
            avg_start = sum(h.get("confidence_start",0) for h in completed) / len(completed)
            avg_final = sum(h.get("confidence_final",0) for h in completed) / len(completed)
            threshold_reached = sum(
                1 for h in completed
                if (h.get("confidence_final") or 0) >= (h.get("confidence_threshold") or 75)
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Sessions", len(completed))
            with col2: st.metric("Avg. start confidence", f"{avg_start:.0f}%")
            with col3: st.metric("Avg. final confidence", f"{avg_final:.0f}%")
            with col4: st.metric("Avg. shift", f"{avg_shift:+.1f}%")

            # Calibration trend
            mid = len(completed) // 2
            early = [h.get("confidence_shift",0) for h in completed[:mid]]
            late = [h.get("confidence_shift",0) for h in completed[mid:]]
            early_avg = sum(early)/max(len(early),1)
            late_avg = sum(late)/max(len(late),1)

            if late_avg < early_avg - 2:
                trend_msg = "📈 Your calibration is **improving** — you're arriving at sessions with more realistic starting confidence."
            elif late_avg > early_avg + 2:
                trend_msg = "📉 Your confidence shifts are **increasing** — you may be starting sessions with less certainty, which is normal as you tackle harder decisions."
            else:
                trend_msg = "➡️ Your calibration is **stable** — consistent pattern across sessions."

            st.markdown("")
            box(trend_msg, style="insight")

            # Threshold reached rate
            st.caption(f"Reached your personalised confidence threshold in {threshold_reached} of {len(completed)} sessions ({int(100*threshold_reached/len(completed))}%).")

        with tab2:
            # ── Bias Profile ─────────────────────────────────────────────────
            st.markdown("### 🧠 Bias Profile")

            bias_data = {}
            for h in completed:
                domain = h.get("domain","unknown")
                for r in h.get("rounds_log",[]):
                    raw = r.get("bias","")
                    if not raw: continue
                    name = raw.split("—")[0].strip()[:60]
                    if name not in bias_data:
                        bias_data[name] = {"count":0,"shifts":0,"domains":[],"shift_sum":0}
                    bias_data[name]["count"] += 1
                    bias_data[name]["domains"].append(domain)
                    if r.get("shifted"):
                        bias_data[name]["shifts"] += 1
                    # Handle both local (shift) and Supabase (confidence_shift) key names
                    shift_val = r.get("shift") or r.get("confidence_shift") or 0
                    bias_data[name]["shift_sum"] += abs(shift_val)

            if bias_data:
                sorted_biases = sorted(bias_data.items(), key=lambda x: x[1]["count"], reverse=True)
                user_key = st.session_state.get("user_key", "")
                corrections = load_bias_corrections(user_key)

                box(
                    "Help CAPCS learn about you — tell us whether each detected bias felt accurate. "
                    "This directly improves how the system challenges you in future sessions.",
                    style="info"
                )

                for bias_name, data in sorted_biases[:5]:
                    count = data["count"]
                    shift_rate = int(100 * data["shifts"] / max(count,1))
                    top_domains = Counter(data["domains"]).most_common(2)
                    domain_str = ", ".join(d[0] for d in top_domains if d[0])
                    indicator = "🔴" if count >= 3 else "🟡" if count == 2 else "🟢"

                    # Show existing correction if any
                    existing = corrections.get(bias_name, {})
                    correction_badge = ""
                    if existing.get("verdict") == "accurate":
                        correction_badge = " ✅ You confirmed this"
                    elif existing.get("verdict") == "inaccurate":
                        correction_badge = " ❌ You disputed this"
                    elif existing.get("verdict") == "partial":
                        correction_badge = " 🔶 You said this was partial"

                    with st.expander(f"{indicator} {bias_name} — {count}x detected | {shift_rate}% shift rate{correction_badge}"):
                        st.caption(f"Most common in: {domain_str or 'various'} decisions")
                        with st.spinner(f"Analysing {bias_name}..."):
                            analysis = get_bias_analysis(
                                bias_name, count, profile,
                                "; ".join(set(data["domains"]))
                            )
                        box(analysis, style="insight")

                        # ── Co-adaptive feedback UI ────────────────────────────────
                        st.markdown("---")
                        st.markdown("**Was this bias detection accurate for you?**")
                        st.caption("Your answer directly updates how CAPCS challenges you in future sessions.")

                        col1, col2, col3 = st.columns(3)
                        key_base = f"correction_{bias_name.replace(' ','_')[:20]}"

                        with col1:
                            if st.button("✅ Yes, accurate", key=f"{key_base}_yes", use_container_width=True):
                                save_bias_correction(user_key, bias_name, "accurate", "")
                                st.success("Noted — CAPCS will keep this in mind as a genuine pattern.")
                                st.rerun()
                        with col2:
                            if st.button("🔶 Partially", key=f"{key_base}_partial", use_container_width=True):
                                st.session_state[f"{key_base}_show_note"] = "partial"
                        with col3:
                            if st.button("❌ Not accurate", key=f"{key_base}_no", use_container_width=True):
                                st.session_state[f"{key_base}_show_note"] = "inaccurate"

                        # Show note input for partial/inaccurate
                        show_note = st.session_state.get(f"{key_base}_show_note", "")
                        if show_note in ("partial", "inaccurate"):
                            note_label = (
                                "What was inaccurate or misleading about this detection?"
                                if show_note == "inaccurate"
                                else "What was partially right and what was off?"
                            )
                            note = st.text_area(
                                note_label,
                                placeholder="e.g. This applies to my financial decisions but not career ones...",
                                height=80,
                                key=f"{key_base}_note_input"
                            )
                            if st.button("Save correction", key=f"{key_base}_save", type="primary"):
                                save_bias_correction(user_key, bias_name, show_note, note)
                                st.session_state.pop(f"{key_base}_show_note", None)
                                st.success("Correction saved — CAPCS will apply more caution with this bias in future sessions.")
                                st.rerun()

                        # Show existing note if there is one
                        if existing.get("note"):
                            st.caption(f"Your note: *\"{existing['note']}\"*")
            else:
                box("No bias data yet — complete more sessions.", style="info")

        with tab3:
            # ── Reasoning Patterns ───────────────────────────────────────────
            st.markdown("### 🔄 Reasoning Patterns")

            total_rounds = sum(h.get("rounds_completed",0) for h in completed)
            total_shifts = sum(1 for h in completed
                for r in h.get("rounds_log",[]) if r.get("shifted"))
            shift_rate = int(100 * total_shifts / max(total_rounds,1))

            # Average rounds to first shift — handle both key names
            first_shifts = []
            for h in completed:
                for r in h.get("rounds_log",[]):
                    if r.get("shifted"):
                        rn = r.get("round_number") or r.get("round") or 1
                        first_shifts.append(rn)
                        break
            avg_rounds_to_shift = sum(first_shifts)/len(first_shifts) if first_shifts else None

            # Domain distribution
            domains = [h.get("domain","unknown") for h in completed if h.get("domain")]
            domain_counts = Counter(domains).most_common()

            # Session duration trend
            durations = [h.get("session_duration_seconds") for h in completed if h.get("session_duration_seconds")]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Shift rate", f"{shift_rate}%")
                st.caption("% of rounds where thinking shifted")
            with col2:
                if avg_rounds_to_shift:
                    st.metric("Avg. rounds to first shift", f"{avg_rounds_to_shift:.1f}")
                else:
                    st.metric("Avg. rounds to first shift", "—")
            with col3:
                if durations:
                    avg_dur = sum(durations)/len(durations)
                    st.metric("Avg. session duration", f"{int(avg_dur//60)}m {int(avg_dur%60)}s")

            st.markdown("")
            if domain_counts:
                st.markdown("**Decision domains:**")
                for domain, count in domain_counts:
                    pct = int(100*count/len(completed))
                    st.markdown(f"- **{domain.capitalize()}** — {count} session{'s' if count>1 else ''} ({pct}%)")

            # Duration trend
            if len(durations) > 2:
                early_dur = sum(durations[:len(durations)//2])/max(len(durations)//2,1)
                late_dur = sum(durations[len(durations)//2:])/max(len(durations)-len(durations)//2,1)
                if late_dur < early_dur * 0.85:
                    box("⏱ Your sessions are getting shorter — you may be becoming more decisive over time.", style="perspective")
                elif late_dur > early_dur * 1.15:
                    box("⏱ Your sessions are getting longer — you may be bringing more complex decisions to CAPCS.", style="insight")

        with tab4:
            # ── Profile Evolution ─────────────────────────────────────────────
            st.markdown("### 👤 Profile")

            profile_display = load_profile()
            field_labels = {
                "age_range": "Age",
                "education_level": "Education",
                "education_field": "Field",
                "values": "Core values",
                "passions": "Passions",
                "current_situation": "Situation",
                "current_job": "Job / Role",
                "main_constraint": "Main constraint",
                "who_is_affected": "Who is affected",
                "decision_style": "Decision style",
                "known_bias": "Known blind spot",
                "success_criteria": "What makes a decision feel right",
            }

            if profile_display:
                for key, label in field_labels.items():
                    val = profile_display.get(key)
                    if val:
                        if isinstance(val, list):
                            val = ", ".join(val)
                        st.markdown(
                            f"<div style='margin:4px 0'>"
                            f"<span style='font-size:11px;color:#888;text-transform:uppercase;"
                            f"letter-spacing:1px'>{label}</span><br>"
                            f"<span style='font-size:14px'>{val}</span></div>",
                            unsafe_allow_html=True
                        )

            # Profile change history
            try:
                sb = get_supabase()
                user_key = st.session_state.get("user_key","")
                hist_res = sb.table("profile_history").select("*").eq("user_key", user_key).order("saved_at").execute()
                history_versions = hist_res.data or []
            except Exception:
                history_versions = []

            st.divider()
            if len(history_versions) <= 1:
                st.caption("Profile last updated: first setup. Updating after situation changes helps CAPCS challenge you more accurately.")
            else:
                st.markdown(f"**Updated {len(history_versions)-1} time{'s' if len(history_versions)-1 > 1 else ''} since setup**")
                # Show timeline of meaningful changes
                meaningful_fields = {
                    "decision_style": "Decision style",
                    "known_bias": "Known blind spot",
                    "main_constraint": "Main constraint",
                    "current_situation": "Situation",
                    "current_job": "Job / Role",
                    "values": "Core values",
                }
                for i in range(1, len(history_versions)):
                    prev = history_versions[i-1]
                    curr = history_versions[i]
                    date = curr.get("saved_at","")[:10]
                    diffs = []
                    for field, label in meaningful_fields.items():
                        pv = prev.get(field, "")
                        cv = curr.get(field, "")
                        if pv != cv and cv:
                            diffs.append(f"**{label}:** {pv or '—'} → {cv}")
                    if diffs:
                        with st.expander(f"Update on {date}", expanded=(i == len(history_versions)-1)):
                            for d in diffs:
                                st.markdown(f"- {d}")
                if not any(
                    history_versions[i-1].get(f) != history_versions[i].get(f)
                    for i in range(1, len(history_versions))
                    for f in meaningful_fields
                ):
                    st.caption("Profile was saved again but no answers changed.")

        st.divider()
    back_phase = st.session_state.get("previous_phase", "input")
    back_label = "← Back to report" if back_phase == "report" else "← Back"
    if st.button(back_label, key="reasoning_profile_back", type="primary", use_container_width=True):
        navigate_to(back_phase)
