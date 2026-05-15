# CAPCS — main entry point and orchestration layer

import streamlit as st
import os
import re
import json
import concurrent.futures
from datetime import datetime
from collections import Counter

from config import (MAX_ROUNDS, PROBE_TURNS, PROFILE_VERSION, CSS_STYLES,
                    UNDECIDED_INITIAL_LABEL, UNDECIDED_MID_SESSION_LABEL, OTHER_LABEL)
from database import (get_supabase, make_user_key, load_profile_for_user, load_profile,
                       save_profile, save_user, delete_profile, load_log, save_log, delete_log,
                       save_session_feedback, feedback_already_submitted,
                       save_bias_correction, load_bias_corrections)
from ai_prompts import (ask_ai, get_opening_question, get_probing_question, get_challenge_response,
                         extract_challenge_fields, get_conversation_message, get_bias, get_explanation,
                         get_perspective, get_question, get_followup_answer, is_followup_question,
                         get_consolidation_question, analyse_answer_quality, detect_response_type,
                         get_session_recommendation, get_bias_analysis, classify_domain, infer_options,
                         get_spark_message, extract_spark_fields, has_enough_signal)
from user_model import (compute_confidence_threshold, build_observed_profile, format_profile,
                         build_history, build_longitudinal_context, QUESTIONS)
from ui_helpers import (confidence_color, badge, label, box, thinking_animation,
                         navigate_to, scroll_to_top, scroll_to_chat_bottom,
                         inject_keepalive, split_options)

CAPCS_DIMENSIONS = """
DIMENSION 1 — Distortions in what the user wants and values:
- Sunk Cost Fallacy: persisting because of past investment, not future value
- Idealization Bias: the unchosen option looks unrealistically perfect from a distance
- Projection Bias: assuming future self will want what present self wants now
- Overconfidence Bias: overestimating how settled or correct their current view is
- Halo Effect: one attractive feature of an option colours the entire evaluation

DIMENSION 2 — Distortions in what the user fears or wants to protect:
- Anticipated Regret: driven by avoiding a future negative feeling, not present reality
- Loss Aversion: fear of losing outweighs equivalent potential gain
- Status Quo Bias: preferring current state because change feels inherently risky
- Omission Bias: believing inaction is safer or more moral than action

DIMENSION 3 — Distortions in what the user assumes:
- False Dichotomy: treating two options as exhaustive when more exist
- Overgeneralization: applying one specific experience as a universal rule
- Constraint Fixation: treating a changeable constraint as if it is fixed and immovable
- Availability Heuristic: overweighting vivid or recent examples when judging likelihood
- Confirmation Bias: seeking evidence that confirms existing beliefs, ignoring the rest
- Anchoring Bias: over-relying on one reference point for all subsequent judgment
- Social Proof Bias: deciding based on what others are doing, not on personal values
"""

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CASPER — Your Personal Thinking Companion", page_icon="👻", layout="centered")

# ── STYLES ─────────────────────────────────────────────────────────────────────
st.markdown(CSS_STYLES, unsafe_allow_html=True)
inject_keepalive()

# ── STARTING PHASE ─────────────────────────────────────────────────────────────
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
    "input_step": "context",
    "_input_context": "",
    "_input_decision": "",
    "_input_leaning": "",
    "_what_shifted": "",
    "_final_choice": "",
    "_confidence_after": None,
    "_confidence_confirmed": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val
# Note: consent gate is checked AFTER user_key is resolved from localStorage
# to avoid forcing returning users through consent/onboarding on every visit

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👻 CASPER")
    st.markdown("*Your personal thinking companion.*")

    # ── Home button ────────────────────────────────────────────────────────────
    if st.button("🏠 New session", key="sidebar_new_session", use_container_width=True, type="primary"):
        st.session_state.phase = "input"
        st.session_state.sub_state = "present"
        st.session_state.current_decision = {}
        st.session_state.followup_exchanges = []
        st.session_state.all_options = []
        st.session_state.session_log = []
        st.session_state["_what_shifted"] = ""
        st.session_state["_final_choice"] = ""
        st.session_state["_confidence_after"] = None
        st.session_state["_confidence_confirmed"] = False
        st.session_state.pop("_report_analysis", None)
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
    "<p style='font-size:24px;font-weight:700;margin:0;padding:0'>👻 CASPER</p>"
    "<p style='font-size:14px;color:#888;margin:0;padding:0'><em>Your personal thinking companion.</em></p>",
    unsafe_allow_html=True
)
st.divider()

# ── USER IDENTIFICATION ────────────────────────────────────────────────────────
# Persistent anonymous user_key stored in browser localStorage.
# On first visit: generate UUID, write to localStorage, save to Supabase.
# On return visits: read UUID from localStorage, restore session.

# ── USER IDENTIFICATION ────────────────────────────────────────────────────────
# Persistent anonymous user_key stored in browser localStorage and passed
# via ?uk= query param on every load.
#
# Recovery flow on session timeout / reconnect:
#   1. Streamlit reconnects with a fresh empty session state
#   2. ?uk= may still be in the URL from the previous session → picked up directly
#   3. If not in URL: JS reads localStorage and does a full redirect to add ?uk=
#      (window.location.replace is safe here — session state is already gone)

# Step 1: read ?uk= from current URL
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
            if row.get("consent_given_at"):
                st.session_state.consent_given = True
                starting = _starting_phase()
                if st.session_state.get("phase") in ("onboarding", "consent"):
                    st.session_state.phase = starting
    except Exception:
        pass

# Step 2: if ?uk= is not in URL and user is not identified, redirect using localStorage.
# window.location.replace triggers a full reload which Streamlit reads correctly.
# This only runs when session state is already blank (timeout/reconnect), so
# there is no session state to lose.
if not st.session_state.get("user_key"):
    st.markdown("""
<script>
(function() {
    const stored = localStorage.getItem('capcs_user_key');
    if (stored && !new URL(window.location.href).searchParams.get('uk')) {
        const url = new URL(window.location.href);
        url.searchParams.set('uk', stored);
        window.location.replace(url.toString());
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

    st.markdown("## 👻 Meet CASPER")
    st.markdown("*Your personal thinking companion.*")
    st.markdown("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🧠 Spots your blind spots**")
        st.caption("Identifies the cognitive bias shaping your thinking — the hidden pattern you can't see from inside it.")
    with col2:
        st.markdown("**💬 Asks, not tells**")
        st.caption("Three focused questions, then a bias reveal, then a direct challenge. You do the thinking. CASPER sharpens it.")
    with col3:
        st.markdown("**📊 Tracks your patterns**")
        st.caption("Builds a reasoning profile across sessions — so you learn how you decide, not just what to decide.")

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
                        st.query_params["uk"] = new_key
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
                        st.query_params["uk"] = recovered_key
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
    st.markdown("<p style='font-size:22px;font-weight:700;margin:0'>👻 Welcome to CASPER</p>", unsafe_allow_html=True)
    st.markdown("*Your personal thinking companion.*")
    st.divider()

    box(
        "<b>Before you start, please read this.</b><br><br>"
        "CASPER is a Socratic decision-making tool that uses AI to challenge your thinking. "
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
                st.session_state["_what_shifted"] = ""
                st.session_state["_final_choice"] = ""
                st.session_state["_confidence_after"] = None
                st.session_state["_confidence_confirmed"] = False
                st.session_state.input_session_counter = st.session_state.get("input_session_counter", 0) + 1
                st.rerun()
        st.divider()

    sc = st.session_state.get("input_session_counter", 0)

    st.markdown("**What are you deciding?**")
    st.caption("This stays private — CAPCS uses it to personalise the conversation.")
    st.markdown("")

    context_val = st.text_area(
        "Your situation",
        placeholder="What's your current situation? Any context useful for this decision — where you are in life, relevant constraints, what's at stake...",
        height=90,
        key=f"form_context_{sc}"
    )
    decision_val = st.text_area(
        "The decision",
        placeholder="What decision are you working through? e.g. I'm in Australia and don't know whether to keep travelling or go back to Europe.",
        height=90,
        key=f"form_decision_{sc}"
    )
    leaning_val = st.text_input(
        "Current leaning (optional)",
        placeholder="What are you leaning towards right now? Leave blank if genuinely undecided.",
        key=f"form_leaning_{sc}"
    )
    st.markdown("")
    st.caption("How clear do you feel going in? (0 = completely lost, 100 = basically decided)")
    confidence = st.slider("Clarity", 0, 100, 35, 5, label_visibility="collapsed", key=f"form_conf_{sc}")
    badge(confidence)
    st.markdown("")

    if st.button("→ Let's think this through", key="challenge_thinking_btn", type="primary", use_container_width=True):
        if not decision_val.strip():
            st.error("Please describe the decision you're working through.")
        else:
            decision_raw = decision_val.strip()
            context_raw = context_val.strip()
            leaning_raw = leaning_val.strip()

            detected = split_options(decision_raw)
            detected = [o for o in detected if o.strip() and len(o) < 80]
            options = " / ".join(detected) if detected else leaning_raw
            is_undecided = (not leaning_raw) or any(w in leaning_raw.lower() for w in
                ["undecided", "not sure", "don't know", "torn", "unsure", "no idea"])

            full_decision = f"{decision_raw}\n\n[CONTEXT]: {context_raw}" if context_raw else decision_raw
            past_sessions = load_log()
            st.session_state.confidence_threshold = compute_confidence_threshold(
                profile, past_sessions, starting_confidence=confidence
            )
            st.session_state.observed_profile = build_observed_profile(past_sessions)
            st.session_state.longitudinal_text = None
            st.session_state.all_options = detected or ([leaning_raw] if leaning_raw else [])
            st.session_state.current_decision = {
                "decision": full_decision,
                "decision_short": decision_raw,
                "context": context_raw,
                "options": options,
                "leaning": leaning_raw,
                "is_undecided": is_undecided,
                "confidence_before": confidence,
                "confidence_start": confidence,
                "timestamp": datetime.now().isoformat(),
                "rounds": 0,
                "rounds_log": [],
                "conversation_history": [],
                "capcs_state": "listening",
                "listening_answers": 0,
                "extra_listening": 0,
                "rejected_biases": [],
                "rejected_options": [],
                "confirmed_bias": "",
            }
            st.session_state.sub_state = "present"
            st.session_state.followup_exchanges = []
            st.session_state.input_step = "context"
            st.session_state.input_messages = []
            st.session_state.phase = "generating"; st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# GENERATING PHASE — dedicated loading page while AI generates the next challenge
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "generating":
    if not st.session_state.get("consent_given"):
        st.session_state.phase = "consent"; st.rerun()
    cd = st.session_state.current_decision
    profile = load_profile()
    observed_profile = st.session_state.get("observed_profile", {})
    user_key_corr = st.session_state.get("user_key", "")
    bias_corrections = load_bias_corrections(user_key_corr) if user_key_corr else {}
    enriched_profile_str = format_profile(profile, observed_profile, bias_corrections)
    history_text = "\n".join([
        f"{'CAPCS' if m['role'] == 'assistant' else 'USER'}: {m['content']}"
        for m in cd.get("conversation_history", [])
    ])
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

    # ── Render chat history inline so the page feels continuous ──────────────
    for r in cd.get("rounds_log", []):
        capcs_msg = r.get("conversation_message") or r.get("question", "")
        if capcs_msg:
            with st.chat_message("assistant", avatar="🧑‍🏫"):
                st.markdown(capcs_msg)
            if r.get("round_state") == "spark":
                bias_n = r.get("bias", "").split("—")[0].strip()[:60]
                if bias_n and r.get("explanation"):
                    with st.expander("💡 What I noticed in your thinking", expanded=False):
                        st.markdown(f"**{bias_n}**")
                        st.markdown(r.get("explanation", ""))
        ans = r.get("answer", "")
        if ans and not ans.startswith("["):
            with st.chat_message("user", avatar="👤"):
                st.markdown(ans)

    # Thinking indicator — shown while AI generates
    thinking_ph = st.empty()
    thinking_ph.markdown(
        '<div style="display:flex;gap:8px;padding:12px 0;align-items:center">'
        '<span style="font-size:22px">🧑‍🏫</span>'
        '<span style="color:#9A7B3A;font-size:14px;letter-spacing:1px">thinking...</span>'
        '</div>',
        unsafe_allow_html=True
    )

    scroll_to_chat_bottom()

    def render_overlay(message, step_label=None):
        pass  # no-op — kept so generating logic compiles unchanged

    try:
        capcs_state = cd.get("capcs_state", "listening")
        emotion = cd.get("answer_signals", {}).get("emotion", "neutral")
        context = cd.get("context", "")
        listening_answers = cd.get("listening_answers", 0)

        if capcs_state == "listening":
            if cd.get("extra_listening", 0) > 0:
                # Loop-back turn: use probing question with loop context
                message = get_probing_question(
                    cd["decision"], cd["options"], cd.get("leaning", ""),
                    cd["confidence_before"], enriched_profile_str,
                    history_text, cd.get("last_answer", ""), context, longitudinal_text,
                    turn_num=listening_answers + 1,
                    loop_context=cd.get("loop_context", "")
                )
            else:
                # Initial Q1/Q2/Q3: opening question determines which to ask via history
                message = get_opening_question(
                    cd["decision"], cd["options"],
                    cd["confidence_before"], enriched_profile_str,
                    context, longitudinal_text,
                    history=history_text
                )
            cd["conversation_message"] = message
            cd["bias_text"] = ""
            cd["explanation_text"] = ""
            cd["perspective_option"] = ""
            cd["perspective_text"] = ""
            cd["question_text"] = message
            cd.setdefault("conversation_history", []).append({"role": "assistant", "content": message})
            st.session_state.current_decision = cd

        elif capcs_state == "spark":
            spark_response = get_spark_message(
                cd.get("conversation_history", []),
                enriched_profile_str,
                context,
                cd.get("rejected_biases", [])
            )
            fields = extract_spark_fields(spark_response)
            message = fields["spark_message"] or spark_response.split("BIAS_NAME:")[0].strip()
            bias_name = fields["bias_name"]

            if not message or "SIGNAL_INSUFFICIENT" in spark_response:
                # Insufficient signal — drop back to listening for one more question
                cd["capcs_state"] = "listening"
                cd["extra_listening"] = 1
                cd["conversation_message"] = ""
                st.session_state.current_decision = cd
            else:
                # If message came back but structured fields weren't parsed, try inline extract
                if not bias_name:
                    import re as _re
                    _m = _re.search(
                        r'(?:called|is called|known as)\s+([A-Z][A-Za-z ]{2,50}?)(?:[.,]|$)',
                        message
                    )
                    bias_name = _m.group(1).strip() if _m else ""

                cd["conversation_message"] = message
                cd["bias_text"] = bias_name
                cd["explanation_text"] = fields["bias_explanation"]
                cd["perspective_option"] = ""
                cd["perspective_text"] = ""
                cd["question_text"] = ""
                cd.setdefault("conversation_history", []).append({
                    "role": "assistant", "content": message,
                    "state": "spark", "bias_name": bias_name,
                    "bias_explanation": fields["bias_explanation"],
                })
                st.session_state.current_decision = cd

        elif capcs_state == "counterattack":
            confirmed_bias = cd.get("confirmed_bias", "")
            full_response = get_challenge_response(
                cd["decision"], cd["options"], cd.get("leaning", ""),
                cd["confidence_before"], enriched_profile_str,
                history_text, cd.get("last_answer", ""), context, longitudinal_text,
                emotion=emotion, turn_num=round_num,
                is_undecided=is_undecided,
                confirmed_bias=confirmed_bias
            )
            conversation_message = get_conversation_message(full_response)
            fields = extract_challenge_fields(full_response)
            if not conversation_message and fields.get("question_text"):
                conversation_message = fields["question_text"]
            option_name = fields["perspective_text"]
            cd["conversation_message"] = conversation_message
            cd["bias_text"] = fields["bias_text"] or confirmed_bias
            cd["explanation_text"] = fields["explanation_text"]
            cd["perspective_option"] = option_name
            cd["perspective_text"] = option_name
            cd["question_text"] = fields["question_text"]
            cd.setdefault("conversation_history", []).append({"role": "assistant", "content": conversation_message})
            if option_name and option_name not in st.session_state.all_options:
                st.session_state.all_options.append(option_name)
            st.session_state.current_decision = cd

        thinking_ph.empty()
        st.session_state.phase = "challenge"
        st.rerun()

    except Exception as e:
        st.error(f"API error: {e}")
        st.stop()


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CHALLENGE LOOP  (state machine: listening → spark → counterattack → conviction)
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "challenge":
    if not st.session_state.get("consent_given"):
        st.session_state.phase = "consent"; st.rerun()

    cd = st.session_state.current_decision
    capcs_state = cd.get("capcs_state", "listening")
    user_key_corr = st.session_state.get("user_key", "")
    bias_corrections = load_bias_corrections(user_key_corr) if user_key_corr else {}
    round_num = cd.get("rounds", 0) + 1
    CONFIDENCE_THRESHOLD = st.session_state.get("confidence_threshold", 75)

    # Pause button
    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("← Pause session", key="pause_session_btn",
                     help="Pause and view profile or settings — your progress will be kept"):
            st.session_state.phase = "input"
            st.rerun()

    # ── Conversation history from rounds_log ──────────────────────────────────
    for r in cd.get("rounds_log", []):
        capcs_msg = r.get("conversation_message") or r.get("question", "")
        if capcs_msg:
            with st.chat_message("assistant", avatar="🧑‍🏫"):
                st.markdown(capcs_msg)
            if r.get("round_state") == "spark":
                bias_n = r.get("bias", "").split("—")[0].strip()[:60]
                if bias_n and r.get("explanation"):
                    with st.expander("💡 What I noticed in your thinking", expanded=False):
                        st.markdown(f"**{bias_n}**")
                        st.markdown(r.get("explanation", ""))
        ans = r.get("answer", "")
        if ans and not ans.startswith("["):
            with st.chat_message("user", avatar="👤"):
                st.markdown(ans)

    # ── Current CAPCS message ─────────────────────────────────────────────────
    conversation_msg = cd.get("conversation_message", "")
    if conversation_msg and capcs_state != "conviction":
        with st.chat_message("assistant", avatar="🧑‍🏫"):
            st.markdown(conversation_msg)

    scroll_to_chat_bottom()

    # ══════════════════════════════════════════════════════════════════════════
    # STATE: LISTENING
    # ══════════════════════════════════════════════════════════════════════════
    if capcs_state == "listening":
        if not conversation_msg:
            st.session_state.phase = "generating"; st.rerun()

        inline_answer = st.chat_input(
            "Respond, think out loud...",
            key=f"chat_listen_{round_num}"
        )
        if inline_answer and inline_answer.strip():
            answer = inline_answer.strip()
            conv_hist = cd.get("conversation_history", [])
            conv_hist.append({"role": "user", "content": answer})

            with st.spinner(""):
                signals = analyse_answer_quality(answer, cd.get("question_text", ""))

            listening_answers = cd.get("listening_answers", 0) + 1
            extra_listening = cd.get("extra_listening", 0)
            new_round = cd.get("rounds", 0) + 1

            rounds_log = cd.get("rounds_log", [])
            rounds_log.append({
                "round": new_round, "round_number": new_round,
                "round_state": "listening",
                "timestamp": datetime.now().isoformat(),
                "bias": "", "explanation": "", "perspective": "",
                "question": cd.get("question_text", ""),
                "conversation_message": cd.get("conversation_message", ""),
                "followups": [], "answer": answer,
                "answer_depth": signals.get("depth", ""),
                "answer_emotion": signals.get("emotion", ""),
                "answer_certainty": signals.get("certainty", ""),
                "answer_key_signal": signals.get("key_signal", ""),
                "shifted": False, "still_undecided": False,
                "how_shifted": "", "leaning": cd.get("leaning", ""),
                "confidence": cd.get("confidence_before", 50),
                "shift": 0, "confidence_shift": 0,
            })

            # Decide next state
            if extra_listening > 0:
                new_extra = extra_listening - 1
                next_state = "spark" if new_extra == 0 else "listening"
            elif listening_answers >= 3:
                next_state = "spark"
                new_extra = 0
            elif listening_answers >= 2:
                with st.spinner(""):
                    next_state = "spark" if has_enough_signal(conv_hist) else "listening"
                new_extra = 0
            else:
                next_state = "listening"
                new_extra = 0

            st.session_state.current_decision = {
                "decision": cd["decision"],
                "decision_short": cd.get("decision_short", ""),
                "context": cd.get("context", ""),
                "options": cd["options"],
                "leaning": cd.get("leaning", ""),
                "is_undecided": cd.get("is_undecided", False),
                "confidence_before": cd["confidence_before"],
                "confidence_start": cd["confidence_start"],
                "timestamp": cd["timestamp"],
                "rounds": new_round,
                "rounds_log": rounds_log,
                "last_answer": answer,
                "conversation_history": conv_hist,
                "capcs_state": next_state,
                "listening_answers": listening_answers,
                "extra_listening": new_extra,
                "rejected_biases": cd.get("rejected_biases", []),
                "rejected_options": cd.get("rejected_options", []),
                "confirmed_bias": cd.get("confirmed_bias", ""),
                "answer_signals": signals,
            }
            st.session_state.phase = "generating"
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STATE: SPARK
    # ══════════════════════════════════════════════════════════════════════════
    elif capcs_state == "spark":
        bias_name_short = cd.get("bias_text", "").split("—")[0].strip()[:60]
        # Only re-generate if the message itself is missing — not just the bias name
        if not conversation_msg:
            st.session_state.phase = "generating"; st.rerun()

        uk = user_key_corr
        existing_corrections = load_bias_corrections(uk) if uk else {}
        existing_corr = existing_corrections.get(bias_name_short, {})

        with st.expander("💡 What I noticed in your thinking", expanded=True):
            st.markdown(f"**{bias_name_short}**")
            st.markdown(cd.get("explanation_text", ""))
            if existing_corr:
                prev = {
                    "accurate": "✅ confirmed", "inaccurate": "❌ didn't fit",
                    "partial": "🔶 partially"
                }.get(existing_corr.get("verdict", ""), "")
                st.caption(f"Previously marked: {prev} — update:")
            kb = f"spark_{bias_name_short[:15].replace(' ','_')}_{round_num}"
            col1, col2, col3 = st.columns(3)
            with col1:
                resonates = st.button("✅ Resonates", key=f"{kb}_yes", use_container_width=True)
            with col2:
                partially = st.button("🔶 Partially", key=f"{kb}_partial", use_container_width=True)
            with col3:
                doesnt_fit = st.button("❌ Doesn't fit", key=f"{kb}_no", use_container_width=True)

        if resonates or partially:
            verdict = "accurate" if resonates else "partial"
            save_bias_correction(uk, bias_name_short, verdict, "")
            new_round = cd.get("rounds", 0) + 1
            rounds_log = cd.get("rounds_log", [])
            rounds_log.append({
                "round": new_round, "round_number": new_round,
                "round_state": "spark",
                "timestamp": datetime.now().isoformat(),
                "bias": cd.get("bias_text", ""), "explanation": cd.get("explanation_text", ""),
                "perspective": "", "question": "",
                "conversation_message": cd.get("conversation_message", ""),
                "followups": [], "answer": f"[{verdict}]",
                "answer_depth": "", "answer_emotion": "", "answer_certainty": "",
                "answer_key_signal": "", "shifted": False, "still_undecided": False,
                "how_shifted": "", "leaning": cd.get("leaning", ""),
                "confidence": cd.get("confidence_before", 50), "shift": 0, "confidence_shift": 0,
            })
            new_cd = dict(cd)
            new_cd["rounds"] = new_round
            new_cd["rounds_log"] = rounds_log
            new_cd["confirmed_bias"] = bias_name_short
            new_cd["capcs_state"] = "counterattack"
            st.session_state.current_decision = new_cd
            st.session_state.phase = "generating"
            st.rerun()

        if doesnt_fit:
            save_bias_correction(uk, bias_name_short, "inaccurate", "")
            new_round = cd.get("rounds", 0) + 1
            rounds_log = cd.get("rounds_log", [])
            rounds_log.append({
                "round": new_round, "round_number": new_round,
                "round_state": "spark_rejected",
                "timestamp": datetime.now().isoformat(),
                "bias": cd.get("bias_text", ""), "explanation": cd.get("explanation_text", ""),
                "perspective": "", "question": "",
                "conversation_message": cd.get("conversation_message", ""),
                "followups": [], "answer": "[inaccurate]",
                "answer_depth": "", "answer_emotion": "", "answer_certainty": "",
                "answer_key_signal": "", "shifted": False, "still_undecided": False,
                "how_shifted": "", "leaning": cd.get("leaning", ""),
                "confidence": cd.get("confidence_before", 50), "shift": 0, "confidence_shift": 0,
            })
            rejected = cd.get("rejected_biases", [])
            if bias_name_short not in rejected:
                rejected.append(bias_name_short)
            loop_ctx = (
                f"Previously tried bias: '{bias_name_short}'. "
                f"User said it doesn't resonate. "
                f"Explore a DIFFERENT angle — do not revisit this bias."
            )
            new_cd = dict(cd)
            new_cd["rounds"] = new_round
            new_cd["rounds_log"] = rounds_log
            new_cd["rejected_biases"] = rejected
            new_cd["extra_listening"] = 2
            new_cd["capcs_state"] = "listening"
            new_cd["loop_context"] = loop_ctx
            st.session_state.current_decision = new_cd
            st.session_state.phase = "generating"
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STATE: COUNTERATTACK
    # ══════════════════════════════════════════════════════════════════════════
    elif capcs_state == "counterattack":
        if not conversation_msg:
            st.session_state.phase = "generating"; st.rerun()

        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            ca_yes = st.button("Yes, this could work", key=f"ca_yes_{round_num}",
                               type="primary", use_container_width=True)
        with col2:
            ca_no = st.button("No, this doesn't work for me", key=f"ca_no_{round_num}",
                              use_container_width=True)
        scroll_to_chat_bottom()

        def _go_conviction():
            accepted_option = cd.get("perspective_text", "")
            if accepted_option and accepted_option not in st.session_state.all_options:
                st.session_state.all_options.append(accepted_option)
            conv_hist = cd.get("conversation_history", [])
            conv_hist.append({"role": "user", "content": "Yes"})
            new_round = cd.get("rounds", 0) + 1
            rounds_log = cd.get("rounds_log", [])
            rounds_log.append({
                "round": new_round, "round_number": new_round,
                "round_state": "counterattack",
                "timestamp": datetime.now().isoformat(),
                "bias": cd.get("bias_text", ""), "explanation": cd.get("explanation_text", ""),
                "perspective": cd.get("perspective_text", ""),
                "question": cd.get("question_text", ""),
                "conversation_message": cd.get("conversation_message", ""),
                "followups": [], "answer": "Yes",
                "answer_depth": "", "answer_emotion": "", "answer_certainty": "",
                "answer_key_signal": "", "shifted": True, "still_undecided": False,
                "how_shifted": "", "leaning": accepted_option or cd.get("leaning", ""),
                "confidence": cd.get("confidence_before", 50), "shift": 0, "confidence_shift": 0,
            })
            new_cd = dict(cd)
            new_cd["rounds"] = new_round
            new_cd["rounds_log"] = rounds_log
            new_cd["conversation_history"] = conv_hist
            new_cd["capcs_state"] = "conviction"
            st.session_state.current_decision = new_cd
            st.rerun()

        if ca_yes:
            _go_conviction()

        if ca_no:
            new_cd = dict(cd)
            new_cd["capcs_state"] = "counterattack_rejected"
            st.session_state.current_decision = new_cd
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STATE: COUNTERATTACK_REJECTED — ask why, then loop back to listening
    # ══════════════════════════════════════════════════════════════════════════
    elif capcs_state == "counterattack_rejected":
        with st.chat_message("assistant", avatar="🧑‍🏫"):
            st.markdown("What specifically doesn't work about it?")
        scroll_to_chat_bottom()

        why_not = st.chat_input("What doesn't work?", key=f"ca_reject_{round_num}")
        if why_not and why_not.strip():
            answer = why_not.strip()
            bias_tried   = cd.get("bias_text", "").split("—")[0].strip()[:60]
            perspective  = cd.get("perspective_text", "")
            conv_hist = cd.get("conversation_history", [])
            conv_hist.append({"role": "user", "content": answer})
            new_round = cd.get("rounds", 0) + 1
            rounds_log = cd.get("rounds_log", [])
            rejected_opts = cd.get("rejected_options", [])
            rejected_biases = cd.get("rejected_biases", [])
            if perspective and perspective not in rejected_opts:
                rejected_opts.append(perspective)
            # Keep the bias in rejected_biases so spark doesn't re-use it
            if bias_tried and bias_tried not in rejected_biases:
                rejected_biases.append(bias_tried)
            rounds_log.append({
                "round": new_round, "round_number": new_round,
                "round_state": "counterattack_rejected",
                "timestamp": datetime.now().isoformat(),
                "bias": cd.get("bias_text", ""), "explanation": cd.get("explanation_text", ""),
                "perspective": perspective, "question": "",
                "conversation_message": cd.get("conversation_message", ""),
                "followups": [], "answer": answer,
                "answer_depth": "", "answer_emotion": "", "answer_certainty": "",
                "answer_key_signal": "", "shifted": False, "still_undecided": False,
                "how_shifted": "", "leaning": cd.get("leaning", ""),
                "confidence": cd.get("confidence_before", 50), "shift": 0, "confidence_shift": 0,
            })
            loop_ctx = (
                f"Previously tried bias: '{bias_tried}'. "
                f"Option proposed: '{perspective}'. "
                f"User said it doesn't work because: '{answer}'. "
                f"Explore a DIFFERENT angle — do not revisit this bias or option."
            )
            st.session_state.current_decision = {
                "decision": cd["decision"],
                "decision_short": cd.get("decision_short", ""),
                "context": cd.get("context", ""),
                "options": cd["options"],
                "leaning": cd.get("leaning", ""),
                "is_undecided": cd.get("is_undecided", False),
                "confidence_before": cd["confidence_before"],
                "confidence_start": cd["confidence_start"],
                "timestamp": cd["timestamp"],
                "rounds": new_round,
                "rounds_log": rounds_log,
                "last_answer": answer,
                "conversation_history": conv_hist,
                "capcs_state": "listening",
                "listening_answers": cd.get("listening_answers", 0),
                "extra_listening": 2,
                "rejected_biases": rejected_biases,
                "rejected_options": rejected_opts,
                "confirmed_bias": "",
                "loop_context": loop_ctx,
                "answer_signals": {},
            }
            st.session_state.phase = "generating"
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STATE: CONVICTION  (5-step closing sequence)
    # ══════════════════════════════════════════════════════════════════════════
    elif capcs_state == "conviction":
        # Guarantee the CAPCS-proposed option is in the list regardless of
        # whether it was added during generating or _go_conviction.
        _perspective = cd.get("perspective_text", "")
        if _perspective and _perspective not in st.session_state.all_options:
            st.session_state.all_options.append(_perspective)

        final_choice      = st.session_state.get("_final_choice", "")
        what_shifted      = st.session_state.get("_what_shifted", "")
        confidence_stored = st.session_state.get("_confidence_after", None)
        conf_confirmed    = st.session_state.get("_confidence_confirmed", False)

        def _save_and_close(fc, ws, conf):
            shift = conf - cd.get("confidence_before", 50)
            rounds_log = cd.get("rounds_log", [])
            rounds_log.append({
                "round": round_num, "round_number": round_num,
                "round_state": "conviction",
                "timestamp": datetime.now().isoformat(),
                "bias": cd.get("bias_text", ""), "explanation": cd.get("explanation_text", ""),
                "perspective": cd.get("perspective_text", ""), "question": "",
                "conversation_message": cd.get("conversation_message", ""),
                "followups": [], "answer": ws,
                "answer_depth": "", "answer_emotion": "", "answer_certainty": "",
                "answer_key_signal": "", "shifted": True, "still_undecided": False,
                "how_shifted": ws, "leaning": fc or cd.get("leaning", ""),
                "confidence": conf, "shift": shift, "confidence_shift": shift,
            })
            entry = {
                "user_key": st.session_state.get("user_key", ""),
                "decision": cd.get("decision_short", ""),
                "context": cd.get("context", ""),
                "options": cd.get("options", ""),
                "all_options": st.session_state.all_options,
                "final_choice": fc or cd.get("leaning", ""),
                "confidence_start": cd.get("confidence_start", conf),
                "confidence_final": conf,
                "confidence_shift": conf - cd.get("confidence_start", conf),
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "confidence_trajectory": [r.get("confidence", 0) for r in rounds_log],
                "rounds_completed": sum(1 for r in rounds_log if r.get("round_state") == "spark") or 1,
                "rounds_log": rounds_log,
                "undecided_outcome": False,
                "domain": classify_domain(cd.get("decision_short", "")),
                "timestamp": cd.get("timestamp", ""),
                "completed_at": datetime.now().isoformat(),
            }
            save_log(entry)
            st.session_state.session_log.append(entry)
            st.session_state.last_completed_entry = entry
            st.session_state.pop("_what_shifted", None)
            st.session_state.pop("_final_choice", None)
            st.session_state["_confidence_after"] = None
            st.session_state["_confidence_confirmed"] = False
            st.session_state.phase = "feedback"
            st.rerun()

        # ── Step 1 — Option choice ────────────────────────────────────────────
        if not final_choice:
            with st.chat_message("assistant", avatar="🧑‍🏫"):
                st.markdown("Based on everything we've explored, which of these feels most true to where you've landed?")
            scroll_to_chat_bottom()
            st.markdown("")
            for i, opt in enumerate(st.session_state.get("all_options", [])):
                if st.button(opt, key=f"conviction_opt_{i}", use_container_width=True):
                    st.session_state["_final_choice"] = opt
                    st.rerun()

        # ── Step 2 — What shifted ─────────────────────────────────────────────
        elif not what_shifted:
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"Going with: **{final_choice}**")
            with st.chat_message("assistant", avatar="🧑‍🏫"):
                st.markdown(f"What made you land on **{final_choice}**?")
            scroll_to_chat_bottom()
            ws_input = st.chat_input("What shifted?", key=f"conviction_shifted_{round_num}")
            if ws_input and ws_input.strip():
                st.session_state["_what_shifted"] = ws_input.strip()
                st.rerun()

        # ── Step 3 — Confidence slider ────────────────────────────────────────
        elif confidence_stored is None:
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"Going with: **{final_choice}**")
            with st.chat_message("user", avatar="👤"):
                st.markdown(what_shifted)
            with st.chat_message("assistant", avatar="🧑‍🏫"):
                st.markdown("And how clear do you feel about this now?")
                live_conf = st.slider(
                    "Clarity", 0, 100,
                    cd.get("confidence_before", 50), 5,
                    label_visibility="collapsed",
                    key=f"conv_conf_{round_num}"
                )
                badge(live_conf)
            scroll_to_chat_bottom()
            if st.button("→ That's my number", key=f"conv_conf_set_{round_num}",
                         use_container_width=True):
                st.session_state["_confidence_after"] = live_conf
                st.rerun()

        # ── Step 4 — Confirm confidence ───────────────────────────────────────
        elif not conf_confirmed:
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"Going with: **{final_choice}**")
            with st.chat_message("user", avatar="👤"):
                st.markdown(what_shifted)
            with st.chat_message("assistant", avatar="🧑‍🏫"):
                st.markdown(
                    f"You're at **{confidence_stored}% clarity** about *{final_choice}*. "
                    f"Does that feel right?"
                )
            scroll_to_chat_bottom()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, that's where I am", key=f"conv_conf_yes_{round_num}",
                             type="primary", use_container_width=True):
                    st.session_state["_confidence_confirmed"] = True
                    st.rerun()
            with col2:
                if st.button("Let me adjust", key=f"conv_conf_adj_{round_num}",
                             use_container_width=True):
                    st.session_state["_confidence_after"] = None
                    st.rerun()

        # ── Step 5 — DDM threshold check ──────────────────────────────────────
        else:
            conf = confidence_stored
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"Going with: **{final_choice}**")
            with st.chat_message("user", avatar="👤"):
                st.markdown(what_shifted)
            scroll_to_chat_bottom()

            if conf >= CONFIDENCE_THRESHOLD:
                if st.button("✓ Complete session", key=f"conviction_complete_{round_num}",
                             type="primary", use_container_width=True):
                    _save_and_close(final_choice, what_shifted, conf)
            else:
                with st.chat_message("assistant", avatar="🧑‍🏫"):
                    st.markdown(
                        f"Your thinking hasn't fully settled yet — you're at **{conf}% clarity**. "
                        f"Do you want to explore this a bit further, or are you comfortable deciding here?"
                    )
                scroll_to_chat_bottom()
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Explore further", key=f"conv_explore_{round_num}",
                                 use_container_width=True):
                        st.session_state["_what_shifted"] = ""
                        st.session_state["_final_choice"] = ""
                        st.session_state["_confidence_after"] = None
                        st.session_state["_confidence_confirmed"] = False
                        confirmed = cd.get("confirmed_bias", "")
                        perspective = cd.get("perspective_text", "")
                        rej_biases = list(cd.get("rejected_biases", []))
                        rej_opts   = list(cd.get("rejected_options", []))
                        if confirmed and confirmed not in rej_biases:
                            rej_biases.append(confirmed)
                        if perspective and perspective not in rej_opts:
                            rej_opts.append(perspective)
                        loop_ctx = (
                            f"Previously tried bias: '{confirmed}'. "
                            f"Option proposed: '{perspective}'. "
                            f"User explored this but confidence was only {conf}% — not enough. "
                            f"Explore a DIFFERENT angle — do not revisit this bias or option."
                        )
                        new_cd = dict(cd)
                        new_cd["capcs_state"] = "listening"
                        new_cd["extra_listening"] = 2
                        new_cd["confirmed_bias"] = ""
                        new_cd["rejected_biases"] = rej_biases
                        new_cd["rejected_options"] = rej_opts
                        new_cd["loop_context"] = loop_ctx
                        st.session_state.current_decision = new_cd
                        st.session_state.phase = "generating"
                        st.rerun()
                with col2:
                    if st.button("I'm satisfied with this", key=f"conv_satisfied_{round_num}",
                                 type="primary", use_container_width=True):
                        _save_and_close(final_choice, what_shifted, conf)


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
            "I would use CASPER again for a future decision.",
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
        st.markdown(f"**Bias cycles completed:** {last.get('rounds_completed',0)}")

        # ── Conversation summary ───────────────────────────────────────────────
        rounds_log_s = last.get("rounds_log", [])
        biases_found = list(dict.fromkeys(
            r.get("bias","").split("—")[0].strip()
            for r in rounds_log_s if r.get("bias") and r.get("round_state") == "spark"
        ))
        options_proposed = list(dict.fromkeys(
            r.get("perspective","")
            for r in rounds_log_s if r.get("perspective") and r.get("round_state") == "counterattack"
        ))
        if biases_found or options_proposed:
            bias_str = " and ".join(biases_found) if biases_found else "a cognitive pattern"
            opt_str  = " and ".join(f'"{o}"' for o in options_proposed if o) if options_proposed else "an alternative path"
            summary_txt = (
                f"CAPCS identified <b>{bias_str}</b> as the main pattern shaping your thinking. "
                f"The session proposed {opt_str} as a way to break that pattern. "
                f"You landed on <b>{last.get('final_choice','—')}</b> with "
                f"<b>{last.get('confidence_final',0)}% clarity</b>."
            )
            box(summary_txt, style="info")

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
            _tab2_ph = st.empty()
            _tab2_ph.caption("⏳ Loading bias profile...")

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

            _tab2_ph.empty()
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
            with st.spinner("Loading profile history..."):
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
