# CASPER — main entry point and orchestration layer

import streamlit as st
import os
import re
import json
from datetime import datetime
from collections import Counter

from config import (MAX_ROUNDS, PROBE_TURNS, PROFILE_VERSION, CSS_STYLES,
                    UNDECIDED_INITIAL_LABEL, UNDECIDED_MID_SESSION_LABEL, OTHER_LABEL)
from database import (get_supabase, make_user_key, load_profile_for_user, load_profile,
                       save_profile, save_user, delete_profile, load_log, save_log, delete_log,
                       save_session_feedback, feedback_already_submitted,
                       save_bias_correction, load_bias_corrections)
from ai_prompts import (ask_ai, get_opening_question, get_probing_question, get_challenge_response,
                         extract_challenge_fields, get_conversation_message, get_followup_answer,
                         is_followup_question, get_consolidation_question, analyse_answer_quality,
                         get_session_recommendation, get_bias_analysis, classify_domain,
                         get_spark_message, extract_spark_fields,
                         get_personalised_suggestions, get_partial_probe,
                         identify_candidate_biases, get_disambiguation_question)
from user_model import (compute_confidence_threshold, build_observed_profile, format_profile,
                         build_longitudinal_context, QUESTIONS)
from ui_helpers import (confidence_color, badge, label, box, thinking_animation,
                         navigate_to, scroll_to_top, scroll_to_chat_bottom,
                         inject_keepalive, split_options)

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
    "_ca_partial_mode": False,
    "_ca_partial_conf": None,
    "_ca_pending_conf": None,
    "_conviction_suggestions": "",
    "_listening_clarification": None,
    "_partial_probe_question": "",
    "_ca_explore_above": False,
    "_recently_used_biases": [],
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
        st.session_state["_ca_partial_mode"] = False
        st.session_state["_ca_partial_conf"] = None
        st.session_state["_ca_pending_conf"] = None
        st.session_state["_ca_explore_above"] = False
        st.session_state["_partial_probe_question"] = ""
        st.session_state["_conviction_suggestions"] = ""
        st.session_state["_listening_clarification"] = None
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
        st.session_state["user_key"] = ""
        st.session_state["cached_profile"] = {}
        st.query_params.clear()
        # Flag tells the next render to remove localStorage instead of restoring from it
        st.session_state["_skip_ls_restore"] = True
        st.rerun()

    st.divider()
    with st.expander("📄 Privacy & Data", expanded=False):
        st.markdown("""
**What we collect**
Anonymised decision text, answers, confidence levels, and session feedback. Your display name is stored to personalise your experience — it is never shared or used outside CASPER.

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
    # Rerun so the sidebar re-renders with the now-populated user state.
    # The condition above only fires when user_key was blank, so no loop risk.
    st.rerun()

# Step 2: if ?uk= is not in URL and user is not identified, redirect using localStorage.
# window.location.replace triggers a full reload which Streamlit reads correctly.
# This only runs when session state is already blank (timeout/reconnect), so
# there is no session state to lose.
if not st.session_state.get("user_key"):
    if st.session_state.pop("_skip_ls_restore", False):
        # User just cleared their data — actively remove the localStorage key in this
        # render so the restoration check below cannot race and restore the old identity.
        st.markdown(
            "<script>localStorage.removeItem('capcs_user_key');</script>",
            unsafe_allow_html=True
        )
    else:
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
        st.caption("Identifies the hidden thinking pattern distorting your decision — before you even notice it's there.")
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
                    existing = sb.table("users").select("user_key, consent_given_at").eq("user_key", new_key).execute()
                    existing_rows = existing.data or []
                    consented = any(r.get("consent_given_at") for r in existing_rows)
                    if existing_rows and consented:
                        st.warning("This username + PIN combination already exists. Use the **Returning user** tab to log in, or choose a different combination.")
                    else:
                        # Upsert: works whether or not the row exists already
                        sb.table("users").upsert({
                            "user_key": new_key,
                            "display_name": display_name,
                        }, on_conflict="user_key").execute()
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
        with st.form("recover_form"):
            ret_username = st.text_input(
                "Username",
                placeholder="Your username",
            )
            ret_pin = st.text_input(
                "PIN",
                placeholder="Your PIN",
                type="password",
            )
            submitted = st.form_submit_button("→ Recover my session", type="primary", use_container_width=True)
        if submitted:
            if not ret_username.strip() or not ret_pin.strip():
                st.error("Please enter both your username and PIN.")
            else:
                recovered_key = make_user_key(ret_username, ret_pin)
                try:
                    sb = get_supabase()
                    res = sb.table("users").select("display_name, consent_given_at").eq("user_key", recovered_key).execute()
                    rows = res.data or []
                    if rows:
                        # Prefer a row with consent_given_at set (handles duplicate-row edge case)
                        row = next((r for r in rows if r.get("consent_given_at")), rows[0])
                        st.session_state.user_key = recovered_key
                        st.session_state.display_name = row.get("display_name") or ret_username.strip()
                        if row.get("consent_given_at"):
                            st.session_state.consent_given = True
                            st.session_state.phase = _starting_phase()
                        else:
                            st.session_state.phase = "consent"
                        st.query_params["uk"] = recovered_key
                        st.markdown(f"<script>localStorage.setItem('capcs_user_key', '{recovered_key}');</script>", unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.error("No account found for this username + PIN. Double-check your credentials — usernames are case-insensitive but PINs are case-sensitive.")
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
    if target == "reasoning_profile":
        st.session_state["_load_log_dirty"] = True
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

    st.markdown("### What data CASPER collects")
    st.markdown("""
CASPER collects the following data when you use it:
- The decisions you describe and the context you provide
- Your answers to Socratic challenges
- Your confidence levels and how they change during a session
- The cognitive biases detected in your reasoning
- Your post-session feedback (if you choose to submit it)
- An anonymous ID generated from your username + PIN — the original credentials are never stored, only the hash
""")

    st.markdown("### How your data is used")
    st.markdown("""
Your data is used for:
- Personalising your experience within CASPER (the system learns your reasoning patterns)
- Improving the tool based on how people use it
- Potential anonymised research and publication in cognitive science contexts

**Your data will never be shared with third parties and will never be used commercially.**
""")

    st.markdown("### Your rights")
    st.markdown("""
- You can delete all your data at any time using the **Clear my data** button in the sidebar
- You can stop using CASPER at any time with no consequence
- Your display name is stored to personalise your experience and is never shared or sold
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
        "→ I agree — continue to CASPER",
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
        btn = "Next →" if step < total - 1 else "Start CASPER →"
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
    st.caption("This stays private — CASPER uses it to personalise the conversation.")
    st.markdown("")

    _decision_examples = [
        "e.g. I've been offered a promotion that means relocating — should I take it or stay in my current role?",
        "e.g. I'm deciding whether to do a Master's degree now or gain work experience first.",
        "e.g. I have two job offers — one pays more but feels less interesting, the other is exciting but risky.",
        "e.g. I'm thinking about leaving my stable job to freelance full-time — should I make the jump?",
        "e.g. I've saved enough for a deposit — should I buy now or keep renting and invest instead?",
        "e.g. My partner and I disagree about whether to move cities — how do I think through what I actually want?",
        "e.g. I'm considering going back to study but I'm not sure the cost and time are worth it.",
        "e.g. Should I end a friendship that's started to feel one-sided, or give it more time?",
        "e.g. I've been offered a chance to start a business with a friend — should I do it or is it too risky?",
        "e.g. I don't know whether to stay in my current relationship or walk away.",
    ]
    _decision_placeholder = _decision_examples[sc % len(_decision_examples)]

    context_val = st.text_area(
        "Your situation",
        placeholder="What's your current situation? Any context useful for this decision — where you are in life, relevant constraints, what's at stake...",
        height=90,
        key=f"form_context_{sc}"
    )
    decision_val = st.text_area(
        "The decision",
        placeholder=_decision_placeholder,
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
                "counterattack_exchanges": [],
                "pre_identified_bias": "",
                "disambiguation_question": "",
                "bias_candidates": [],
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
        f"{'CASPER' if m['role'] == 'assistant' else 'USER'}: {m['content']}"
        for m in cd.get("conversation_history", [])
    ])
    if not st.session_state.get("longitudinal_text"):
        past_sessions = load_log()
        past_completed = [h for h in past_sessions if h.get("completed_at")]
        st.session_state.longitudinal_text = build_longitudinal_context(past_completed)
        # Extract biases used in the last 3 completed sessions for longitudinal diversity
        recent_biases = list(dict.fromkeys(
            r.get("bias", "").split("—")[0].strip()
            for s in past_completed[-3:]
            for r in s.get("rounds_log", [])
            if r.get("bias") and r.get("round_state") in ("spark", "counterattack")
        ))
        st.session_state["_recently_used_biases"] = [b for b in recent_biases if b]
    longitudinal_text = st.session_state.longitudinal_text or ""

    rounds_log_so_far = cd.get("rounds_log", [])
    # Detect rising uncertainty via answer_certainty signals from recent listening
    # rounds. Per-round confidence slider isn't updated mid-session, so this is
    # the best available proxy for "user is getting more confused, not less."
    recent_listening = [
        r for r in rounds_log_so_far if r.get("round_state") == "listening"
    ][-3:]
    low_certainty_count = sum(
        1 for r in recent_listening
        if r.get("answer_certainty") in ("low", "hedging")
    )
    confidence_dropped = low_certainty_count >= 2
    sustained_drop = low_certainty_count >= 3
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
                    expander_title = f"What is {bias_n}?" if bias_n else "What is this pattern?"
                    with st.expander(expander_title, expanded=True):
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

    try:
        capcs_state = cd.get("capcs_state", "listening")
        emotion = cd.get("answer_signals", {}).get("emotion", "neutral")
        context = cd.get("context", "")
        listening_answers = cd.get("listening_answers", 0)

        if capcs_state == "listening":
            if sustained_drop and listening_answers >= 2:
                message = get_consolidation_question(
                    cd["decision"], cd.get("leaning", ""),
                    rounds_log_so_far, enriched_profile_str, history_text
                )
            elif cd.get("disambiguation_question"):
                # Pre-generated disambiguation question — use directly, then clear
                message = cd["disambiguation_question"]
                cd["disambiguation_question"] = ""
            elif cd.get("extra_listening", 0) > 0:
                # Loop-back turn: use probing question with loop context
                message = get_probing_question(
                    cd["decision"], cd["options"], cd.get("leaning", ""),
                    cd["confidence_before"], enriched_profile_str,
                    history_text, cd.get("last_answer", ""), context, longitudinal_text,
                    turn_num=listening_answers + 1,
                    loop_context=cd.get("loop_context", "")
                )
            else:
                # Initial Q1/Q2/Q3
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
            bias_resonance_val = cd.get("bias_resonance", "full") or "full"
            existing_conv_msg = cd.get("conversation_message", "")
            existing_bias_name = cd.get("bias_text", "").split("—")[0].strip()[:60]

            if existing_conv_msg and existing_bias_name:
                # Resonance changed after initial spark — only regenerate counterattack
                bias_name = existing_bias_name
                full_ca = get_challenge_response(
                    cd["decision"], cd["options"], cd.get("leaning", ""),
                    cd.get("confidence_before", cd.get("confidence_start", 50)),
                    enriched_profile_str, history_text,
                    cd.get("last_answer", ""), context, longitudinal_text,
                    emotion=emotion, turn_num=round_num,
                    is_undecided=is_undecided,
                    confirmed_bias=bias_name,
                    bias_resonance=bias_resonance_val
                )
                ca_msg = get_conversation_message(full_ca)
                ca_fields = extract_challenge_fields(full_ca)
                cd["counterattack_message"] = ca_msg or ""
                cd["perspective_text"] = ca_fields["perspective_text"] or cd.get("perspective_text", "")
                if ca_fields["perspective_text"] and ca_fields["perspective_text"] not in st.session_state.all_options:
                    st.session_state.all_options.append(ca_fields["perspective_text"])
                st.session_state.current_decision = cd
            else:
                spark_response = get_spark_message(
                    cd.get("conversation_history", []),
                    enriched_profile_str,
                    context,
                    cd.get("rejected_biases", []),
                    recently_used_biases=st.session_state.get("_recently_used_biases", []),
                    pre_identified_bias=cd.get("pre_identified_bias", "")
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
                    # Generate counterattack immediately so spark state shows it inline
                    if bias_name:
                        full_ca = get_challenge_response(
                            cd["decision"], cd["options"], cd.get("leaning", ""),
                            cd.get("confidence_before", cd.get("confidence_start", 50)),
                            enriched_profile_str, history_text,
                            cd.get("last_answer", ""), context, longitudinal_text,
                            emotion=emotion, turn_num=round_num,
                            is_undecided=is_undecided,
                            confirmed_bias=bias_name,
                            bias_resonance=bias_resonance_val
                        )
                        ca_msg = get_conversation_message(full_ca)
                        ca_fields = extract_challenge_fields(full_ca)
                        cd["counterattack_message"] = ca_msg or ""
                        cd["perspective_text"] = ca_fields["perspective_text"] or ""
                        if ca_fields["perspective_text"] and ca_fields["perspective_text"] not in st.session_state.all_options:
                            st.session_state.all_options.append(ca_fields["perspective_text"])
                    st.session_state.current_decision = cd

        thinking_ph.empty()
        st.session_state.phase = "challenge"
        st.rerun()

    except Exception as e:
        st.error(f"API error: {e}")
        st.stop()


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CHALLENGE LOOP  (state machine: listening → spark → conviction)
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "challenge":
    if not st.session_state.get("consent_given"):
        st.session_state.phase = "consent"; st.rerun()

    cd = st.session_state.current_decision
    capcs_state = cd.get("capcs_state", "listening")
    user_key_corr = st.session_state.get("user_key", "")
    bias_corrections = load_bias_corrections(user_key_corr) if user_key_corr else {}
    profile = load_profile()
    observed_profile = st.session_state.get("observed_profile", {})
    enriched_profile_str = format_profile(profile, observed_profile, bias_corrections)
    context = cd.get("context", "")
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
                    expander_title = f"What is {bias_n}?" if bias_n else "What is this pattern?"
                    with st.expander(expander_title, expanded=True):
                        st.markdown(f"**{bias_n}**")
                        st.markdown(r.get("explanation", ""))
        ans = r.get("answer", "")
        if ans and not ans.startswith("["):
            with st.chat_message("user", avatar="👤"):
                st.markdown(ans)

    # ── Current CASPER message ────────────────────────────────────────────────
    # Spark and conviction states render their own message blocks
    conversation_msg = cd.get("conversation_message", "")
    if conversation_msg and capcs_state not in ("conviction", "spark"):
        with st.chat_message("assistant", avatar="🧑‍🏫"):
            st.markdown(conversation_msg)

    scroll_to_chat_bottom()

    # ══════════════════════════════════════════════════════════════════════════
    # STATE: LISTENING
    # ══════════════════════════════════════════════════════════════════════════
    if capcs_state == "listening":
        if not conversation_msg:
            st.session_state.phase = "generating"; st.rerun()

        # ── Clarification exchange: user asked a question, CAPCS clarified ────
        clarification_state = st.session_state.get("_listening_clarification")
        answer = None

        if clarification_state:
            with st.chat_message("user", avatar="👤"):
                st.markdown(clarification_state["user_q"])
            with st.chat_message("assistant", avatar="🧑‍🏫"):
                st.markdown(clarification_state["capcs_reply"])
            scroll_to_chat_bottom()
            real_input = st.chat_input(
                "Respond, think out loud... or ask a question if something isn't clear",
                key=f"chat_clarify_{round_num}"
            )
            if real_input and real_input.strip():
                st.session_state["_listening_clarification"] = None
                answer = real_input.strip()
                conv_hist = cd.get("conversation_history", [])
                conv_hist.append({"role": "user", "content": answer})
        else:
            inline_answer = st.chat_input(
                "Respond, think out loud... or ask a question if something isn't clear",
                key=f"chat_listen_{round_num}"
            )
            if inline_answer and inline_answer.strip():
                raw = inline_answer.strip()
                if is_followup_question(raw):
                    # User asked a question — clarify without advancing state
                    with st.spinner(""):
                        clarification = get_followup_answer(
                            cd.get("conversation_message", ""),
                            raw,
                            cd.get("decision", ""),
                            enriched_profile_str,
                            "\n".join(
                                f"{'CAPCS' if m['role'] == 'assistant' else 'USER'}: {m['content']}"
                                for m in cd.get("conversation_history", [])
                            )
                        )
                    conv_h = list(cd.get("conversation_history", []))
                    conv_h.append({"role": "user", "content": raw})
                    conv_h.append({"role": "assistant", "content": clarification})
                    new_cd = dict(cd)
                    new_cd["conversation_history"] = conv_h
                    st.session_state.current_decision = new_cd
                    st.session_state["_listening_clarification"] = {
                        "user_q": raw,
                        "capcs_reply": clarification,
                    }
                    st.rerun()
                answer = raw
                conv_hist = cd.get("conversation_history", [])
                conv_hist.append({"role": "user", "content": answer})

        if answer is None:
            st.stop()

        # ── Normal answer processing ──────────────────────────────────────────
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
            # If the user mentioned a new concrete option in their answer, surface it
            new_opt = signals.get("new_option", "").strip()
            if new_opt and new_opt not in st.session_state.all_options:
                st.session_state.all_options.append(new_opt)

            # ── Inductive bias identification ─────────────────────────────────
            # All three questions must be answered before the diagnostic runs.
            # If two biases are too close after Q3, one disambiguation question
            # (Q4) is asked; after that the winner is used for spark.
            pre_identified = cd.get("pre_identified_bias", "")
            new_extra = 0

            if extra_listening > 0:
                # Disambiguation question was just answered — final identification
                new_extra = extra_listening - 1
                if new_extra == 0:
                    with st.spinner(""):
                        final_cands = identify_candidate_biases(
                            conv_hist, enriched_profile_str, context
                        )
                    if final_cands:
                        pre_identified = final_cands[0]["bias"]
                        cd["bias_candidates"] = final_cands
                    next_state = "spark"
                else:
                    next_state = "listening"

            elif listening_answers >= 3:
                # All three answers in — run the diagnostic
                with st.spinner(""):
                    candidates = identify_candidate_biases(
                        conv_hist, enriched_profile_str, context
                    )
                if candidates:
                    top = candidates[0]
                    second = candidates[1] if len(candidates) > 1 else None
                    gap = top["score"] - second["score"] if second else 10

                    # Always store the full candidate list for uncertainty display
                    cd["bias_candidates"] = candidates

                    if top["score"] >= 7 and gap >= 3:
                        # Clear winner — spark directly
                        pre_identified = top["bias"]
                        next_state = "spark"
                        cd["disambiguation_question"] = ""
                    else:
                        # Two close candidates — ask one disambiguation question
                        if second:
                            with st.spinner(""):
                                disambig_q = get_disambiguation_question(
                                    top, second,
                                    conv_hist, enriched_profile_str,
                                    cd["decision"], context
                                ) or ""
                        else:
                            disambig_q = ""
                        cd["disambiguation_question"] = disambig_q
                        next_state = "listening"
                        new_extra = 1
                else:
                    # No candidates — spark anyway (get_spark_message handles it)
                    next_state = "spark"

            else:
                # Fewer than 3 answers — keep asking
                next_state = "listening"

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
                "loop_context": cd.get("loop_context", ""),
                "bias_text": cd.get("bias_text", ""),
                "explanation_text": cd.get("explanation_text", ""),
                "perspective_text": cd.get("perspective_text", ""),
                "perspective_option": cd.get("perspective_option", ""),
                "question_text": cd.get("question_text", ""),
                "conversation_message": cd.get("conversation_message", ""),
                # Inductive bias identification fields
                "pre_identified_bias": pre_identified,
                "disambiguation_question": cd.get("disambiguation_question", ""),
                "bias_candidates": cd.get("bias_candidates", []),
            }
            st.session_state.phase = "generating"
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STATE: SPARK
    # ══════════════════════════════════════════════════════════════════════════
    elif capcs_state == "spark":
        bias_name_short = cd.get("bias_text", "").split("—")[0].strip()[:60]
        if not conversation_msg or not cd.get("counterattack_message"):
            st.session_state.phase = "generating"; st.rerun()

        uk = user_key_corr
        counterattack_msg = cd.get("counterattack_message", "")
        confidence_start = cd.get("confidence_start", 35)
        ca_pending_conf = st.session_state.get("_ca_pending_conf", None)

        # ── Block 1: Possible pattern identified ─────────────────────────────
        with st.container(border=True):

            # Likelihood table at the top — before the spark message
            # Fall back to a single synthetic entry if diagnostic had no scored candidates
            _candidates = cd.get("bias_candidates", [])
            if not _candidates and bias_name_short:
                _candidates = [{"bias": bias_name_short, "score": 5, "evidence": ""}]
            if _candidates:
                _top_name = _candidates[0].get("bias", bias_name_short)
                st.markdown(f"The most likely pattern is **{_top_name}**")
                st.markdown("")
                for _c in _candidates:
                    # Absolute confidence: score is 1-10, multiply by 10 for %
                    # This shows each bias's own confidence independently,
                    # rather than normalising to 100 % across candidates.
                    _pct = max(0, min(100, int(_c.get("score", 0)) * 10))
                    _name = _c.get("bias", "")
                    _is_top = _name.split("—")[0].strip() == bias_name_short
                    _filled = min(10, _pct // 10)
                    _bar = "█" * _filled + "░" * (10 - _filled)
                    _style = "font-weight:600;" if _is_top else "color:#6b7280;"
                    st.markdown(
                        f"<div style='font-size:13px;{_style}margin-bottom:4px'>"
                        f"{_name} &nbsp; <code>{_bar}</code> {_pct}%<br>"
                        f"<span style='font-size:11px;font-weight:400;color:#9ca3af'>"
                        f"{_c.get('evidence','')}</span></div>",
                        unsafe_allow_html=True
                    )
                st.markdown("---")

            st.markdown("**What CASPER observed**")
            st.markdown(conversation_msg)
            st.markdown("---")
            if bias_name_short:
                st.markdown(f"What is **{bias_name_short}**?")
            st.markdown(cd.get("explanation_text", ""))

            st.markdown("")
            _current_resonance = cd.get("bias_resonance", "")
            if _current_resonance:
                _label_map = {"full": "✅ Resonates", "partial": "🔶 Partially", "none": "❌ Doesn't fit"}
                st.caption(f"Your response: {_label_map.get(_current_resonance, _current_resonance)}")
            else:
                kb = f"spark_{bias_name_short[:15].replace(' ','_')}_{round_num}"
                col1, col2, col3 = st.columns(3)
                with col1:
                    resonates = st.button("✅ Resonates", key=f"{kb}_yes", use_container_width=True)
                with col2:
                    partially = st.button("🔶 Partially", key=f"{kb}_partial", use_container_width=True)
                with col3:
                    doesnt_fit = st.button("❌ Doesn't fit", key=f"{kb}_no", use_container_width=True)

                if resonates:
                    save_bias_correction(uk, bias_name_short, "accurate", "")
                    new_cd = dict(cd)
                    new_cd["bias_resonance"] = "full"
                    new_cd["confirmed_bias"] = bias_name_short
                    st.session_state.current_decision = new_cd
                    st.rerun()
                if partially:
                    save_bias_correction(uk, bias_name_short, "partial", "")
                    new_cd = dict(cd)
                    new_cd["bias_resonance"] = "partial"
                    new_cd["confirmed_bias"] = bias_name_short
                    new_cd["counterattack_message"] = ""  # regenerate with resonance tone
                    st.session_state.current_decision = new_cd
                    st.rerun()
                if doesnt_fit:
                    save_bias_correction(uk, bias_name_short, "inaccurate", "")
                    rejected = list(cd.get("rejected_biases", []))
                    if bias_name_short and bias_name_short not in rejected:
                        rejected.append(bias_name_short)
                    new_cd = dict(cd)
                    new_cd["bias_resonance"] = "none"
                    new_cd["confirmed_bias"] = bias_name_short
                    new_cd["rejected_biases"] = rejected
                    new_cd["counterattack_message"] = ""  # regenerate with resonance tone
                    st.session_state.current_decision = new_cd
                    st.rerun()

        # ── Routing helpers ───────────────────────────────────────────────────
        def _go_conviction_from_spark(conf_val):
            perspective = cd.get("perspective_text", "")
            if perspective and perspective not in st.session_state.all_options:
                st.session_state.all_options.append(perspective)
            conv_hist = list(cd.get("conversation_history", []))
            conv_hist.append({"role": "user", "content": f"[confidence: {conf_val}%]"})
            new_round = cd.get("rounds", 0) + 1
            rl = list(cd.get("rounds_log", []))
            # Log spark round so history replay and Rounds tab can reconstruct the bias reveal
            rl.append({
                "round": new_round, "round_number": new_round,
                "round_state": "spark",
                "timestamp": datetime.now().isoformat(),
                "bias": cd.get("bias_text", ""), "explanation": cd.get("explanation_text", ""),
                "perspective": "", "question": "",
                "conversation_message": cd.get("conversation_message", ""),
                "followups": [], "answer": "",
                "answer_depth": "", "answer_emotion": "", "answer_certainty": "",
                "answer_key_signal": "", "shifted": False, "still_undecided": False,
                "how_shifted": "", "leaning": cd.get("leaning", ""),
                "confidence": cd.get("confidence_before", 50), "shift": 0, "confidence_shift": 0,
            })
            rl.append({
                "round": new_round, "round_number": new_round,
                "round_state": "counterattack",
                "timestamp": datetime.now().isoformat(),
                "bias": cd.get("bias_text", ""), "explanation": cd.get("explanation_text", ""),
                "perspective": perspective, "question": "",
                "conversation_message": cd.get("counterattack_message", ""),
                "followups": [], "answer": f"[confidence: {conf_val}%]",
                "answer_depth": "", "answer_emotion": "", "answer_certainty": "",
                "answer_key_signal": "", "shifted": True, "still_undecided": False,
                "how_shifted": "", "leaning": perspective or cd.get("leaning", ""),
                "confidence": conf_val,
                "shift": conf_val - cd.get("confidence_start", 50),
                "confidence_shift": conf_val - cd.get("confidence_start", 50),
            })
            new_cd = dict(cd)
            new_cd["rounds"] = new_round
            new_cd["rounds_log"] = rl
            new_cd["conversation_history"] = conv_hist
            new_cd["capcs_state"] = "conviction"
            st.session_state.current_decision = new_cd
            st.session_state["_final_choice"] = perspective or cd.get("leaning", "")
            st.session_state["_confidence_after"] = conf_val
            st.session_state["_ca_partial_mode"] = False
            st.session_state["_ca_partial_conf"] = None
            st.session_state["_ca_pending_conf"] = None
            st.session_state["_ca_explore_above"] = False
            st.session_state["_partial_probe_question"] = ""
            st.session_state["_conviction_suggestions"] = ""
            st.session_state["_listening_clarification"] = None
            st.rerun()

        def _go_rejected_from_spark(conf_val):
            bias_tried = cd.get("bias_text", "").split("—")[0].strip()[:60]
            perspective = cd.get("perspective_text", "")
            rejected_opts = list(cd.get("rejected_options", []))
            rejected_biases = list(cd.get("rejected_biases", []))
            if perspective and perspective not in rejected_opts:
                rejected_opts.append(perspective)
            conv_hist_r = list(cd.get("conversation_history", []))
            conv_hist_r.append({"role": "user", "content": f"[confidence: {conf_val}% — option rejected]"})
            new_round_r = cd.get("rounds", 0) + 1
            rl_r = list(cd.get("rounds_log", []))
            # Log spark round so history replay and Rounds tab can reconstruct the bias reveal
            rl_r.append({
                "round": new_round_r, "round_number": new_round_r,
                "round_state": "spark",
                "timestamp": datetime.now().isoformat(),
                "bias": cd.get("bias_text", ""), "explanation": cd.get("explanation_text", ""),
                "perspective": "", "question": "",
                "conversation_message": cd.get("conversation_message", ""),
                "followups": [], "answer": "",
                "answer_depth": "", "answer_emotion": "", "answer_certainty": "",
                "answer_key_signal": "", "shifted": False, "still_undecided": False,
                "how_shifted": "", "leaning": cd.get("leaning", ""),
                "confidence": cd.get("confidence_before", 50), "shift": 0, "confidence_shift": 0,
            })
            rl_r.append({
                "round": new_round_r, "round_number": new_round_r,
                "round_state": "counterattack_rejected",
                "timestamp": datetime.now().isoformat(),
                "bias": cd.get("bias_text", ""), "explanation": cd.get("explanation_text", ""),
                "perspective": perspective, "question": "",
                "conversation_message": cd.get("counterattack_message", ""),
                "followups": [], "answer": f"[confidence: {conf_val}%]",
                "answer_depth": "", "answer_emotion": "", "answer_certainty": "",
                "answer_key_signal": "", "shifted": False, "still_undecided": False,
                "how_shifted": "", "leaning": cd.get("leaning", ""),
                "confidence": conf_val, "shift": 0, "confidence_shift": 0,
            })
            loop_ctx = (
                f"Previously tried bias: '{bias_tried}'. "
                f"Option proposed: '{perspective}'. "
                f"User gave only {conf_val}% confidence — below their starting level. Option rejected. "
                f"Explore a DIFFERENT angle."
            )
            st.session_state["_ca_partial_mode"] = False
            st.session_state["_ca_partial_conf"] = None
            st.session_state["_ca_pending_conf"] = None
            st.session_state["_ca_explore_above"] = False
            st.session_state["_partial_probe_question"] = ""
            st.session_state["_listening_clarification"] = None
            st.session_state.current_decision = {
                "decision": cd["decision"], "decision_short": cd.get("decision_short", ""),
                "context": cd.get("context", ""), "options": cd["options"],
                "leaning": cd.get("leaning", ""), "is_undecided": cd.get("is_undecided", False),
                "confidence_before": conf_val, "confidence_start": cd["confidence_start"],
                "timestamp": cd["timestamp"], "rounds": new_round_r,
                "rounds_log": rl_r,
                "last_answer": f"[confidence: {conf_val}%]",
                "conversation_history": conv_hist_r,
                "capcs_state": "listening", "listening_answers": cd.get("listening_answers", 0),
                "extra_listening": 2, "rejected_biases": rejected_biases,
                "rejected_options": rejected_opts, "confirmed_bias": "",
                "loop_context": loop_ctx, "answer_signals": {},
                "counterattack_exchanges": [],
                "pre_identified_bias": "", "disambiguation_question": "", "bias_candidates": [],
            }
            st.session_state.phase = "generating"
            st.rerun()

        # ── Block 2: Option suggested + confidence slider ─────────────────────
        st.markdown("")
        with st.container(border=True):
            st.markdown("**Option suggested**")
            st.markdown(counterattack_msg)
            st.markdown("")

            if ca_pending_conf is None:
                # Phase 1 — slider
                conf_val = st.slider(
                    "How much does this feel like it could work for you?",
                    0, 100, confidence_start, 5,
                    key=f"spark_ca_conf_{round_num}"
                )
                badge(conf_val)
                st.markdown("")
                if st.button("→ Submit", key=f"spark_ca_submit_{round_num}",
                             type="primary", use_container_width=True):
                    if conf_val <= confidence_start:
                        _go_rejected_from_spark(conf_val)
                    else:
                        st.session_state["_ca_pending_conf"] = conf_val
                        st.rerun()
            else:
                # Phase 2 — threshold routing
                conf_val = ca_pending_conf
                explore_above = st.session_state.get("_ca_explore_above", False)
                perspective_option = cd.get("perspective_text", "")

                if st.session_state.get("_ca_partial_mode"):
                    # Generate and cache probe question inside container, display outside
                    probe_q = st.session_state.get("_partial_probe_question", "")
                    if not probe_q:
                        with st.spinner(""):
                            probe_q = get_partial_probe(
                                cd.get("perspective_text", ""),
                                cd.get("confirmed_bias", "") or cd.get("bias_text", ""),
                                cd.get("conversation_history", []),
                                enriched_profile_str,
                                above_threshold=explore_above
                            ) or "What's coming up for you when you imagine actually going with this?"
                        st.session_state["_partial_probe_question"] = probe_q

                elif conf_val >= CONFIDENCE_THRESHOLD:
                    st.markdown(
                        f"You're at **{conf_val}% confidence** about *{perspective_option}* — "
                        f"not full certainty yet, but enough to move forward if you're ready. "
                        f"Do you want to continue, or keep exploring?"
                    )
                    scroll_to_chat_bottom()
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("→ Continue", key=f"spark_ca_above_continue_{round_num}",
                                     type="primary", use_container_width=True):
                            _go_conviction_from_spark(conf_val)
                    with col2:
                        if st.button("Explore further", key=f"spark_ca_above_explore_{round_num}",
                                     use_container_width=True):
                            st.session_state["_ca_partial_mode"] = True
                            st.session_state["_ca_partial_conf"] = conf_val
                            st.session_state["_ca_explore_above"] = True
                            st.rerun()

                else:
                    st.markdown(
                        f"You're at **{conf_val}% confidence** with this option. "
                        f"Does that feel like enough to move forward, or would you like to explore further?"
                    )
                    scroll_to_chat_bottom()
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("I'm happy with this", key=f"spark_ca_partial_happy_{round_num}",
                                     type="primary", use_container_width=True):
                            _go_conviction_from_spark(conf_val)
                    with col2:
                        if st.button("Explore further", key=f"spark_ca_partial_explore_{round_num}",
                                     use_container_width=True):
                            st.session_state["_ca_partial_mode"] = True
                            st.session_state["_ca_partial_conf"] = conf_val
                            st.session_state["_ca_explore_above"] = False
                            st.rerun()

        # ── Probe question displayed outside the container so it isn't clipped ─
        if st.session_state.get("_ca_partial_mode") and ca_pending_conf is not None:
            probe_q = st.session_state.get("_partial_probe_question", "")
            explore_above = st.session_state.get("_ca_explore_above", False)
            if probe_q:
                with st.chat_message("assistant", avatar="🧑‍🏫"):
                    st.markdown(probe_q)
                # No scroll_to_chat_bottom here — it would scroll the probe
                # question behind the fixed chat_input at the bottom of the viewport
            partial_reply = st.chat_input(
                "What's coming up for you?",
                key=f"spark_ca_partial_{round_num}"
            )
            if partial_reply and partial_reply.strip():
                partial_answer = partial_reply.strip()
                conf_val_p = st.session_state.get("_ca_partial_conf", confidence_start)
                perspective = cd.get("perspective_text", "")
                conv_hist_p = list(cd.get("conversation_history", []))
                conv_hist_p.append({"role": "assistant", "content": probe_q})
                conv_hist_p.append({"role": "user", "content": partial_answer})
                new_round_p = cd.get("rounds", 0) + 1
                rl_p = list(cd.get("rounds_log", []))
                rl_p.append({
                    "round": new_round_p, "round_number": new_round_p,
                    "round_state": "counterattack_partial",
                    "timestamp": datetime.now().isoformat(),
                    "bias": cd.get("bias_text", ""), "explanation": cd.get("explanation_text", ""),
                    "perspective": perspective, "question": probe_q,
                    "conversation_message": probe_q,  # show probe Q (not counterattack) in replay
                    "followups": [], "answer": partial_answer,
                    "answer_depth": "", "answer_emotion": "", "answer_certainty": "",
                    "answer_key_signal": "", "shifted": False, "still_undecided": False,
                    "how_shifted": "", "leaning": cd.get("leaning", ""),
                    "confidence": conf_val_p, "shift": 0, "confidence_shift": 0,
                })
                if explore_above:
                    loop_ctx = (
                        f"Option proposed: '{perspective}' ({conf_val_p}% confidence — above clarity bar). "
                        f"User chose to explore further. They said: '{partial_answer}'. "
                        f"Ask 2 focused questions about what specifically is in the way of full commitment, "
                        f"then propose a refined or adapted version of the same option. "
                        f"Do NOT abandon the option entirely — refine it."
                    )
                else:
                    loop_ctx = (
                        f"Option proposed: '{perspective}' ({conf_val_p}% confidence). "
                        f"User said this didn't fully land: '{partial_answer}'. "
                        f"Ask 2 questions to gather new signal, then spark again with a refined or different bias."
                    )
                st.session_state["_ca_partial_mode"] = False
                st.session_state["_ca_partial_conf"] = None
                st.session_state["_ca_pending_conf"] = None
                st.session_state["_ca_explore_above"] = False
                st.session_state["_partial_probe_question"] = ""
                st.session_state.current_decision = {
                    "decision": cd["decision"], "decision_short": cd.get("decision_short", ""),
                    "context": cd.get("context", ""), "options": cd["options"],
                    "leaning": cd.get("leaning", ""), "is_undecided": cd.get("is_undecided", False),
                    "confidence_before": conf_val_p, "confidence_start": cd["confidence_start"],
                    "timestamp": cd["timestamp"], "rounds": new_round_p,
                    "rounds_log": rl_p, "last_answer": partial_answer,
                    "conversation_history": conv_hist_p,
                    "capcs_state": "listening", "listening_answers": cd.get("listening_answers", 0),
                    "extra_listening": 2, "rejected_biases": cd.get("rejected_biases", []),
                    "rejected_options": cd.get("rejected_options", []), "confirmed_bias": "",
                    "loop_context": loop_ctx, "answer_signals": {},
                    "counterattack_exchanges": [],
                    "pre_identified_bias": "", "disambiguation_question": "", "bias_candidates": [],
                }
                st.session_state.phase = "generating"
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STATE: CONVICTION
    # ══════════════════════════════════════════════════════════════════════════
    elif capcs_state == "conviction":
        _perspective = cd.get("perspective_text", "")
        if _perspective and _perspective not in st.session_state.all_options:
            st.session_state.all_options.append(_perspective)

        # final_choice is set programmatically from the counterattack slider
        final_choice = st.session_state.get("_final_choice", "") or _perspective
        what_shifted = st.session_state.get("_what_shifted", "")
        conf = st.session_state.get("_confidence_after", cd.get("confidence_start", 50))

        def _save_and_close(fc, ws, conf_val):
            shift = conf_val - cd.get("confidence_before", 50)
            rounds_log = cd.get("rounds_log", [])
            rounds_log.append({
                "round": round_num, "round_number": round_num,
                "round_state": "conviction",
                "timestamp": datetime.now().isoformat(),
                "bias": cd.get("bias_text", ""), "explanation": cd.get("explanation_text", ""),
                "perspective": cd.get("perspective_text", ""), "question": "",
                "conversation_message": "",
                "followups": [], "answer": ws,
                "answer_depth": "", "answer_emotion": "", "answer_certainty": "",
                "answer_key_signal": "", "shifted": True, "still_undecided": False,
                "how_shifted": ws, "leaning": fc or cd.get("leaning", ""),
                "confidence": conf_val, "shift": shift, "confidence_shift": shift,
            })
            entry = {
                "user_key": st.session_state.get("user_key", ""),
                "decision": cd.get("decision_short", ""),
                "context": cd.get("context", ""),
                "options": cd.get("options", ""),
                "all_options": st.session_state.all_options,
                "final_choice": fc or cd.get("leaning", ""),
                "confidence_start": cd.get("confidence_start", conf_val),
                "confidence_final": conf_val,
                "confidence_shift": conf_val - cd.get("confidence_start", conf_val),
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "confidence_trajectory": [cd.get("confidence_start", conf_val), conf_val],
                "rounds_completed": sum(1 for r in rounds_log if r.get("round_state") == "spark") or 1,
                "rounds_log": rounds_log,
                "conversation_history": cd.get("conversation_history", []),
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
            st.session_state["_ca_partial_mode"] = False
            st.session_state["_ca_partial_conf"] = None
            st.session_state["_ca_pending_conf"] = None
            st.session_state["_ca_explore_above"] = False
            st.session_state["_partial_probe_question"] = ""
            st.session_state["_conviction_suggestions"] = ""
            st.session_state["_listening_clarification"] = None
            st.session_state.phase = "feedback"
            st.rerun()

        # ── Step 1 — Acknowledgement + "What shifted?" ───────────────────────
        if not what_shifted:
            with st.chat_message("assistant", avatar="🧑‍🏫"):
                st.markdown(f"You've landed on **{final_choice}**.")
                st.markdown("What shifted for you?")
            scroll_to_chat_bottom()
            what_shifted_input = st.chat_input(
                "What shifted for you?",
                key=f"conviction_ws_{round_num}"
            )
            if what_shifted_input and what_shifted_input.strip():
                ws = what_shifted_input.strip()
                conv_hist = cd.get("conversation_history", [])
                conv_hist.append({"role": "assistant", "content": f"You've landed on {final_choice}. What shifted for you?"})
                conv_hist.append({"role": "user", "content": ws})
                new_cd = dict(cd)
                new_cd["conversation_history"] = conv_hist
                st.session_state.current_decision = new_cd
                st.session_state["_what_shifted"] = ws
                st.rerun()

        # ── Step 2 — Personalised suggestions + Complete session ─────────────
        else:
            with st.chat_message("assistant", avatar="🧑‍🏫"):
                st.markdown(f"You've landed on **{final_choice}**.")
                st.markdown("What shifted for you?")
            with st.chat_message("user", avatar="👤"):
                st.markdown(what_shifted)
            scroll_to_chat_bottom()

            suggestions = st.session_state.get("_conviction_suggestions", "")
            if not suggestions:
                with st.spinner(""):
                    suggestions = get_personalised_suggestions(
                        final_choice,
                        cd.get("confirmed_bias", "") or cd.get("bias_text", ""),
                        enriched_profile_str,
                        what_shifted
                    )
                st.session_state["_conviction_suggestions"] = suggestions

            if suggestions:
                with st.chat_message("assistant", avatar="🧑‍🏫"):
                    st.markdown(suggestions)

            scroll_to_chat_bottom()
            st.markdown("")
            if st.button("✓ Complete session", key=f"conviction_complete_{round_num}",
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
            "Your answers help validate whether CASPER actually helps people think more clearly.",
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
        bias_cycles = len(biases_found) or last.get("rounds_completed", 1)
        st.markdown(f"**Rounds:** {bias_cycles} bias {'cycle' if bias_cycles == 1 else 'cycles'} explored")
        if biases_found or options_proposed:
            bias_str = " and ".join(biases_found) if biases_found else "a cognitive pattern"
            opt_str  = " and ".join(f'"{o}"' for o in options_proposed if o) if options_proposed else "an alternative path"
            summary_txt = (
                f"CASPER identified <b>{bias_str}</b> as the main pattern shaping your thinking. "
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
        else:
            # Group raw round entries into bias cycles.
            # A cycle = listening Q&A → spark (bias named) → counterattack (perspective).
            # spark_rejected and counterattack_rejected are noted within the cycle.
            cycles = []
            pending_listening = []
            for r in rounds_log_display:
                state = r.get("round_state", "")
                if state == "listening":
                    pending_listening.append(r)
                elif state in ("spark", "spark_rejected"):
                    cycles.append({
                        "listening": pending_listening,
                        "spark": r,
                        "counterattack": None,
                    })
                    pending_listening = []
                elif state in ("counterattack", "counterattack_rejected"):
                    if cycles:
                        cycles[-1]["counterattack"] = r
                # conviction rounds are intentionally skipped — captured in Summary tab

            for i, cycle in enumerate(cycles, 1):
                spark_r = cycle["spark"]
                ca_r    = cycle["counterattack"]
                bias_name = (spark_r.get("bias", "") or "").split("—")[0].strip()[:60]
                accepted  = ca_r and ca_r.get("round_state") == "counterattack"
                rejected_spark = spark_r.get("round_state") == "spark_rejected"
                rejected_ca    = ca_r and ca_r.get("round_state") == "counterattack_rejected"

                if accepted:
                    status = "✅ Accepted"
                elif rejected_ca:
                    status = "↩️ Rejected option"
                elif rejected_spark:
                    status = "↩️ Rejected bias"
                else:
                    status = "➡️"

                label = f"Bias cycle {i} — {bias_name} ({status})" if bias_name else f"Cycle {i} ({status})"

                with st.expander(label):
                    # ── Conversation (listening Q&A) ──────────────────────────
                    if cycle["listening"]:
                        st.markdown("**Conversation**")
                        for lr in cycle["listening"]:
                            msg = lr.get("conversation_message") or lr.get("question", "")
                            if msg:
                                st.markdown(f"> 🧑‍🏫 {msg}")
                            if lr.get("answer"):
                                st.markdown(f"> 👤 {lr['answer']}")
                            for fq in lr.get("followups", []):
                                if fq.get("question"):
                                    st.markdown(f"> 🧑‍🏫 ↩️ *{fq['question']}*")
                                if fq.get("answer"):
                                    st.markdown(f"> 👤 {fq['answer']}")
                        st.divider()

                    # ── Spark (bias reveal) ───────────────────────────────────
                    spark_msg = spark_r.get("conversation_message", "")
                    if spark_msg:
                        st.markdown(f"🧑‍🏫 {spark_msg}")
                    if bias_name:
                        st.markdown(f"⚠️ **Bias identified:** {bias_name}")
                    if spark_r.get("explanation"):
                        st.caption(spark_r["explanation"])
                    if rejected_spark:
                        st.caption("*You said this didn't resonate — CASPER tried a different angle.*")

                    # ── Counterattack (perspective proposed) ──────────────────
                    if ca_r:
                        st.divider()
                        ca_msg = ca_r.get("conversation_message") or ca_r.get("question", "")
                        if ca_msg:
                            st.markdown(f"🧑‍🏫 {ca_msg}")
                        if ca_r.get("perspective"):
                            st.markdown(f"💡 **Option proposed:** {ca_r['perspective']}")
                        if accepted:
                            st.caption("✅ You accepted this perspective.")
                        elif rejected_ca:
                            if ca_r.get("answer"):
                                st.markdown(f"👤 **Why it didn't work:** {ca_r['answer']}")
                            st.caption("↩️ CASPER tried a different approach.")

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

    if not completed:
        box("Complete your first session to see your history here.", style="info")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Sessions", "🎯 Calibration", "🧠 Biases", "🔄 Patterns", "👤 Profile"])

        with tab1:
            # ── Session History ───────────────────────────────────────────────
            st.markdown("### 📋 Session History")
            sessions_sorted = sorted(completed, key=lambda h: h.get("completed_at", ""), reverse=True)
            for idx, h in enumerate(sessions_sorted):
                date_str = h.get("completed_at", "")[:10] if h.get("completed_at") else "—"
                decision = h.get("decision", "")
                conf_start = h.get("confidence_start", "?")
                conf_final = h.get("confidence_final", "?")
                conf_shift = h.get("confidence_shift", 0) or 0
                shift_sign = "+" if conf_shift >= 0 else ""
                final_choice = h.get("final_choice", "—") or "—"
                domain = h.get("domain", "")
                rounds_log_h = h.get("rounds_log", [])

                # Collect spark rounds (rounds where a bias was named)
                spark_rounds_data = [
                    r for r in rounds_log_h
                    if r.get("bias") and (
                        r.get("round_state") == "spark"
                        or (not r.get("round_state") and r.get("bias"))
                    )
                ]
                # Deduplicate by bias name
                seen_biases = set()
                unique_spark_rounds = []
                for r in spark_rounds_data:
                    bname = r.get("bias", "").split("—")[0].strip()
                    if bname and bname not in seen_biases:
                        seen_biases.add(bname)
                        unique_spark_rounds.append(r)

                # Perspective lives on the counterattack round, not the spark round.
                # Build bias_name → perspective lookup from counterattack rounds.
                perspective_by_bias = {
                    r.get("bias", "").split("—")[0].strip().lower(): r.get("perspective", "")
                    for r in rounds_log_h
                    if r.get("round_state") in ("counterattack", "counterattack_rejected")
                    and r.get("perspective")
                }
                accepted_by_bias = {
                    r.get("bias", "").split("—")[0].strip().lower(): r.get("round_state") == "counterattack"
                    for r in rounds_log_h
                    if r.get("round_state") in ("counterattack", "counterattack_rejected")
                }

                shift_icon = "📈" if conf_shift > 0 else ("📉" if conf_shift < 0 else "➡️")
                dec_preview = decision[:65] + "…" if len(decision) > 65 else decision

                with st.expander(f"{shift_icon} {date_str}  —  {dec_preview}", expanded=(idx == 0)):

                    # ── Top metrics ───────────────────────────────────────────
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown("**Clarity**")
                        st.markdown(f"{conf_start}% → {conf_final}% ({shift_sign}{conf_shift}%)")
                    with col_b:
                        st.markdown("**Chose**")
                        st.markdown(final_choice)
                    with col_c:
                        st.markdown("**Domain**")
                        st.markdown(domain.capitalize() if domain else "—")

                    if len(decision) > 65:
                        st.markdown("")
                        st.caption(decision)

                    # ── Bias cycles ───────────────────────────────────────────
                    if unique_spark_rounds:
                        st.markdown("")
                        st.markdown("**What CASPER noticed**")
                        for r in unique_spark_rounds:
                            bias_name = r.get("bias", "").split("—")[0].strip()
                            explanation = r.get("explanation", "")
                            perspective = perspective_by_bias.get(bias_name.lower(), "") or r.get("perspective", "")
                            accepted = accepted_by_bias.get(bias_name.lower(), r.get("shifted", False))
                            resonance = "✅ Resonated" if accepted else "↩️ Didn't land"

                            st.markdown(
                                f"<div style='border-left:3px solid #7c3aed;padding:8px 12px;"
                                f"margin:6px 0;border-radius:0 6px 6px 0;background:#faf5ff'>"
                                f"<div style='font-weight:600;font-size:14px;margin-bottom:4px'>"
                                f"🔍 {bias_name} &nbsp; <span style='font-size:12px;"
                                f"font-weight:400;color:#6b7280'>{resonance}</span></div>"
                                + (f"<div style='font-size:13px;color:#374151;margin-bottom:4px'>"
                                   f"{explanation}</div>" if explanation else "")
                                + (f"<div style='font-size:12px;color:#6b7280'>"
                                   f"💡 Alternative explored: <em>{perspective}</em></div>"
                                   if perspective else "")
                                + "</div>",
                                unsafe_allow_html=True
                            )

                    # ── Decision insight ─────────────────────────────────────
                    if final_choice and final_choice != "—":
                        st.markdown("")
                        st.markdown("**Your decision**")

                        # Find the initial leaning (first non-empty leaning in listening rounds)
                        initial_leaning = ""
                        for r in rounds_log_h:
                            if r.get("leaning") and r.get("round_state") in ("listening", "", None):
                                initial_leaning = r["leaning"].strip()
                                break

                        choice_changed = bool(
                            initial_leaning
                            and initial_leaning.lower() != final_choice.lower()
                        )
                        gained_clarity = conf_shift > 0

                        # One-liner interpretation
                        if choice_changed and gained_clarity:
                            verdict = "↔️ You moved away from your initial leaning and gained clarity — the challenge shifted your direction."
                            verdict_color = "#065f46"
                            verdict_bg = "#d1fae5"
                        elif choice_changed and not gained_clarity:
                            verdict = "↔️ Your choice changed, but full clarity didn't land — worth sitting with this one."
                            verdict_color = "#92400e"
                            verdict_bg = "#fef3c7"
                        elif not choice_changed and gained_clarity:
                            verdict = "✓ You confirmed your original direction with greater clarity — the challenge strengthened your reasoning rather than changing it."
                            verdict_color = "#1e40af"
                            verdict_bg = "#dbeafe"
                        else:
                            verdict = "→ You held your original direction — the challenge didn't move your thinking this time."
                            verdict_color = "#374151"
                            verdict_bg = "#f3f4f6"

                        # Show initial leaning if it changed
                        leaning_line = ""
                        if choice_changed and initial_leaning:
                            leaning_line = (
                                f"<div style='font-size:12px;color:#6b7280;margin-bottom:4px'>"
                                f"Started leaning: <em>{initial_leaning}</em></div>"
                            )

                        # Main bias for this session (first spark round)
                        main_bias = (unique_spark_rounds[0].get("bias","").split("—")[0].strip()
                                     if unique_spark_rounds else "")
                        bias_line = ""
                        if main_bias:
                            bias_line = (
                                f"<div style='font-size:12px;color:#6b7280;margin-bottom:6px'>"
                                f"Bias at play: <em>{main_bias}</em></div>"
                            )

                        st.markdown(
                            f"<div style='border-radius:8px;padding:10px 14px;"
                            f"background:{verdict_bg};margin:4px 0'>"
                            f"<div style='font-weight:600;font-size:15px;margin-bottom:6px'>"
                            f"{final_choice}</div>"
                            + leaning_line
                            + bias_line
                            + f"<div style='font-size:13px;color:{verdict_color}'>{verdict}</div>"
                            + "</div>",
                            unsafe_allow_html=True
                        )

                    # ── Conversation replay ───────────────────────────────────
                    conv = h.get("conversation_history", [])
                    if conv:
                        st.markdown("")
                        with st.expander("💬 View conversation", expanded=False):
                            for msg in conv:
                                role = msg.get("role", "")
                                content = msg.get("content", "")
                                if role == "assistant":
                                    st.markdown(f"🧑‍🏫 **CASPER:** {content}")
                                elif role == "user":
                                    st.markdown(f"👤 **You:** {content}")
                                st.markdown("")

        with tab2:
            # ── Section 1: Calibration Card ───────────────────────────────────────
            st.markdown("### 🎯 Confidence Calibration")
            if len(completed) < 2:
                box("Complete more sessions to see trends across time.", style="info")

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

        with tab3:
            # ── Bias Profile ─────────────────────────────────────────────────
            st.markdown("### 🧠 Bias Profile")
            if len(completed) < 2:
                box("Complete more sessions to see patterns across time.", style="info")
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
                    "Help CASPER learn about you — tell us whether each detected bias felt accurate."
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
                        _analysis_key = f"_bias_analysis_{bias_name[:30]}_{count}"
                        if _analysis_key not in st.session_state:
                            with st.spinner(f"Analysing {bias_name}..."):
                                st.session_state[_analysis_key] = get_bias_analysis(
                                    bias_name, count, profile,
                                    "; ".join(set(data["domains"]))
                                )
                        box(st.session_state[_analysis_key], style="insight")

                        # ── Co-adaptive feedback UI ────────────────────────────────
                        st.markdown("---")
                        st.markdown("**Was this bias detection accurate for you?**")
                        st.caption("Your answer directly updates how CASPER challenges you in future sessions.")

                        col1, col2, col3 = st.columns(3)
                        key_base = f"correction_{bias_name.replace(' ','_')[:20]}"

                        with col1:
                            if st.button("✅ Yes, accurate", key=f"{key_base}_yes", use_container_width=True):
                                save_bias_correction(user_key, bias_name, "accurate", "")
                                st.success("Noted — CASPER will keep this in mind as a genuine pattern.")
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
                                st.success("Correction saved — CASPER will apply more caution with this bias in future sessions.")
                                st.rerun()

                        # Show existing note if there is one
                        if existing.get("note"):
                            st.caption(f"Your note: *\"{existing['note']}\"*")
            else:
                box(
                    "No bias data yet. Complete a session and return here — "
                    "bias patterns will appear after your first full cycle.",
                    style="info"
                )

        with tab4:
            # ── Reasoning Patterns ───────────────────────────────────────────
            st.markdown("### 🔄 Reasoning Patterns")
            if len(completed) < 2:
                box("Complete more sessions to see your reasoning patterns.", style="info")

            # Shift rate = perspectives accepted / perspectives proposed
            # (counterattack accepted vs total counterattack rounds)
            total_ca = sum(
                1 for h in completed for r in h.get("rounds_log", [])
                if r.get("round_state") in ("counterattack", "counterattack_rejected")
            )
            total_shifts = sum(
                1 for h in completed for r in h.get("rounds_log", [])
                if r.get("round_state") == "counterattack"
            )
            shift_rate = int(100 * total_shifts / max(total_ca, 1))

            # Average round number at which a perspective was first accepted
            first_shifts = []
            for h in completed:
                for r in h.get("rounds_log", []):
                    if r.get("round_state") == "counterattack":
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
                    box("⏱ Your sessions are getting longer — you may be bringing more complex decisions to CASPER.", style="insight")

        with tab5:
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
                st.caption("Profile last updated: first setup. Updating after situation changes helps CASPER challenge you more accurately.")
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
