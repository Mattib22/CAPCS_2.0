# CASPER — all Supabase database operations

import streamlit as st
import hashlib
import datetime as _dt
from datetime import datetime
import json
import os
from supabase import create_client
from config import PROFILE_VERSION


def make_user_key(username: str, pin: str) -> str:
    """
    Generate a deterministic anonymous ID from username + PIN.
    sha256(username:pin) — same combo always produces the same key.
    Nothing identifiable is stored — only the hash.
    """
    raw = f"{username.strip().lower()}:{pin.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


@st.cache_resource
def get_supabase():
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

def save_profile(profile_data: dict) -> bool:
    user_key = st.session_state.get("user_key", "")
    if not user_key or user_key == "anonymous":
        st.error("⚠️ Could not save your profile — no valid account found. Please log in again.")
        return False
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
        sb.table("profiles").upsert(row, on_conflict="user_key").execute()
        try:
            sb.table("profile_history").insert({**row, "saved_at": datetime.now().isoformat()}).execute()
        except Exception:
            pass  # profile_history failure is non-critical
        return True
    except Exception as e:
        st.error(f"⚠️ Could not save your profile: {e}. Please check your connection and try again.")
        return False

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


def delete_profile(user_key: str):
    try:
        sb = get_supabase()
        sb.table("profiles").delete().eq("user_key", user_key).execute()
        sb.table("profile_history").delete().eq("user_key", user_key).execute()
    except Exception:
        pass


# ── SESSION LOG FUNCTIONS ──────────────────────────────────────────────────────
def _collapse_rounds_to_product_rounds(rounds_log: list) -> list:
    """
    Convert the granular in-memory rounds_log (one entry per Q/A/state) into
    one row per product round (full bias cycle: listening → spark → counterattack).
    Conviction rows are skipped — that data lives on the session row.
    """
    product_rounds = []
    listening_buffer = []

    for r in rounds_log:
        state = r.get("round_state", "")

        if state == "listening":
            q = r.get("question", "") or r.get("conversation_message", "")
            a = r.get("answer", "")
            listening_buffer.append({
                "question": q,
                "answer": a,
                "leaning": r.get("leaning", ""),
            })

        elif state == "spark":
            product_rounds.append({
                "round_number": len(product_rounds) + 1,
                "bias": r.get("bias", ""),
                "explanation": r.get("explanation", ""),
                "perspective": "",
                "accepted": False,
                "confidence_before": r.get("confidence"),
                "confidence_after": None,
                "confidence_shift": 0,
                "initial_leaning": (
                    listening_buffer[0].get("leaning", "") if listening_buffer
                    else r.get("leaning", "")
                ),
                "listening_qa": listening_buffer[:],
            })
            listening_buffer = []

        elif state in ("counterattack", "counterattack_rejected", "counterattack_partial"):
            if product_rounds:
                pr = product_rounds[-1]
                if r.get("perspective"):
                    pr["perspective"] = r["perspective"]
                pr["accepted"] = (state == "counterattack")
                pr["confidence_after"] = r.get("confidence")
                pr["confidence_shift"] = r.get("shift", 0) or r.get("confidence_shift", 0)

    return product_rounds


def _product_rounds_to_rounds_log(rows: list) -> list:
    """
    Reconstruct a rounds_log compatible with the rest of the app from the
    simplified product-round rows stored in Supabase.
    Each DB row → one synthetic spark entry + one counterattack/rejected entry
    + one listening entry (for initial-leaning detection).
    """
    rounds_log = []
    for r in rows:
        n = r.get("round_number", 0)
        # Synthetic listening row — only what the app actually reads from it
        if r.get("initial_leaning"):
            rounds_log.append({
                "round_number": n,
                "round_state": "listening",
                "leaning": r.get("initial_leaning", ""),
                "bias": "", "explanation": "", "perspective": "",
                "shifted": False, "shift": 0,
                "confidence": r.get("confidence_before"),
            })
        # Spark row
        rounds_log.append({
            "round_number": n,
            "round_state": "spark",
            "bias": r.get("bias", ""),
            "explanation": r.get("explanation", ""),
            "perspective": "",
            "conversation_message": "",
            "shifted": False, "shift": 0,
            "confidence": r.get("confidence_before"),
        })
        # Counterattack row
        ca_state = "counterattack" if r.get("accepted") else "counterattack_rejected"
        rounds_log.append({
            "round_number": n,
            "round_state": ca_state,
            "bias": r.get("bias", ""),
            "explanation": r.get("explanation", ""),
            "perspective": r.get("perspective", ""),
            "shifted": bool(r.get("accepted")),
            "shift": r.get("confidence_shift", 0),
            "confidence_shift": r.get("confidence_shift", 0),
            "confidence": r.get("confidence_after"),
        })
    return rounds_log


def save_log(entry: dict):
    user_key = st.session_state.get("user_key", "")
    if not user_key or user_key == "anonymous":
        st.error("⚠️ Could not save your session — no valid account found. Please log in again.")
        return
    try:
        sb = get_supabase()
        session_duration = None
        try:
            start = datetime.fromisoformat(entry.get("timestamp", ""))
            end = datetime.fromisoformat(entry.get("completed_at", ""))
            session_duration = int((end - start).total_seconds())
        except Exception:
            pass

        # Extract what_shifted from conviction round (skipped in product rounds)
        what_shifted = next(
            (r.get("answer", "") for r in entry.get("rounds_log", [])
             if r.get("round_state") == "conviction"
             and r.get("answer", "") and not r.get("answer", "").startswith("[")),
            ""
        )

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
            "what_shifted": what_shifted,
            "rounds_completed": entry.get("rounds_completed"),
            "session_duration_seconds": session_duration,
            "conversation_history": entry.get("conversation_history", []),
            "undecided_outcome": entry.get("undecided_outcome", False),
        }
        try:
            session_res = sb.table("sessions").insert(session_row).execute()
        except Exception:
            # Retry without what_shifted in case the column doesn't exist yet
            session_row.pop("what_shifted", None)
            session_res = sb.table("sessions").insert(session_row).execute()
        session_id = session_res.data[0]["id"] if session_res.data else None
        entry["id"] = session_id

        # One row per product round (bias cycle), not per Q/A step
        if session_id:
            product_rounds = _collapse_rounds_to_product_rounds(
                entry.get("rounds_log", [])
            )
            for pr in product_rounds:
                sb.table("rounds").insert({
                    "session_id": session_id,
                    "user_key": user_key,
                    "round_number": pr["round_number"],
                    "bias": pr["bias"],
                    "explanation": pr["explanation"],
                    "perspective": pr["perspective"],
                    "accepted": pr["accepted"],
                    "confidence_before": pr["confidence_before"],
                    "confidence_after": pr["confidence_after"],
                    "confidence_shift": pr["confidence_shift"],
                    "initial_leaning": pr["initial_leaning"],
                    "listening_qa": pr["listening_qa"],
                }).execute()

        st.session_state["_load_log_dirty"] = True

    except Exception as e:
        st.error(f"Could not save session: {e}")


def load_log() -> list:
    """
    Load session history from Supabase. Cached per Streamlit run to avoid
    multiple round-trips per page render.
    """
    user_key = st.session_state.get("user_key", "")
    if not user_key or user_key == "anonymous":
        return []
    cache_key = f"_load_log_cache_{user_key}"
    if cache_key in st.session_state and st.session_state.get("_load_log_dirty") is not True:
        return st.session_state[cache_key]
    try:
        sb = get_supabase()
        sessions_res = (
            sb.table("sessions").select("*")
            .eq("user_key", user_key).order("completed_at").execute()
        )
        sessions = sessions_res.data or []

        rounds_res = (
            sb.table("rounds").select("*")
            .eq("user_key", user_key).order("round_number").execute()
        )
        rounds_by_session: dict = {}
        for r in (rounds_res.data or []):
            rounds_by_session.setdefault(r.get("session_id"), []).append(r)

        for s in sessions:
            product_rows = rounds_by_session.get(s.get("id"), [])
            s["rounds_log"] = _product_rounds_to_rounds_log(product_rows)

        st.session_state[cache_key] = sessions
        st.session_state["_load_log_dirty"] = False
        return sessions
    except Exception:
        return []

def delete_log(user_key: str):
    try:
        sb = get_supabase()
        sb.table("sessions").delete().eq("user_key", user_key).execute()
        sb.table("rounds").delete().eq("user_key", user_key).execute()
        sb.table("session_feedback").delete().eq("user_key", user_key).execute()
    except Exception:
        pass


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
    """Load all user corrections keyed by bias name. Ordered oldest-first so the
    latest correction always wins when the same bias has been rated multiple times."""
    try:
        sb = get_supabase()
        res = (
            sb.table("bias_corrections")
            .select("*")
            .eq("user_key", user_key)
            .order("created_at")
            .execute()
        )
        corrections = {}
        for row in (res.data or []):
            corrections[row["bias_name"]] = {
                "verdict": row["verdict"],
                "note": row["note"]
            }
        return corrections
    except Exception:
        return {}
