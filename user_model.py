# CAPCS — user profile logic, onboarding questions, and longitudinal context

from collections import Counter


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

            # Shift rate: accepted counterattacks / total counterattack rounds
            total_ca = sum(
                1 for h in completed for r in h.get("rounds_log", [])
                if r.get("round_state") in ("counterattack", "counterattack_rejected")
            )
            total_shifts = sum(
                1 for h in completed for r in h.get("rounds_log", [])
                if r.get("round_state") == "counterattack"
            )
            shift_rate = total_shifts / max(total_ca, 1)
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

    # Shift rate: accepted counterattacks / total counterattack rounds
    total_ca = sum(
        1 for h in completed for r in h.get("rounds_log", [])
        if r.get("round_state") in ("counterattack", "counterattack_rejected")
    )
    total_shifts = sum(
        1 for h in completed for r in h.get("rounds_log", [])
        if r.get("round_state") == "counterattack"
    )
    if total_ca > 0:
        shift_rate = int(100 * total_shifts / total_ca)
        observed["observed_shift_rate"] = f"{shift_rate}% of perspectives accepted"

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

    # Avg round number at which a perspective was first accepted
    first_shifts = []
    for h in completed:
        for r in h.get("rounds_log", []):
            if r.get("round_state") == "counterattack":
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

    # Shift pattern: accepted counterattacks / total counterattack rounds
    total_ca = sum(
        1 for h in completed for r in h.get("rounds_log", [])
        if r.get("round_state") in ("counterattack", "counterattack_rejected")
    )
    total_shifts = sum(
        1 for h in completed for r in h.get("rounds_log", [])
        if r.get("round_state") == "counterattack"
    )
    shift_rate = int(100 * total_shifts / max(total_ca, 1))
    lines.append(f"Perspective acceptance rate: {shift_rate}% of proposed alternatives accepted")
    if shift_rate < 20:
        lines.append("Behaviour note: this user rarely accepts alternatives — use stronger, more grounded challenges.")
    elif shift_rate > 70:
        lines.append("Behaviour note: this user accepts alternatives readily — ensure perspectives are well-founded.")

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

    # Round at which perspective was first accepted
    first_shifts = []
    for h in completed:
        for r in h.get("rounds_log", []):
            if r.get("round_state") == "counterattack":
                rn = r.get("round_number") or r.get("round") or 1
                first_shifts.append(rn)
                break
    if first_shifts:
        avg_rts = sum(first_shifts) / len(first_shifts)
        lines.append(f"Typically accepts a perspective around round {avg_rts:.1f}")

    # Perspective diversity tracking — collect all past perspectives so they are never repeated.
    # Perspectives are stored as plain strings in the new system (not "OPTION: X" format).
    past_perspectives = []
    for h in completed:
        for r in h.get("rounds_log", []):
            p = (r.get("perspective") or "").strip()
            if not p:
                continue
            # Handle legacy "OPTION: X\nWHY: Y" format if present
            if "OPTION:" in p:
                for line in p.split("\n"):
                    if line.strip().startswith("OPTION:"):
                        opt = line.replace("OPTION:", "").strip()[:80]
                        if opt and opt not in past_perspectives:
                            past_perspectives.append(opt)
            else:
                opt = p[:80]
                if opt not in past_perspectives:
                    past_perspectives.append(opt)
    if past_perspectives:
        lines.append(
            f"PERSPECTIVES ALREADY OFFERED IN PAST SESSIONS (never repeat these, "
            f"generate genuinely new ones): {' | '.join(past_perspectives[:10])}"
        )

    lines.append("=== END OF USER HISTORY ===")
    return "\n".join(lines)


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
