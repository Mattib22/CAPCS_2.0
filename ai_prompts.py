# CAPCS — all Gemini AI calls and prompt functions (v2)

import streamlit as st
import os
import google.generativeai as genai
import json
import re


def _get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


def ask_ai(prompt, max_tokens=1200):
    try:
        genai.configure(api_key=_get_api_key())
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
    return ask_ai(prompt, 1500)


def get_probing_question(decision, options, leaning, confidence, profile_str,
                          history, last_answer, context, longitudinal="", turn_num=2):
    """
    Turns 2 through PROBE_TURNS: pure Socratic probing — understand the user's thinking.
    No bias named. Forms hypothesis internally but says nothing.
    """
    long_section = f"\n{longitudinal}" if longitudinal else ""
    prompt = f"""You are a thinking partner on turn {turn_num} of a conversation. You have been listening carefully.

Your job this turn: ask ONE focused question that goes deeper into what the person just said.

Rules:
- Build DIRECTLY on their last answer — use their exact words or phrases to show you heard them
- One question only, max 35 words, ends with ?
- Cannot be answered yes/no
- Do NOT name any bias, pattern, or psychological concept
- Do NOT offer any new option or perspective yet — that comes later
- Be specific to what they actually said, not a generic follow-up
- Warm, direct, curious tone — like a thoughtful friend

CONVERSATION HISTORY:
{history}

USER'S LAST ANSWER: {last_answer}

DECISION: {decision}
OPTIONS: {options}
LEANING: {leaning or 'not specified'}
CONTEXT: {context}
PROFILE:
{profile_str}{long_section}

Output only the question, nothing else."""
    return ask_ai(prompt, 1500)


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

    prompt = f"""You are a thinking partner helping someone work through a real decision. You have been listening for {turn_num} turns.

{tone_note}
{undecided_note}
{drop_note}

Write ONE conversational message of MAXIMUM 80 WORDS. Count your words before outputting. If over 80, cut ruthlessly. Do not pad. Do not explain yourself.

The message does exactly three things:

1. REFLECT (1 sentence): Use the user's exact words from their last answer. Prove you heard something specific.

2. NAME (exactly 2 sentences, use this exact structure):
   Sentence 1: "That feeling of [what they described specifically, using their words], despite [the counter-desire or tension they also mentioned], is a classic case of [Bias Name]."
   Sentence 2: "You're prioritizing [what the bias causes them to prioritize] over [what they're ignoring or suppressing as a result]."
   The bias name must feel like a revelation — precise and surprising, not generic.

3. COUNTERATTACK (2 sentences): Introduce ONE concrete option as the direct antidote to the bias. Ground it in the user's specific constraints (money, location, relationship, visa, time — whatever they mentioned). Warm structure: "One way to counteract this is [Option] — [one sentence why this is the antidote, specific to their context]." Do NOT say "the counterforce is X" — that sounds mechanical. Then end with ONE question that asks the user to EVALUATE the option, not plan within it.
   WRONG question: "How could X fit your travel plans?" (assumes acceptance)
   RIGHT question: "Does X feel like something that could actually work for you, or does something about it feel off?" (asks for evaluation)
   The user has not agreed to the option yet. Do not assume they have.

Tone: warm, direct. Like a thoughtful friend who knows cognitive science. Never say "I notice" or "it seems like." Final sentence must end with a question mark.

---EXTRACT---
BIAS: [bias name only — max 6 words]
EXPLANATION: [what this bias is and why it appeared — max 40 words]
PERSPECTIVE: [the counterattacking option — max 8 words]
QUESTION: [exact question from above — copy it]
---END---

FULL CONVERSATION (use ALL of this for bias detection — not just the last message):
{history}

USER'S LAST ANSWER: {last_answer}

IMPORTANT: Detect the bias from patterns across the FULL CONVERSATION above. Reference specific things the user said in earlier turns, not just their last message. The final question must ask the user to evaluate the new option — not plan within it. Bad: "What skills would you need?" Good: "Does this feel like something that could actually work for you, or does something feel off?"

DECISION: {decision}
OPTIONS CONSIDERED SO FAR: {options}
LEANING: {leaning or 'genuinely undecided'}
CONFIDENCE: {confidence}%
CONTEXT: {context}
PROFILE:
{profile_str}{long_section}"""
    return ask_ai(prompt, 4096)


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
    return ask_ai(prompt, 3000)

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
    return ask_ai(prompt, 2000)

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
    return ask_ai(prompt, 3000)

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
    return ask_ai(prompt, 2000)

def is_followup_question(text: str) -> bool:
    """Return True if the user is asking a follow-up rather than making a statement."""
    lower = text.lower().strip()
    if "?" in text:
        return True
    followup_phrases = [
        "what do you mean", "in what way", "how so", "can you explain",
        "tell me more", "why do you", "what is that", "how does that",
        "what's that", "give me an example", "like what", "how come",
        "what does that mean", "i don't understand", "not sure what you mean"
    ]
    return any(p in lower for p in followup_phrases)


def get_followup_answer(perspective_text, user_question, decision, profile_str, history):
    """Answer a follow-up question briefly and stay in conversation."""
    prompt = f"""The user asked a follow-up about something you said. Answer in 2-3 short sentences. Be direct and specific to their situation. Do NOT introduce a new challenge, bias, or option.

WHAT YOU SAID: {perspective_text}
THEIR QUESTION: {user_question}
DECISION: {decision}
PROFILE: {profile_str}
HISTORY:
{history}"""
    return ask_ai(prompt, 1000)


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
    return ask_ai(prompt, 1000)

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
        result = ask_ai(prompt, 800)
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


def detect_response_type(user_message: str, turn_num: int = 99) -> str:
    """
    Returns: 'convinced' | 'not_convinced'

    convinced     -> user explicitly signals they accept the new option or feel clear
    not_convinced -> anything else (default — conviction must be explicit)
    """
    if turn_num <= 2:
        return "not_convinced"

    msg = user_message.lower().strip()

    conviction_signals = [
        "that makes sense", "you're right", "i think you're right",
        "that's a good point", "i hadn't thought", "that resonates",
        "maybe i should", "i'm starting to think", "that actually",
        "i can see", "i agree", "that helps", "i feel clearer",
        "i'm leaning towards", "i've decided", "i think i should"
    ]

    resistance_signals = [
        "but ", "however", "unfortunately", "that won't", "i can't",
        "it's not that simple", "yes but", "yeah but", "i don't think",
        "that's not", "the problem is", "the issue is", "what about",
        "in what way", "how would", "what if"
    ]

    has_question = "?" in user_message
    word_count = len(msg.split())

    if any(s in msg for s in conviction_signals):
        return "convinced"

    if has_question or any(msg.startswith(w) for w in ["how", "why", "what", "but", "yes but", "yeah but"]):
        return "not_convinced"

    if any(s in msg for s in resistance_signals):
        return "not_convinced"

    if word_count < 10 and not any(s in msg for s in conviction_signals):
        return "not_convinced"

    return "not_convinced"


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
    return ask_ai(prompt, 1500)

def get_decision_domain(decision, profile):
    """Classify the decision into a domain for Phase 3 longitudinal analysis."""
    prompt = f"""Classify this decision into ONE of these domains:
career, education, relationships, finance, location, identity, health, lifestyle, other

Decision: {decision}

Output only the single domain word, lowercase, nothing else."""
    return ask_ai(prompt, 50)

def classify_domain(decision: str) -> str:
    return get_decision_domain(decision, profile=None)

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
        result = ask_ai(prompt, 600).strip()
        opts = [line.strip(" -•*").strip() for line in result.split("\n") if line.strip()]
        # Filter to reasonable lengths
        return [o for o in opts if 2 <= len(o.split()) <= 8][:4]
    except Exception:
        return []
