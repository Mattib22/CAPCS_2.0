# CASPER — all Gemini AI calls and prompt functions (v2)

import streamlit as st
import os
import random
import google.generativeai as genai
import json
import re


BIAS_TO_DIMENSION = {
    "Sunk Cost Fallacy": 1, "Idealization Bias": 1, "Projection Bias": 1,
    "Overconfidence Bias": 1, "Halo Effect": 1,
    "Anticipated Regret": 2, "Loss Aversion": 2, "Status Quo Bias": 2, "Omission Bias": 2,
    "False Dichotomy": 3, "Overgeneralization": 3, "Constraint Fixation": 3,
    "Availability Heuristic": 3, "Confirmation Bias": 3, "Anchoring Bias": 3,
    "Social Proof Bias": 3,
}

CASPER_DIMENSIONS = """
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


def _shuffled_dim_bias_lists() -> dict:
    """Return {1: [bias, ...], 2: [...], 3: [...]} with each list shuffled."""
    dim_biases: dict = {}
    current_dim = 0
    for line in CASPER_DIMENSIONS.strip().splitlines():
        if line.startswith("DIMENSION"):
            try:
                current_dim = int(line.split()[1])
            except (IndexError, ValueError):
                pass
            dim_biases[current_dim] = []
        elif line.startswith("-") and current_dim:
            name = line[1:].split(":")[0].strip()
            dim_biases[current_dim].append(name)
    for lst in dim_biases.values():
        random.shuffle(lst)
    return dim_biases


def _shuffled_dimensions() -> str:
    """Return CASPER_DIMENSIONS with dimensions and biases in randomised order.
    Prevents the model from anchoring on the first-listed biases every call."""
    sections = []
    current_header = None
    current_biases = []
    for line in CASPER_DIMENSIONS.strip().splitlines():
        if line.startswith("DIMENSION"):
            if current_header:
                random.shuffle(current_biases)
                sections.append((current_header, current_biases))
            current_header = line
            current_biases = []
        elif line.startswith("-"):
            current_biases.append(line)
    if current_header:
        random.shuffle(current_biases)
        sections.append((current_header, current_biases))
    random.shuffle(sections)
    return "\n".join(
        header + "\n" + "\n".join(biases)
        for header, biases in sections
    )


def _get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


def ask_ai(prompt, max_tokens=1200, json_mode=False):
    try:
        genai.configure(api_key=_get_api_key())
        gen_cfg = {"temperature": 0.8, "max_output_tokens": max_tokens}
        if json_mode:
            gen_cfg["response_mime_type"] = "application/json"
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt, generation_config=gen_cfg)
        try:
            text = response.text.strip()
        except ValueError:
            return ""
        text = text.lstrip("*").strip()
        for prefix in ["BIAS:","PERSPECTIVE:","EXPLANATION:","QUESTION:",
                       "Bias:","Perspective:","Explanation:","Question:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        # If response ends mid-sentence (no terminal punctuation), trim back
        # to the last complete sentence so the user sees something coherent.
        if text and text[-1] not in ".!?\"'":
            import re as _re
            last = max(
                (m.end() for m in _re.finditer(r'[.!?]["\']?\s', text)),
                default=0
            )
            if last > len(text) // 2:
                text = text[:last].strip()
        return text
    except Exception as e:
        # Store last error for debugging
        try:
            import streamlit as _st
            _st.session_state["_ask_ai_last_error"] = str(e)[:200]
        except Exception:
            pass
        return ""


# ── NEW CONVERSATIONAL AI FUNCTIONS ────────────────────────────────────────────

def get_opening_question(decision, options, confidence, profile_str, context,
                          longitudinal="", history=""):
    """
    Engineered three-question structure targeting the three bias dimensions.
    Q1: Dimension 1 — wants/values
    Q2: Dimension 2 — fears/losses
    Q3: Dimension 3 — assumptions
    """
    long_section = f"\n{longitudinal}" if longitudinal else ""
    confidence_label = (
        "completely lost" if confidence < 20 else
        "quite uncertain" if confidence < 40 else
        "roughly 50/50" if confidence < 60 else
        "leaning one way" if confidence < 80 else
        "fairly settled"
    )
    prompt = f"""You are having a real conversation with someone about an important decision in their life. You are warm, perceptive, and psychologically minded — like a trusted friend who also understands how people think.

Your job is to ask ONE question. Count the USER messages in the conversation history to know which question to ask:

0 user messages → ask Question 1
1 user message → ask Question 2
2 user messages → ask Question 3

---

QUESTION 1 — Open what this decision is really about for them

Step 1 — Find the emotionally loaded word in the decision text.
Read the DECISION field only. Find the single word or short concept that carries the most personal weight — a word the user chose that implies something about identity, freedom, belonging, security, or meaning. This word must come from what the user wrote, not from the profile. Do not invent a concept not in their text. Do NOT use the name of one of the options as the loaded word — "going to the pub" or "staying home" are options, not loaded concepts.
Examples of valid loaded words: "settle", "escape", "commit", "go back", "stay", "make it work", "give up", "finally".
If the decision text contains no emotionally loaded word beyond the option names, ask what the option they are leaning towards represents for them more broadly: "What is going to the pub really about for you?" Do not probe a word that isn't there.

Step 2 — Find the profile friction.
Read only the education, job, situation, and values fields. Find the one or two elements that create real friction with this specific decision — something that makes the choice more complicated or meaningful for this person. A profile element creates friction only if it directly conflicts with or complicates one of the two options.
Do NOT use the known_bias or blind spot field.
Do not cite profile elements that apply equally to both options.

Step 3 — Write the question.
Write a short, warm message that:
- Names both options naturally in one sentence
- Weaves in the profile element(s) that create friction
- Asks what the emotionally loaded word from Step 1 means to them personally

The question must NOT include any of the following — each one implies a specific bias pattern before the user has said anything:
- "at this point in your life" / "right now" — implies comparing present self to future self (Projection Bias tell)
- "what would you be giving up" — implies loss framing (Loss Aversion tell)
- "how settled are you" / "how certain" — implies questioning their confidence (Overconfidence tell)
- "what drew you to" / "what makes it appealing" — implies halo/idealization framing
- "have you already invested" / "how much have you put in" — implies sunk cost framing

Just ask what the loaded word means to them in the context of their specific situation. Let their answer introduce whatever frame is real for them.

Gold standard example:
"You're weighing continuing your travels in Australia against the idea of returning to Europe to settle, especially with your background in Cognitive Science and your current work in hospitality. What does 'settle' genuinely mean for you in this situation?"

Why this works: 'settle' comes from the user's own words. Profile elements create real tension. No temporal framing, no implied comparison, no suggestion of which pattern to confirm.

---

QUESTION 2 — Open what feels difficult or significant about this

Read their first answer. Do NOT diagnose a pattern yet. Find the thing they said that feels most emotionally significant — something they care about, something that weighs on them, something uncertain — and ask about THAT specific thing in a way that invites them to go deeper.

The goal is simply to hear more about what feels hard, heavy, or meaningful in this decision. The diagnostic will identify the pattern from their answer. Your job is only to create space for them to reveal it.

The question must:
- Come entirely from their actual words — not from a pattern template
- Feel like a curious friend who just heard something important and wants to understand it better
- Be open enough that any of the four D2 patterns could surface: avoiding a future feeling, fear of losing something, pull of the familiar, or preference for inaction

The question must NOT:
- Present a binary ("is it X or Y?") — binary frames pre-select the answer
- Use words like "cost", "lose", "give up", "leave behind" — these pre-frame the answer as loss territory and signal Loss Aversion
- Mention or imply a future feeling they might be avoiding — signals Anticipated Regret
- Ask whether the current situation is really serving them — signals Status Quo Bias
- Ask whether inaction is also a choice — signals Omission Bias

Just ask about what feels most significant or difficult about this specific situation for this person. Let their words tell you what kind of weight it is.

✗ BAD: "Is this driven by what you want, or by wanting to avoid regretting it later?" — reveals Anticipated Regret
✗ BAD: "What would you be walking away from?" — pre-frames as loss, reveals Loss Aversion
✗ BAD: "What matters most to you right now?" — "right now" implies present-state anchoring
✓ GOOD: "What feels hardest about this choice for you personally?"
✓ GOOD: "What's weighing on you most about this decision?"

---

QUESTION 3 — Invite them to look at their own thinking

Read their first two answers. Find ONE specific thing in what they said that is doing heavy lifting in their reasoning — a belief, a comparison they're making, an experience they're drawing on, an expectation about what others think, or something they've implied without examining. Ask a genuinely curious question about THAT specific thing.

The question must come entirely from their words — not from a pattern category. It should sound like someone who listened carefully and is simply curious about something specific they said, not like someone probing for a flaw in their thinking.

What to look for in their answers (any of these could be the thing worth asking about):
- A belief or conclusion they stated without explaining where it comes from
- A comparison or reference point they're anchoring everything to
- A past experience they seem to be treating as a universal guide
- An expectation about what others do, expect, or think
- An option or path they've dismissed without explaining why
- Something they said with certainty that might not be as settled as it sounds

The question must NOT use any pattern's signature framing:
- NOT anything about whether options are "really" limited — signals False Dichotomy
- NOT anything about whether a limit is fixed or changeable — signals Constraint Fixation
- NOT "how certain are you" — signals Constraint Fixation or Overconfidence
- NOT anything about whether evidence is balanced — signals Confirmation Bias
- NOT anything about what others think or do — signals Social Proof
- NOT anything about how typical or common something is — signals Availability Heuristic
- NOT anything about what they'd do without reference points — signals Anchoring

Just ask a direct, warm, curious question about the specific thing they said. Their answer will reveal what assumption is driving it.

✗ BAD: "Is going back to Europe really your only alternative?" — signals False Dichotomy
✗ BAD: "How certain are you that the relationship can't change?" — signals Constraint Fixation
✗ BAD: "How typical is that experience?" — signals Availability Heuristic
✓ GOOD: "What makes you say that [specific thing they stated]?"
✓ GOOD: "Where does [specific belief or comparison they mentioned] come from for you?"

---

RULES (apply to every question):
- One question only. Never two.
- Max 50 words for the entire message.
- Never ask about tasks, logistics, activities, or external facts.
- NEVER start with "You said..." — this is the most common failure mode. It makes the question feel like a transcript, not a conversation.
  ✗ BAD: "You said 'meeting new people'. What does buying groceries feel like it would do to that?"
  ✓ GOOD: "What would staying home actually cost you, beyond the practical side of things?"
- Do not echo their exact words back verbatim at the start of your message.
- Speak like a person in a conversation, not a form.
- Second person, warm, direct.
- End with exactly one question mark.

DECISION: {decision}
OPTIONS: {options}
USER IS LEANING TOWARDS: {options} ({confidence_label})
CONTEXT: {context}
PROFILE:
{profile_str}
CONVERSATION SO FAR:
{history}{long_section}"""
    return ask_ai(prompt, 4000)


def get_probing_question(decision, options, leaning, confidence, profile_str,
                          history, last_answer, context, longitudinal="", turn_num=2,
                          loop_context=""):
    """
    Loop-back probing question. When loop_context is set, the AI must narrow in on
    the specific contradiction or gap already revealed — not open new territory.
    """
    long_section = f"\n{longitudinal}" if longitudinal else ""

    if loop_context:
        # ── Loop-back pass: narrow in on the specific gap already revealed ────
        focus_instruction = f"""CONTEXT FROM PREVIOUS LOOP:
{loop_context}

You are NOT exploring fresh territory. The previous loop already revealed a specific tension, hesitation, or contradiction. Your job now is to NARROW IN on that specific gap — ask a question that goes directly into the heart of what's blocking resolution.

Do NOT ask a general question about feelings or life in broad terms.
Do NOT ask about new topics.
DO ask the question that makes the specific contradiction visible — the one the user is probably already aware of but hasn't fully faced.

The question should feel like: "you've been circling this, let me name what you're circling."

Examples of narrowing in (not opening up):
✓ "You said adventure matters but also security — what does it mean that you can't currently imagine having both?"
✓ "What would it mean to you if staying turned out to be the right choice?"
✗ "What part of this chapter of life would you feel most incomplete about?" (too broad — opens new territory)
✗ "What matters most to you right now?" (Q1-style — already done)"""
    else:
        # ── Standard diagnostic probe ─────────────────────────────────────────
        focus_instruction = """Your hidden goal: build a diagnostic picture of which cognitive bias is most active. You do not reveal this. Each question is designed to surface or rule out a specific hypothesis.

Your job this turn: re-read the full conversation. What pattern is becoming clearer? What contradiction has the user not yet confronted? Ask the question that gives you the sharpest evidence — not a generic open question.

What makes a good diagnostic question:
- It probes the specific mental mechanism you suspect, not feelings in general
- It reveals a contradiction or assumption the user hasn't examined
- It cannot be answered with facts or logistics"""

    prompt = f"""You are a psychologist-style thinking partner. You have been listening carefully.

{focus_instruction}

Rules (apply always):
- One question only, max 35 words, ends with ?
- Cannot be answered yes/no
- Do NOT name any bias, pattern, or psychological concept
- Do NOT ask about facts, logistics, or activities
- Do NOT open with "You mentioned", "You said", "You told me", or any echo of their words
- Warm, direct, curious — speaks to what is happening inside them

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
    return ask_ai(prompt, 4000)


def get_challenge_response(  # noqa: PLR0913
        decision, options, leaning, confidence, profile_str,
        history, last_answer, context, longitudinal="",
        emotion="neutral", turn_num=2,
        is_undecided=False, confidence_dropped=False,
        sustained_drop=False, confirmed_bias="",
        bias_resonance="full"):
    """
    State: counterattack. User confirmed a bias. Introduce the direct antidote option.
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

    confirmed_instruction = ""
    if confirmed_bias:
        confirmed_instruction = f"""The user has confirmed that **{confirmed_bias}** resonates with them — do not re-name or re-explain it.

THE COUNTERATTACKING OPTION — CRITICAL:

The user's current leaning is: {leaning or 'genuinely undecided'}
Their confidence in that leaning is: {confidence}%

Step 1: Identify what **{confirmed_bias}** is specifically blocking or distorting WITHIN the user's leaning. What is it making harder to see, harder to do, or harder to believe about the path they are already choosing?

Step 2: Use the confidence level to determine how far the option can deviate from the leaning:

- 60% and above: the option must stay WITHIN the leaning. Expand or clarify how the user sees their preferred path. Do NOT propose the other option or suggest they change direction.

- 35% to 59%: the option can bridge both directions. Propose a middle path that takes something meaningful from each option without fully committing to either.

- Below 35%: the option can more freely explore the other direction. The user is not strongly committed to their leaning and is genuinely open.

Step 3: Before writing the option, apply this test: "Does this option help the user move forward on {leaning or 'their preferred direction'}, or does it push them toward the other option?" If confidence is 60%+ and the option pushes toward the other option — rewrite it.

The option must address the specific bias detected.
Confidence determines the direction of the option.
Bias determines the content of the option.

The option must be grounded in what the user said in the conversation — including any constraints they mentioned (visa, money, location, time, projects already in progress). Do NOT generate from profile alone.

Max 80 words total. No re-explanation of the bias — the spark already did that. Start directly with the option.
Do NOT ask a question at the end — a confidence slider will follow your message.

The user's response to the bias was: {bias_resonance} ("full" / "partial" / "none")
- full: standard counterattack — write as normal
- partial: open with "Even if that pattern doesn't feel completely like yours..." then continue with the option
- none: open with "Setting that aside — here's a different way to look at this..." and frame the option as a fresh angle, not a confirmation of the bias

Tone: warm, direct."""

    if confirmed_instruction:
        prompt = f"""You are a thinking partner helping someone work through a real decision.

{confirmed_instruction}

FULL CONVERSATION — read every turn before writing anything:
{history}

DECISION: {decision}
OPTIONS CONSIDERED SO FAR: {options}
LEANING: {leaning or 'genuinely undecided'}
CONFIDENCE: {confidence}%
CONTEXT: {context}
PROFILE:
{profile_str}{long_section}

After your conversational message, output this block exactly as shown — this is required:
---EXTRACT---
BIAS: [bias name only — max 6 words]
EXPLANATION: [what this bias is and why it appeared — max 40 words]
PERSPECTIVE: [the option you proposed — max 8 words, no punctuation]
---END---"""
        return ask_ai(prompt, 4096)

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

3. COUNTERATTACK (2 sentences): Introduce ONE concrete option as the direct antidote to the bias. The option MUST reference at least one specific word, fear, desire, or constraint the user actually mentioned — not a generic alternative invented from the profile alone. Ground it in the user's specific constraints (money, location, relationship, visa, time — whatever they mentioned). Warm structure: "One way to counteract this is [Option] — [one sentence why this is the antidote, specific to their context]." Do NOT say "the counterforce is X" — that sounds mechanical. Then end with ONE question that asks the user to EVALUATE the option, not plan within it.
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

FULL CONVERSATION — read every turn before writing anything:
{history}

BIAS DETECTION RULES (follow strictly):
- Read ALL user answers in the conversation above, not just the last one
- The bias you name must explain a pattern visible across MULTIPLE answers, not a single response
- Cross-reference what the user said with their PROFILE below — the bias should be grounded in both their words AND who they are
- Quote or paraphrase something from an EARLIER turn (not the last message) when naming the bias
- The final question must ask the user to EVALUATE the new option, not plan within it
  WRONG: "How could X fit your travel plans?" (assumes acceptance)
  RIGHT: "Does X feel like something that could actually work for you, or does something about it feel off?"

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
    return ask_ai(prompt, 4000)

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
    return ask_ai(prompt, 4000)

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
    return ask_ai(prompt, 4000)

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
    return ask_ai(prompt, 4000)

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
    return ask_ai(prompt, 4000)


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
    return ask_ai(prompt, 4000)

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
        result = ask_ai(prompt, 4000)
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


def has_enough_signal(conversation_history: list) -> bool:
    """Lightweight check: enough signal in conversation to name a specific earned bias?"""
    history_text = "\n".join([
        f"{'CASPER' if m['role'] == 'assistant' else 'USER'}: {m['content']}"
        for m in conversation_history
    ])
    prompt = f"""Read this conversation carefully.

{history_text}

Is there enough signal in the user's answers to identify a specific, earned cognitive bias — one grounded in what they actually said across their responses, not a generic one?

Answer YES or NO only. Nothing else."""
    result = ask_ai(prompt, 50)
    return result.strip().upper().startswith("YES")


def get_spark_message(conversation_history: list, profile_str: str, context: str,
                      rejected_biases: list = None,
                      recently_used_biases: list = None,
                      pre_identified_bias: str = "") -> str:
    """
    Spark turn. If pre_identified_bias is set, skips selection and writes
    the narrative for that specific bias. Otherwise runs the full selection.
    Returns spark message + BIAS_NAME/BIAS_EXPLANATION fields.
    Returns 'SIGNAL_INSUFFICIENT' if no bias is clearly earned.
    """
    if rejected_biases is None:
        rejected_biases = []
    if recently_used_biases is None:
        recently_used_biases = []

    history_text = "\n".join([
        f"{'CASPER' if m['role'] == 'assistant' else 'USER'}: {m['content']}"
        for m in conversation_history
    ])

    # ── Fast path: bias already identified by the diagnostic ─────────────────
    if pre_identified_bias:
        prompt = f"""The active cognitive bias in this conversation has been identified as: {pre_identified_bias}

Write the spark message as a hypothesis, not a diagnosis. Use words like "might", "could be", "there's a pattern that might be at play here." Never write "this bias is making you" or "you are experiencing X." The tone should feel like a perceptive friend noticing something, not a clinician delivering a verdict. The bias name appears naturally mid-sentence, not as a label.

Write a spark message of 2-3 sentences (max 60 words) that:
1. Opens with a concrete observation grounded in something the user specifically said — reference their actual words or situation, not a generic statement. E.g. "The way you described X suggests..." or "When you said Y, there's a pattern underneath that..."
2. Names '{pre_identified_bias}' naturally mid-sentence as a possibility: "this might be {pre_identified_bias}" or "there's a pattern here that could be {pre_identified_bias}"
3. Suggests in one sentence what it might be doing to their thinking in this specific situation — use conditional language ("which might be making...", "this could be why...")

Do NOT introduce any option or ask a question.
Write in second person only.
Never open with a generic statement that could apply to anyone — always anchor to what this person actually said.

After the message, output on new lines:
BIAS_NAME: {pre_identified_bias}
BIAS_EXPLANATION: [one plain English sentence defining what this bias is — generic, not personalised]
D1_BIAS: [most likely bias from: {", ".join(_shuffled_dim_bias_lists().get(1, []))}]
D1_SCORE: [confidence 1-10]
D2_BIAS: [most likely bias from: {", ".join(_shuffled_dim_bias_lists().get(2, []))}]
D2_SCORE: [confidence 1-10]
D3_BIAS: [most likely bias from: {", ".join(_shuffled_dim_bias_lists().get(3, []))}]
D3_SCORE: [confidence 1-10]

FULL CONVERSATION:
{history_text}

CONTEXT: {context}
PROFILE:
{profile_str}"""
        return ask_ai(prompt, 4000)

    rejected_note = ""
    if rejected_biases:
        rejected_note = f"\nBiases tried this session that the user rejected — do NOT reuse: {', '.join(rejected_biases)}\n"
    if recently_used_biases:
        rejected_note += f"\nBiases identified in this user's recent sessions — strongly avoid repeating unless the evidence is overwhelming and no other bias fits: {', '.join(recently_used_biases)}\n"

    prompt = f"""You are an expert psychologist. Your task: identify the SINGLE most precise cognitive bias active in this conversation — not the most obvious one, the most precise one.

This is a precision task, not a pattern-matching task. Read for what is SPECIFIC and UNIQUE about how this person thinks.

SELECTION PROCESS — follow these steps in order:
1. Read the full conversation. What is the core distortion? Not what the decision is about, but what specific mental pattern is distorting the user's reasoning.
2. Consider all 15 biases in the taxonomy simultaneously — do NOT pre-filter by decision type or category.
3. HARD CONSTRAINT on over-diagnosed biases: Before selecting Loss Aversion, Status Quo Bias, or Sunk Cost Fallacy, you MUST ask: "Is there another bias in the list that explains this pattern equally well or better?" If yes — use that other bias. These three are systematically over-diagnosed; only fall back to them if they are uniquely the best explanation and no other bias fits.
4. The bias must be EARNED by specific evidence from what the user actually said — not inferred from the decision type alone (e.g., "it's a career decision so probably Status Quo Bias").
5. Prefer the bias that would genuinely SURPRISE the user — the one they haven't already thought of themselves.
{rejected_note}
{_shuffled_dimensions()}

If no bias from this list is clearly earned from the full conversation, respond with exactly:
SIGNAL_INSUFFICIENT

Write the spark message as a hypothesis, not a diagnosis. Use words like "might", "could be", "there's a pattern that might be at play here." Never write "this bias is making you" or "you are experiencing X." The tone should feel like a perceptive friend noticing something, not a clinician delivering a verdict. The bias name appears naturally mid-sentence, not as a label.

Otherwise write a spark message of 2-3 sentences (max 60 words) that:
1. Opens with a tentative observation about their thinking — not a quote or paraphrase of what they said; use hedging language ("there might be", "one pattern that could be at play")
2. Names the bias naturally mid-sentence as a possibility: "this might be Y" or "there's a pattern here that could be Y"
3. Suggests in one sentence what this bias might be doing to their thinking — use conditional language ("which might be making...", "this could be why...")

Do NOT introduce any new option. Do NOT ask a question.
Write in second person only. Never write "The user" or refer to the user in third person.
Never open with a direct quote or paraphrase of the user's words. The following openers are all forbidden: "You said", "You mentioned", "You told me", "You noted", "You described", "When you say", "As you said", "Since you said", "Given that you said", "That feeling of '[quote]'", "Your feeling that '[quote]'", "The user states", "It sounds like you", "You're feeling that", "In saying that".
Never open with a third-person observation about their thinking as if viewed from the outside — "The certainty with which you're treating X..." is forbidden because it frames the user as an object of analysis. Speak from inside the conversation: "That pull toward X...", "What's keeping you from...", "There's a pattern here..."
Start with your own direct observation — e.g. "That pull toward X is making Y feel like the only option" or "There's a pattern here where X keeps getting treated as fixed..."

IMPORTANT — you MUST include all structured fields on new lines after the message:
BIAS_NAME: [bias name only — max 6 words]
BIAS_EXPLANATION: [one plain English sentence defining what this bias is — generic, not personalised]
D1_BIAS: [most likely bias from: {", ".join(_shuffled_dim_bias_lists().get(1, []))}]
D1_SCORE: [confidence 1-10]
D2_BIAS: [most likely bias from: {", ".join(_shuffled_dim_bias_lists().get(2, []))}]
D2_SCORE: [confidence 1-10]
D3_BIAS: [most likely bias from: {", ".join(_shuffled_dim_bias_lists().get(3, []))}]
D3_SCORE: [confidence 1-10]

FULL CONVERSATION:
{history_text}

CONTEXT: {context}
PROFILE:
{profile_str}"""

    return ask_ai(prompt, 4000)


def extract_spark_fields(spark_response: str) -> dict:
    """Parse structured fields from get_spark_message output."""
    import re as _re
    result = {
        "spark_message": "", "bias_name": "", "bias_explanation": "",
        "dim_biases": {},  # {1: (bias, score), 2: ..., 3: ...}
    }

    def _field(pattern):
        m = _re.search(pattern, spark_response, _re.IGNORECASE)
        return m.group(1).splitlines()[0].strip().strip("*_").strip() if m else ""

    result["bias_name"]        = _field(r'BIAS[\s_]?NAME\s*:\s*(.+)')
    result["bias_explanation"] = _field(r'BIAS[\s_]?EXPLANATION\s*:\s*(.+)')

    for d in (1, 2, 3):
        bias  = _field(rf'D{d}[\s_]?BIAS\s*:\s*(.+)')
        score_raw = _field(rf'D{d}[\s_]?SCORE\s*:\s*(.+)')
        try:
            score = max(1, min(10, int(_re.search(r'\d+', score_raw).group())))
        except Exception:
            score = 5
        if bias:
            result["dim_biases"][d] = (bias, score)

    # Enforce consistency: the chosen bias (BIAS_NAME) must appear in its correct
    # dimension slot and have the highest score among the three dimensions.
    chosen = result["bias_name"]
    if chosen:
        chosen_dim = BIAS_TO_DIMENSION.get(chosen, 0)
        if chosen_dim:
            existing = result["dim_biases"].get(chosen_dim)
            existing_score = existing[1] if existing else 5
            # Place chosen bias in its dimension with at least the existing score
            result["dim_biases"][chosen_dim] = (chosen, max(existing_score, 8))
            # Ensure chosen bias has the top score across all dimensions
            max_other = max((s for d, (_, s) in result["dim_biases"].items()
                             if d != chosen_dim), default=0)
            cur_score = result["dim_biases"][chosen_dim][1]
            if cur_score <= max_other:
                result["dim_biases"][chosen_dim] = (chosen, max_other + 1)

    # Narrative = everything before the first structured field line
    first_field_m = _re.search(
        r'[\*\_\-\s]*(?:BIAS[\s_]?NAME|D[123][\s_]?BIAS)\s*:',
        spark_response, _re.IGNORECASE
    )
    cutoff = len(spark_response)
    if first_field_m:
        line_start = spark_response.rfind("\n", 0, first_field_m.start())
        cutoff = line_start if line_start != -1 else first_field_m.start()

    narrative = spark_response[:cutoff].strip()
    narrative = _re.sub(r'\n[-\*—]{2,}\s*$', '', narrative).strip()
    result["spark_message"] = narrative

    return result


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
- Max 100 words
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
    return ask_ai(prompt, 4000)

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
        result = ask_ai(prompt, 4000).strip()
        opts = [line.strip(" -•*").strip() for line in result.split("\n") if line.strip()]
        # Filter to reasonable lengths
        return [o for o in opts if 2 <= len(o.split()) <= 8][:4]
    except Exception:
        return []


def get_counterattack_followup(conversation_history: list, user_message: str,
                                option_proposed: str, bias_confirmed: str,
                                profile_str: str, context: str) -> str:
    """
    Handles free-text user responses during the counterattack state.
    Classifies the response and routes:
      - Clarifying question  → answer directly, stay in counterattack (CONTINUE)
      - Request for alternatives → offer a refined option, stay in counterattack (CONTINUE)
      - Pushback / objection → acknowledge and refine, stay in counterattack (CONTINUE)
      - Conviction signal    → warm acknowledgement + CONVINCED signal

    Output format:
      [conversational response — max 80 words]
      ---SIGNAL---
      ROUTE: CONVINCED or CONTINUE
      ---END---
    """
    history_text = "\n".join([
        f"{'CASPER' if m['role'] == 'assistant' else 'USER'}: {m['content']}"
        for m in conversation_history
    ])

    prompt = f"""You are a thinking partner mid-conversation. You proposed a specific option to help someone break a cognitive bias, and they've responded. Read their response carefully.

OPTION YOU PROPOSED: {option_proposed}
BIAS IDENTIFIED: {bias_confirmed}
USER'S RESPONSE: {user_message}

Their response falls into one of four types — identify which and reply accordingly:

TYPE 1 — CLARIFYING QUESTION: They want to understand the option better ("What does that mean?", "How would that work?", "Like what exactly?")
→ Answer directly and concretely in 2-3 sentences. Be specific to their situation — not generic. End with: "Does that make it feel more or less workable for you?"

TYPE 2 — REQUESTING ALTERNATIVES: They want a different option ("Are there other ways?", "What else could work?", "That specific thing won't work for me")
→ One sentence acknowledging what didn't land. Offer ONE different option that still counters {bias_confirmed} but from a different angle. Ground it in something specific they said. End with the evaluation question.

TYPE 3 — PUSHBACK / OBJECTION: They name a specific reason the option won't work ("I can't because...", "That assumes X", "Yes but what about Y")
→ One sentence acknowledging the specific constraint they named. Refine the option to account for it — if the constraint makes it genuinely unworkable, modify it rather than abandon it. End with the evaluation question.

TYPE 4 — CONVICTION SIGNAL: They clearly accept or feel ready ("Yes, that makes sense", "I think that's it", "OK I get it now", "That actually resonates", "I'm going to do that")
→ One warm sentence of genuine acknowledgement — something that reflects what specifically shifted for them, not "Great!" Then output the signal.

The evaluation question for types 1–3: "Does this feel like something that could actually work for you, or does something about it feel off?"

FULL CONVERSATION:
{history_text}

CONTEXT: {context}
PROFILE:
{profile_str}

Rules:
- Max 80 words for the conversational response
- Second person, warm, direct
- Never say "I notice", "it seems like", or "the counterforce"
- Never open with hollow affirmations ("Great!", "Absolutely!", "Of course!")
- Reference the user's specific words — not generic alternatives from the profile alone

Output your response, then on new lines:
---SIGNAL---
ROUTE: CONVINCED or CONTINUE
---END---"""

    return ask_ai(prompt, 4096)


def extract_counterattack_signal(response: str) -> dict:
    """
    Parse the ---SIGNAL--- block from get_counterattack_followup output.
    Returns dict with 'message' (what the user sees) and 'route' (CONVINCED or CONTINUE).
    Falls back to CONTINUE on any parse failure.
    """
    default = {"message": response.strip(), "route": "CONTINUE"}
    try:
        if "---SIGNAL---" not in response:
            return default
        message = response.split("---SIGNAL---")[0].strip()
        signal_block = response.split("---SIGNAL---")[1].split("---END---")[0].strip()
        route = "CONTINUE"
        for line in signal_block.split("\n"):
            if line.strip().upper().startswith("ROUTE:"):
                val = line.split(":", 1)[1].strip().upper()
                if val in ("CONVINCED", "CONTINUE"):
                    route = val
        return {"message": message, "route": route}
    except Exception:
        return default


def get_counterattack_followup(question: str, option_proposed: str,
                               conversation_history: list, profile_str: str) -> str:
    """
    Brief response to a user question about the counterattack option.
    Stays focused on the proposed option — does not open new territory.
    """
    history_text = "\n".join([
        f"{'CASPER' if m['role'] == 'assistant' else 'USER'}: {m['content']}"
        for m in conversation_history[-8:]
    ])
    prompt = f"""You are CASPER, a psychologist-style thinking partner.
You proposed this option to the user: {option_proposed}

The user responded with:
USER: {question}

If they asked a direct question, answer it concretely and specifically — draw on their profile and the proposed option as context.
If they pushed back or expressed doubt, engage with it honestly and directly.
Do not deflect with another question. Do not ask them what they think. Just respond.

Keep it to 2-4 sentences. Warm, direct, specific to their situation.

RECENT CONVERSATION:
{history_text}

PROFILE:
{profile_str}

Output only your response."""
    return ask_ai(prompt, 2000) or "The option is about using renting's lower financial commitment to create space for income growth — whether through upskilling, a role change, or building savings over time."


def get_partial_probe(option_proposed: str, confirmed_bias: str,
                       conversation_history: list, profile_str: str,
                       above_threshold: bool = False) -> str:
    """
    When the user explores further after the counterattack.
    - above_threshold=False: user is below their clarity bar — surface what's blocking.
    - above_threshold=True: user is above their clarity bar but still uncertain —
      surface what would take them all the way.
    """
    history_text = "\n".join([
        f"{'CAPCS' if m['role'] == 'assistant' else 'USER'}: {m['content']}"
        for m in conversation_history
    ])

    if above_threshold:
        framing = f"""The user is largely convinced by this option but something is still holding back full commitment — they chose to keep exploring rather than move forward.

OPTION PROPOSED: {option_proposed}
CONFIRMED BIAS: {confirmed_bias}

Your job: ask ONE question that surfaces what the remaining hesitation is about. Not what doesn't work logistically — what's the internal gap between being mostly convinced and fully clear.

The answer is probably a fear, a value conflict, or something they haven't admitted to themselves yet. Your question should make it visible.

Examples:
✓ "What are you protecting by not fully committing to this?"
✓ "What would need to be true for you to go all the way with it?"
✓ "When you imagine actually doing it, what feeling comes up first?"
✗ "What doesn't work about it?" (too broad)
✗ "What are your concerns?" (too generic)"""
    else:
        framing = f"""The user evaluated this option and something isn't fully landing — they haven't rejected it, but they can't fully commit either.

OPTION PROPOSED: {option_proposed}
CONFIRMED BIAS: {confirmed_bias}

Your job: ask ONE question that surfaces what's blocking internally — not a logistical objection, but a fear, an assumption, a value conflict, or something unconscious.

Examples:
✓ "What would it cost you personally to go in that direction?"
✓ "What are you protecting by not fully committing to this?"
✓ "What feeling comes up when you imagine actually doing it?"
✗ "What specifically doesn't work?" (too logistical)
✗ "Can you explain your hesitation?" (too generic)"""

    prompt = f"""{framing}

Rules:
- Max 30 words
- One question only, ends with ?
- Do NOT ask about facts, logistics, or external constraints
- Warm, psychologist-like tone — genuinely curious, not evaluative

FULL CONVERSATION:
{history_text}

PROFILE:
{profile_str}

Output only the question."""
    return ask_ai(prompt, 400)


def identify_candidate_biases(conversation_history: list, profile_str: str,
                               context: str,
                               rejected_biases: list = None) -> list:
    """
    Diagnostic step after listening questions.
    Returns up to 3 candidate biases (one per dimension) sorted by score.
    rejected_biases: biases already tried this session — excluded from results.
    """
    import json as _json, re as _re

    if rejected_biases is None:
        rejected_biases = []

    # Split answers into earlier rounds vs most recent to give recency weight
    all_user_msgs = [m for m in conversation_history if m.get("role") == "user"
                     and not m.get("content", "").startswith("[")]
    if not all_user_msgs:
        return []

    recent_cutoff = max(1, len(all_user_msgs) - 2)
    earlier_turns = "\n".join(f"USER (earlier): {m['content']}"
                               for m in all_user_msgs[:recent_cutoff])
    recent_turns  = "\n".join(f"USER (most recent): {m['content']}"
                               for m in all_user_msgs[recent_cutoff:])

    rejected_note = (f"\nDo NOT return any of these — already tried this session: "
                     f"{', '.join(rejected_biases)}\n") if rejected_biases else ""

    prompt = f"""Score one cognitive bias per dimension based on what the user said.
Give more weight to the most recent answers — they reflect where the person is NOW.{rejected_note}

EARLIER ANSWERS:
{earlier_turns}

MOST RECENT ANSWERS (weight these more heavily):
{recent_turns}

CONTEXT: {context}

DIMENSION 1 BIASES: Sunk Cost Fallacy, Idealization Bias, Projection Bias, Overconfidence Bias, Halo Effect
DIMENSION 2 BIASES: Anticipated Regret, Loss Aversion, Status Quo Bias, Omission Bias
DIMENSION 3 BIASES: False Dichotomy, Overgeneralization, Constraint Fixation, Availability Heuristic, Confirmation Bias, Anchoring Bias, Social Proof Bias

Output exactly 3 lines — one per dimension in order D1, D2, D3.
Each line: bias name, score (1-10), one short evidence quote separated by |
Score meaning: 8-10 strong, 5-7 moderate, 1-4 weak.

Example output:
Projection Bias | 6 | I keep thinking I will feel the same way
Anticipated Regret | 4 | I don't want to regret not trying
False Dichotomy | 9 | Either I travel or I settle"""

    result = ask_ai(prompt, 300)
    try:
        import streamlit as _st
        _st.session_state["_debug_raw_diagnostic"] = repr(result[:300]) if result else "<empty>"
    except Exception:
        pass
    try:
        if not result or not result.strip():
            return []
        valid = []
        dim = 1
        for line in result.strip().splitlines():
            line = line.strip()
            if not line or dim > 3:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                bias = parts[0].strip(" -•*1234567890.")
                try:
                    score = max(1, min(10, int(_re.search(r'\d+', parts[1]).group())))
                except Exception:
                    score = 5
                evidence = parts[2].strip() if len(parts) >= 3 else ""
                if bias:
                    valid.append({"bias": bias, "dimension": dim, "evidence": evidence, "score": score})
                    dim += 1
        valid.sort(key=lambda x: x["score"], reverse=True)
        return valid[:3]
    except Exception as e:
        try:
            import streamlit as _st
            _st.session_state["_debug_raw_diagnostic"] += f" | err:{e}"
        except Exception:
            pass
        return []


def get_disambiguation_question(cand_a: dict, cand_b: dict,
                                 conversation_history: list, profile_str: str,
                                 decision: str, context: str) -> str:
    """
    Two candidate biases are close. Generate ONE question whose answer would
    clearly differentiate between them, without naming either bias.
    """
    history_text = "\n".join([
        f"{'CAPCS' if m['role'] == 'assistant' else 'USER'}: {m['content']}"
        for m in conversation_history
    ])
    prompt = f"""You are diagnosing between two possible cognitive biases that both seem active in this conversation.

CANDIDATE A: {cand_a['bias']}
Evidence so far: {cand_a['evidence']}

CANDIDATE B: {cand_b['bias']}
Evidence so far: {cand_b['evidence']}

Write ONE question whose answer will clearly tell you which of these two biases is dominant.
- If A is dominant, the user's answer will confirm {cand_a['bias']}
- If B is dominant, the user's answer will confirm {cand_b['bias']}

The question must feel completely natural — like a psychologist following a genuine thread of curiosity. The user must not sense they're being tested between two options.

Rules:
- Max 35 words
- One question only, ends with ?
- Warm and direct — speaks to what's happening inside them
- Do NOT name any bias or psychological concept
- Do NOT repeat a question already asked in the conversation

FULL CONVERSATION:
{history_text}

DECISION: {decision}
CONTEXT: {context}
PROFILE:
{profile_str}

Output only the question."""
    return ask_ai(prompt, 400)


def get_personalised_suggestions(final_choice: str, confirmed_bias: str,
                                   profile_str: str, what_shifted: str = "") -> str:
    """Conviction state: warm closing with 2-3 personalised concrete directions."""
    shifted_section = f"\nUSER SAID WHAT SHIFTED: {what_shifted}" if what_shifted else ""
    prompt = f"""You are closing a decision-making session. The user has landed on their choice after working through a cognitive bias.

Write a warm, specific closing message that does two things:
1. Opening (1 sentence): Acknowledge what just happened — name the choice and what it means for this specific person given their profile.
2. Suggestions (2-3 bullet points): Concrete next directions given their final choice, the bias that was identified, and their profile. Each bullet must be specific to their situation — not generic advice.

Format:
[Opening sentence]

- [Concrete direction 1 — specific to their profile/situation]
- [Concrete direction 2 — specific to their profile/situation]
- [Concrete direction 3 — optional, only if genuinely different and useful]

Max 100 words total. Second person, warm. No hollow affirmations. Every direction must name something specific to this person.

FINAL CHOICE: {final_choice}
CONFIRMED BIAS: {confirmed_bias}{shifted_section}
PROFILE:
{profile_str}"""
    return ask_ai(prompt, 4000)
