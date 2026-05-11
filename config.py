# CAPCS — constants and CSS styles (no Streamlit calls, no logic)

MAX_ROUNDS = 5
PROBE_TURNS = 3   # Turns 1-3 = probing questions only; Turn 4+ = challenge with bias
PROFILE_VERSION = "4"  # Bumped: added passions question

# Canonical labels — use these everywhere instead of magic strings
UNDECIDED_INITIAL_LABEL = "I'm genuinely undecided"   # shown on decision input
UNDECIDED_MID_SESSION_LABEL = "I'm still undecided"   # shown in round radio
OTHER_LABEL = "Other"

CSS_STYLES = """
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
"""
