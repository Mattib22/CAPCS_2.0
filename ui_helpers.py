# CAPCS — pure UI utility functions

import streamlit as st
import re


def confidence_color(score):
    if score >= 75: return "#d4edda", "#155724"
    if score >= 40: return "#fff3cd", "#856404"
    return "#f8d7da", "#721c24"

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

def thinking_animation(message="CAPCS is thinking"):
    """Render a branded animated loading indicator."""
    st.markdown(f"""
    <div class="capcs-thinking">
        <div class="capcs-icon">🧑‍🏫</div>
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


def inject_keepalive():
    """
    Inject a JS heartbeat that moves a hidden element every 30 s.
    This prevents the browser from suspending the Streamlit WebSocket
    connection when the tab is idle, which is the main cause of
    short-inactivity logouts on Streamlit Cloud.
    """
    st.markdown("""
<script>
(function() {
    if (window._capcsKeepalive) return;   // only one instance
    window._capcsKeepalive = setInterval(function() {
        // Touch a hidden div — enough to signal activity to the browser
        var el = window.parent.document.getElementById('capcs-keepalive');
        if (!el) {
            el = window.parent.document.createElement('div');
            el.id = 'capcs-keepalive';
            el.style.display = 'none';
            window.parent.document.body.appendChild(el);
        }
        el.setAttribute('data-ts', Date.now());
    }, 30000);
})();
</script>
""", unsafe_allow_html=True)


def scroll_to_chat_bottom():
    """Inject JS to scroll the chat to the bottom after new messages are rendered."""
    st.markdown(
        """<script>
        setTimeout(function() {
            var sel = [
                'section[data-testid="stMain"]',
                '[data-testid="stAppViewContainer"]',
                'section.main',
                '.main'
            ];
            for (var i = 0; i < sel.length; i++) {
                var el = window.parent.document.querySelector(sel[i]);
                if (el) { el.scrollTop = el.scrollHeight; break; }
            }
        }, 120);
        </script>""",
        unsafe_allow_html=True
    )


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
