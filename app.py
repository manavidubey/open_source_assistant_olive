"""
AI Personal Assistant — Streamlit Application

Premium chat interface supporting both OSS (Qwen 2.5) and Frontier
(Gemini / OpenAI) models with side-by-side comparison mode.
"""

import streamlit as st
import time
import json
from config import Config
from assistants.oss_assistant import OSSAssistant
from assistants.frontier_assistant import FrontierAssistant
from observability.logger import ObservabilityLogger

# ── Page Configuration ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Personal Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #F1F5F9;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 16px !important;
        margin-bottom: 12px !important;
        border: 1px solid rgba(99, 102, 241, 0.1) !important;
    }

    /* Title styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1 0%, #06B6D4 50%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .sub-title {
        text-align: center;
        color: #94A3B8;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(6, 182, 212, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #6366F1;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .status-safe {
        background: rgba(34, 197, 94, 0.15);
        color: #22C55E;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .status-blocked {
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Model selector pills */
    .stRadio > div {
        display: flex;
        gap: 8px;
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 16px 0;
    }

    div[data-testid="stVerticalBlock"] > div:has(> div.stButton) {
        display: flex;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize Session State ────────────────────────────────────
def init_session():
    if "config" not in st.session_state:
        st.session_state.config = Config.from_env()

    if "logger" not in st.session_state:
        st.session_state.logger = ObservabilityLogger()

    if "oss_assistant" not in st.session_state:
        st.session_state.oss_assistant = None

    if "frontier_assistant" not in st.session_state:
        st.session_state.frontier_assistant = None

    if "messages_oss" not in st.session_state:
        st.session_state.messages_oss = []

    if "messages_frontier" not in st.session_state:
        st.session_state.messages_frontier = []

    if "active_model" not in st.session_state:
        st.session_state.active_model = "oss"

    if "mode" not in st.session_state:
        st.session_state.mode = "single"


def get_assistant(model_type: str):
    """Get or create the specified assistant."""
    config = st.session_state.config

    if model_type == "oss":
        if st.session_state.oss_assistant is None:
            st.session_state.oss_assistant = OSSAssistant(
                model_id=config.hf_model_id,
                hf_token=config.hf_api_token,
                system_prompt=config.system_prompt,
                max_memory_turns=config.max_memory_turns,
                enable_guardrails=config.enable_guardrails,
                logger=st.session_state.logger,
            )
        return st.session_state.oss_assistant

    elif model_type == "frontier":
        if st.session_state.frontier_assistant is None:
            st.session_state.frontier_assistant = FrontierAssistant(
                provider=config.frontier_provider,
                gemini_api_key=config.gemini_api_key,
                gemini_model=config.gemini_model,
                openai_api_key=config.openai_api_key,
                openai_model=config.openai_model,
                system_prompt=config.system_prompt,
                max_memory_turns=config.max_memory_turns,
                enable_guardrails=config.enable_guardrails,
                logger=st.session_state.logger,
            )
        return st.session_state.frontier_assistant


# ── Sidebar ──────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="main-title" style="font-size:1.5rem;">🤖 Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Mode selection
        st.markdown("### 🎯 Mode")
        mode = st.radio(
            "Select mode",
            ["Single Model", "Compare (Side-by-Side)"],
            label_visibility="collapsed",
            key="mode_radio",
        )
        st.session_state.mode = "compare" if "Compare" in mode else "single"

        if st.session_state.mode == "single":
            st.markdown("### 🧠 Model")
            model = st.radio(
                "Select model",
                ["OSS — Qwen 2.5", "Frontier — Gemini / OpenAI"],
                label_visibility="collapsed",
                key="model_radio",
            )
            st.session_state.active_model = "oss" if "OSS" in model else "frontier"

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Settings
        st.markdown("### ⚙️ Settings")
        config = st.session_state.config
        config.enable_guardrails = st.toggle("🛡️ Safety Guardrails", value=config.enable_guardrails)
        config.max_memory_turns = st.slider("💾 Memory Window", 5, 50, config.max_memory_turns)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Actions
        st.markdown("### 🔧 Actions")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages_oss = []
            st.session_state.messages_frontier = []
            if st.session_state.oss_assistant:
                st.session_state.oss_assistant.reset()
            if st.session_state.frontier_assistant:
                st.session_state.frontier_assistant.reset()
            st.rerun()

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Metrics
        st.markdown("### 📊 Session Metrics")
        metrics = st.session_state.logger.get_metrics()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Requests", metrics.get("total_requests", 0))
        with col2:
            st.metric("Avg Latency", f"{metrics.get('avg_latency_ms', 0):.0f}ms")

        col3, col4 = st.columns(2)
        with col3:
            st.metric("🛡️ Blocks", metrics.get("safety_blocks", 0))
        with col4:
            st.metric("🔧 Tools", metrics.get("tool_uses", 0))


# ── Chat Interface ───────────────────────────────────────────────
def render_single_chat():
    """Render single-model chat interface."""
    model_type = st.session_state.active_model
    model_name = "OSS — Qwen 2.5" if model_type == "oss" else "Frontier"
    messages_key = f"messages_{model_type}"

    st.markdown(f"**Active Model:** `{model_name}`")

    # Display messages
    for msg in st.session_state[messages_key]:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input(f"Message {model_name}..."):
        # Add user message
        st.session_state[messages_key].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                assistant = get_assistant(model_type)
                response = assistant.chat(prompt)
            st.markdown(response)

        st.session_state[messages_key].append({"role": "assistant", "content": response})


def render_compare_chat():
    """Render side-by-side comparison chat."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟣 OSS — Qwen 2.5")
        for msg in st.session_state.messages_oss:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🟣"):
                st.markdown(msg["content"])

    with col2:
        st.markdown("#### 🔵 Frontier")
        for msg in st.session_state.messages_frontier:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🔵"):
                st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Message both models..."):
        st.session_state.messages_oss.append({"role": "user", "content": prompt})
        st.session_state.messages_frontier.append({"role": "user", "content": prompt})

        # Get responses from both
        oss = get_assistant("oss")
        frontier = get_assistant("frontier")

        oss_response = oss.chat(prompt)
        frontier_response = frontier.chat(prompt)

        st.session_state.messages_oss.append({"role": "assistant", "content": oss_response})
        st.session_state.messages_frontier.append({"role": "assistant", "content": frontier_response})

        st.rerun()


# ── Main App ─────────────────────────────────────────────────────
def main():
    init_session()
    render_sidebar()

    # Header
    st.markdown('<div class="main-title">AI Personal Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Compare Open Source vs Frontier AI Models — with Memory, Tools & Safety</div>', unsafe_allow_html=True)

    # Render appropriate mode
    if st.session_state.mode == "compare":
        render_compare_chat()
    else:
        render_single_chat()


if __name__ == "__main__":
    main()
