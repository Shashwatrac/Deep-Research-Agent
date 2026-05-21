
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agent.llm import OllamaClient
from agent.loop import run_agent
from agent.session import (
    create_session,
    get_conversation_history,
    get_turn_history,
    init_db,
    list_sessions,
)
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SERPER_API_KEY,
    SESSION_DB,
    TAVILY_API_KEY,
)

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🎯",
    layout="wide",
)
st.markdown("""
<style>

/* Main app background */
.stApp {
    background: #111827;
    color: #f3f4f6;
}

/* Progress box */
.progress-box {
    background: linear-gradient(135deg, #1f2937, #111827);
    border-left: 4px solid #60a5fa;
    padding: 12px 16px;
    border-radius: 10px;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 0.84em;
    color: #dbeafe;
    margin-bottom: 8px;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.6;
    box-shadow: 0 3px 10px rgba(0,0,0,0.18);
    transition: all 0.2s ease;
}

/* Hover effect */
.progress-box:hover {
    transform: translateY(-1px);
    box-shadow: 0 5px 14px rgba(0,0,0,0.24);
}

/* Status colors */
.status-ok {
    color: #4ade80;
    font-weight: 600;
}

.status-err {
    color: #f87171;
    font-weight: 600;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Chat message cards */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    padding: 8px;
    margin-bottom: 10px;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #1e293b;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Buttons */
.stButton button {
    border-radius: 8px;
    border: none;
    transition: all 0.2s ease;
}

.stButton button:hover {
    transform: translateY(-1px);
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)
# st.markdown("""
# <style>
# .progress-box {
#     background: #1a1d2e;
#     border-left: 3px solid #4f8ef7;
#     padding: 10px 14px;
#     border-radius: 5px;
#     font-family: 'Courier New', monospace;
#     font-size: 0.82em;
#     color: #b8c8e8;
#     margin-bottom: 6px;
#     white-space: pre-wrap;
#     word-break: break-word;
#     line-height: 1.5;
# }
# .status-ok  { color: #4caf50; font-weight: bold; }
# .status-err { color: #f44336; font-weight: bold; }
# </style>
# """, unsafe_allow_html=True)

# ── init ──────────────────────────────────────────────────────────────────────
init_db(SESSION_DB)

if "session_id"   not in st.session_state:
    st.session_state.session_id   = create_session(SESSION_DB)
if "messages"     not in st.session_state:
    st.session_state.messages     = []
if "turn_details" not in st.session_state:
    st.session_state.turn_details = []

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 Deep Research Agent")
    st.caption("Local LLM + Tavily web search")
    st.divider()

    # Ollama status check
    st.subheader(" Local LLM (Ollama)")
    ollama_url   = st.text_input("Ollama URL",   value=OLLAMA_BASE_URL)
    ollama_model = st.text_input("Model name",   value=OLLAMA_MODEL,
                                  help="Must be pulled first: ollama pull llama3.2")

    client_check = OllamaClient(ollama_url, ollama_model)
    if client_check.is_available():
        available_models = client_check.list_models()
        if available_models:
            st.success(f"✅ Ollama running — {len(available_models)} model(s) found")
            with st.expander("Available models"):
                for m in available_models:
                    tick = "✅ " if ollama_model in m else "   "
                    st.text(f"{tick}{m}")
        else:
            st.warning("⚠️ Ollama running but no models pulled yet.\nRun: `ollama pull llama3.2`")
    else:
        st.error("❌ Ollama not reachable.\nRun: `ollama serve`")

    st.divider()

    # Search API keys
    st.subheader("🌐 Search API")
    tavily_key = st.text_input("Tavily API Key", value=TAVILY_API_KEY,
                                type="password", placeholder="tvly-...")
    serper_key = st.text_input("Serper API Key (fallback)", value=SERPER_API_KEY,
                                type="password", placeholder="optional")

    if tavily_key:
        st.success("✅ Tavily key set")
    elif serper_key:
        st.info("ℹ️ Using Serper (Tavily preferred)")
    else:
        st.error("❌ Set a search API key")

    st.divider()

    # Session management
    st.subheader("💬 Sessions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New", use_container_width=True):
            st.session_state.session_id   = create_session(SESSION_DB)
            st.session_state.messages     = []
            st.session_state.turn_details = []
            st.rerun()
    with col2:
        if st.button("🔄 Reload", use_container_width=True):
            st.rerun()

    sessions = list_sessions(SESSION_DB)
    if sessions:
        st.caption(f"{len(sessions)} saved session(s)")
        for s in sessions[:8]:
            short_id  = s["session_id"][:8]
            date_str  = s["updated_at"][:10]
            is_active = s["session_id"] == st.session_state.session_id
            label     = f"{'▶ ' if is_active else ''}📁 {short_id}… ({date_str})"
            if st.button(label, key=f"load_{s['session_id']}", use_container_width=True):
                st.session_state.session_id = s["session_id"]
                hist = get_conversation_history(s["session_id"], SESSION_DB)
                st.session_state.messages = [
                    {"role": m["role"], "content": m["content"]} for m in hist
                ]
                st.session_state.turn_details = get_turn_history(s["session_id"], SESSION_DB)
                st.rerun()

# ── main area ─────────────────────────────────────────────────────────────────
col_title, col_meta = st.columns([4, 1])
with col_title:
    st.header("🎯 Deep Research Agent")
    st.caption(
        "Ask anything. The agent will plan search queries, fetch sources, "
        "and produce a cited answer — all running locally."
    )
with col_meta:
    turns_done = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("Session", st.session_state.session_id[:8] + "…")
    st.metric("Turns", turns_done)

st.divider()

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about technology, science, finance, policy, or any research topic..."):

    if not (tavily_key or serper_key):
        st.error("Add a Tavily or Serper API key in the sidebar first.")
        st.stop()

    if not client_check.is_available():
        st.error("Ollama isn't running. Start it with: `ollama serve`")
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        progress_area = st.empty()
        answer_area   = st.empty()
        sources_area  = st.container()

        progress_lines = []
        full_answer    = ""
        final_data     = {}

        for event in run_agent(
            query=prompt,
            session_id=st.session_state.session_id,
            tavily_key=tavily_key,
            serper_key=serper_key,
            ollama_base_url=ollama_url,
            ollama_model=ollama_model,
            db_path=SESSION_DB,
        ):
            etype = event.get("type")

            if etype == "progress":
                progress_lines.append(event["message"])
                display = "\n".join(progress_lines[-14:])
                progress_area.markdown(
                    f'<div class="progress-box">{display}</div>',
                    unsafe_allow_html=True,
                )

            elif etype == "answer_chunk":
                full_answer += event["content"]
                answer_area.markdown(full_answer + "▌")

            elif etype == "done":
                progress_area.empty()
                answer_area.markdown(full_answer)
                final_data = event.get("data", {})

                citations = final_data.get("citations", [])
                if citations:
                    with sources_area:
                        with st.expander(f"📚 {len(citations)} sources used", expanded=False):
                            for c in citations:
                                st.markdown(f"**{c['index']}.** [{c['title']}]({c['url']})  `{c['domain']}`")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Sources found",  final_data.get("sources_found", 0))
                        c2.metric("Pages fetched",  final_data.get("pages_fetched", 0))
                        c3.metric("Search queries", len(final_data.get("search_queries", [])))

            elif etype == "error":
                progress_area.empty()
                st.error(f"⚠️ {event.get('message')}")

        if full_answer:
            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            st.session_state.turn_details = get_turn_history(st.session_state.session_id, SESSION_DB)
