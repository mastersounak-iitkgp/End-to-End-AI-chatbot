"""
Streamlit Frontend — Persistent Multi-Session LangGraph & Mistral AI Chatbot with Tools
======================================================================================
Run with: streamlit run app.py

Architecture:
  • LangGraph workflow           →  START -> chat -> [tools_condition] -> tools -> chat -> END
  • Tools                        →  DuckDuckGo Search, Math Calculator, Stock Price Checker
  • Message history persistence  →  chatbot_memory.db  (SQLite, survives restarts)
  • Conversation metadata        →  conversations_meta.json  (titles, system prompts, timestamps)
  • Rendering                    →  Native Streamlit chat primitives (st.chat_message, st.status)
"""

import json
import uuid
import os
from datetime import datetime

import streamlit as st
from langchain_core.messages import HumanMessage

from chatbot_backend import create_chatbot_app, stream_chat_response
from rag_engine import (
    index_pdf_document,
    get_thread_doc_info,
    clear_thread_index,
    compute_file_hash,
)

# ── Constants ──────────────────────────────────────────────────────────────────
_DIR          = os.path.dirname(os.path.abspath(__file__))
DB_PATH       = os.path.join(_DIR, "chatbot_memory.db")
META_PATH     = os.path.join(_DIR, "conversations_meta.json")
MAX_TITLE_LEN = 44          # chars before truncation in title

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, accurate, and articulate AI assistant equipped with 4 specialized tools:\n"
    "1. Document Search (search_uploaded_documents): to search and answer questions based on the user's uploaded PDF document.\n"
    "2. DuckDuckGo Web Search: for current events, news, live sports scores, and real-time facts.\n"
    "3. Calculator: for reliable and precise mathematical calculations and functions.\n"
    "4. Stock Price Checker: for live stock/crypto market prices and company valuation metrics.\n"
    "Rules:\n"
    "- If the user asks about an uploaded document, contract, or PDF, ALWAYS call `search_uploaded_documents`.\n"
    "- When answering from a document, cite the source document name and page numbers.\n"
    "- Call tools when needed to get accurate, real-time facts or exact math.\n"
    "- Format all your responses using clean Markdown (e.g. bold, lists, links, tables, code blocks). Do NOT output raw HTML tags.\n"
    "- If search or stock results are unavailable, provide a safe, polite, and helpful answer based on your general knowledge without throwing an error."
)

TOOL_META = {
    "search_uploaded_documents": {"icon": "📄", "label": "Document Search (RAG)"},
    "duckduckgo_search":         {"icon": "🌐", "label": "DuckDuckGo Web Search"},
    "calculate_math":            {"icon": "🧮", "label": "Math Calculator"},
    "check_stock_price":         {"icon": "📈", "label": "Stock Price Checker"},
}


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Chatbot with Tools",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Global background ──────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar shell ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(10, 8, 30, 0.94);
    border-right: 1px solid rgba(139, 92, 246, 0.2);
    backdrop-filter: blur(16px);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Sidebar app title */
.sb-title {
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sb-sub {
    font-size: 0.72rem;
    color: #94a3b8 !important;
    margin-top: 0.15rem;
    margin-bottom: 0.9rem;
}

/* Sidebar labels */
.sb-label {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94a3b8 !important;
    font-weight: 600;
    margin: 0.8rem 0 0.35rem 0;
}

/* ── Tools Card in Sidebar ───────────────────────────────────────── */
.tools-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 12px;
    padding: 0.75rem;
    margin-bottom: 0.8rem;
}
.tool-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.35rem 0;
    font-size: 0.78rem;
    color: #cbd5e1 !important;
}
.tool-item span.icon {
    font-size: 0.92rem;
}
.tool-item span.name {
    font-weight: 600;
    color: #a78bfa !important;
}

/* ── New Chat button (type=primary) ─────────────────────────────── */
[data-testid="stSidebar"] [data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 0.5rem 1rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 2px 12px rgba(124, 58, 237, 0.35) !important;
}
[data-testid="stSidebar"] [data-testid="baseButton-primary"]:hover {
    opacity: 0.87 !important;
    transform: translateY(-1px) !important;
}

/* ── Conversation list buttons (type=secondary) ─────────────────── */
[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
    background: transparent !important;
    color: #94a3b8 !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    text-align: left !important;
    padding: 0.42rem 0.75rem !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    width: 100% !important;
    transition: all 0.15s !important;
    box-shadow: none !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    margin-bottom: 2px !important;
}
[data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {
    background: rgba(139, 92, 246, 0.14) !important;
    color: #e2e8f0 !important;
    border-color: rgba(139, 92, 246, 0.3) !important;
}

/* ── Text area (system prompt editor) ──────────────────────────── */
[data-testid="stSidebar"] .stTextArea textarea {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    line-height: 1.45 !important;
}

/* Read-only system prompt box */
.sp-readonly {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    font-size: 0.77rem;
    color: #94a3b8 !important;
    line-height: 1.45;
    word-break: break-word;
    margin-bottom: 0.4rem;
}

/* ── Header card ────────────────────────────────────────────────── */
.header-card {
    background: linear-gradient(135deg, rgba(139,92,246,0.18), rgba(59,130,246,0.18));
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 18px;
    padding: 1.25rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
    text-align: center;
}
.header-card h1 {
    font-size: 1.55rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.25rem 0;
}
.header-card .conv-title-display {
    color: #94a3b8;
    font-size: 0.82rem;
    font-weight: 500;
}

/* ── Style Native Streamlit Chat Messages ───────────────────────── */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.9rem;
    backdrop-filter: blur(8px);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.22), rgba(37, 99, 235, 0.22));
    border-color: rgba(139, 92, 246, 0.35);
}

/* ── Style Expanders & Status Widgets ───────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(15, 12, 41, 0.45) !important;
    border: 1px solid rgba(139, 92, 246, 0.22) !important;
    border-radius: 12px !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stStatusWidget"] {
    background: rgba(15, 12, 41, 0.55) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 12px !important;
    margin-bottom: 0.6rem !important;
}

/* ── Chat input ─────────────────────────────────────────────────── */
.stChatInputContainer {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(139, 92, 246, 0.4) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px) !important;
}
.stChatInputContainer textarea {
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stChatInputContainer button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    border-radius: 10px !important;
}

/* ── Dividers ───────────────────────────────────────────────────── */
hr { border-color: rgba(255, 255, 255, 0.08) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Persistence helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_meta() -> dict:
    """Load conversation metadata dict from disk."""
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_meta() -> None:
    """Write st.session_state.conversations_meta to disk."""
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(st.session_state.conversations_meta, f, indent=2, ensure_ascii=False)


def load_display_messages(thread_id: str) -> list:
    """
    Reconstruct the UI message list from LangGraph's SQLite state.
    Preserves tool calls, tool outputs, and responses across restarts.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = st.session_state.app.get_state(config)
        raw   = state.values.get("messages", [])
    except Exception:
        return []

    result = []
    current_tools = []
    for m in raw:
        if m.type == "system":
            continue
        elif m.type == "human":
            result.append({"role": "user", "content": m.content})
            current_tools = []
        elif m.type == "ai":
            if getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    current_tools.append({
                        "name": tc.get("name"),
                        "args": tc.get("args", {}),
                        "id": tc.get("id"),
                        "output": "",
                    })
            if m.content:
                content = m.content
                if isinstance(content, list):
                    content = "".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                result.append({
                    "role": "assistant",
                    "content": content,
                    "tools_used": list(current_tools),
                })
                current_tools = []
        elif m.type == "tool":
            for t in current_tools:
                if t.get("id") == getattr(m, "tool_call_id", None) or t.get("name") == getattr(m, "name", None):
                    t["output"] = str(m.content)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Conversation management
# ═══════════════════════════════════════════════════════════════════════════════

def new_conversation(system_prompt: str | None = None) -> str:
    """Create a metadata entry for a brand-new conversation."""
    thread_id = str(uuid.uuid4())
    st.session_state.conversations_meta[thread_id] = {
        "id":            thread_id,
        "title":         "New Chat",
        "created_at":    datetime.now().isoformat(),
        "system_prompt": system_prompt or DEFAULT_SYSTEM_PROMPT,
    }
    save_meta()
    return thread_id


def switch_to(thread_id: str) -> None:
    """Activate a conversation and reload history from SQLite state."""
    st.session_state.active_thread_id  = thread_id
    st.session_state.display_messages  = load_display_messages(thread_id)
    st.session_state.first_turn        = (len(st.session_state.display_messages) == 0)


def set_title_from_message(thread_id: str, first_message: str) -> None:
    """Auto-title a conversation based on the first user message."""
    title = first_message[:MAX_TITLE_LEN].rstrip()
    if len(first_message) > MAX_TITLE_LEN:
        title += "…"
    st.session_state.conversations_meta[thread_id]["title"] = title
    save_meta()


# ═══════════════════════════════════════════════════════════════════════════════
# Session state initialization (runs once per Streamlit session)
# ═══════════════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    if "app" not in st.session_state:
        st.session_state.app = create_chatbot_app(DB_PATH)

    if "conversations_meta" not in st.session_state:
        st.session_state.conversations_meta = load_meta()

    if "active_thread_id" not in st.session_state:
        meta = st.session_state.conversations_meta
        if meta:
            latest = max(meta.keys(), key=lambda k: meta[k].get("created_at", ""))
            switch_to(latest)
        else:
            tid = new_conversation()
            switch_to(tid)

    if "display_messages" not in st.session_state:
        st.session_state.display_messages = load_display_messages(
            st.session_state.active_thread_id
        )
    if "first_turn" not in st.session_state:
        st.session_state.first_turn = (len(st.session_state.display_messages) == 0)

    if "indexed_hashes" not in st.session_state:
        st.session_state.indexed_hashes = {}


_init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── App brand ────────────────────────────────────────────────────────
    st.markdown('<div class="sb-title">🤖 AI Chatbot with Tools</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-sub">Mistral AI · LangGraph ToolNode · SQLite Persistence</div>',
        unsafe_allow_html=True,
    )

    # ── Available Tools Panel ─────────────────────────────────────────────
    st.markdown('<p class="sb-label">Active Capabilities</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tools-card">
        <div class="tool-item">
            <span class="icon">📄</span>
            <span class="name">Document RAG</span> — Query uploaded PDFs up to 200MB
        </div>
        <div class="tool-item">
            <span class="icon">🌐</span>
            <span class="name">Web Search</span> — DuckDuckGo live facts & scores
        </div>
        <div class="tool-item">
            <span class="icon">🧮</span>
            <span class="name">Calculator</span> — Reliable AST math & formulas
        </div>
        <div class="tool-item">
            <span class="icon">📈</span>
            <span class="name">Stock Price</span> — Free quotes & market data
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── New Chat ─────────────────────────────────────────────────────────
    if st.button("＋  New Chat", use_container_width=True, type="primary", key="new_chat_btn"):
        tid = new_conversation()
        switch_to(tid)
        st.rerun()

    # ── Per-Session Document Upload (RAG) ─────────────────────────────────
    active_tid = st.session_state.active_thread_id
    doc_info = get_thread_doc_info(active_tid)

    st.markdown('<p class="sb-label">Document Context (RAG)</p>', unsafe_allow_html=True)

    if doc_info:
        st.markdown(
            f"""
            <div style="background:rgba(124,58,237,0.18); border:1px solid rgba(139,92,246,0.35); border-radius:10px; padding:0.55rem 0.75rem; margin-bottom:0.5rem;">
                <div style="font-size:0.8rem; font-weight:600; color:#c4b5fd;">📄 {doc_info['file_name']}</div>
                <div style="font-size:0.71rem; color:#94a3b8; margin-top:2px;">
                    {doc_info['total_pages']} pages · {doc_info['chunk_count']} chunks indexed
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Remove Document", key=f"clear_doc_{active_tid}", use_container_width=True):
            clear_thread_index(active_tid)
            if active_tid in st.session_state.indexed_hashes:
                del st.session_state.indexed_hashes[active_tid]
            st.rerun()
    else:
        uploaded_file = st.file_uploader(
            "Upload PDF (up to 200MB)",
            type=["pdf"],
            key=f"uploader_{active_tid}",
            label_visibility="collapsed",
            help="Upload any PDF up to 200MB to ask questions about it in this chat.",
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_hash = compute_file_hash(uploaded_file.name, uploaded_file.size, file_bytes)

            if st.session_state.indexed_hashes.get(active_tid) != file_hash:
                with st.status(f"Indexing '{uploaded_file.name}'...", expanded=True) as status:
                    try:
                        st.write("📖 Extracting pages using PyMuPDF...")
                        doc_meta = index_pdf_document(file_bytes, uploaded_file.name, active_tid)
                        st.write(f"✂️ Created {doc_meta['chunk_count']} semantic chunks...")
                        st.write("🧠 Generating local CPU embeddings & building hybrid index...")
                        status.update(
                            label=f"✓ '{uploaded_file.name}' indexed successfully!",
                            state="complete",
                            expanded=False,
                        )
                        st.session_state.indexed_hashes[active_tid] = file_hash
                        st.rerun()
                    except Exception as exc:
                        status.update(label=f"❌ Indexing failed: {exc}", state="error", expanded=True)

    st.markdown("---")

    # ── Conversation list ────────────────────────────────────────────────
    st.markdown('<p class="sb-label">Conversations</p>', unsafe_allow_html=True)

    sorted_convs = sorted(
        st.session_state.conversations_meta.values(),
        key=lambda c: c.get("created_at", ""),
        reverse=True,   # newest first
    )

    if not sorted_convs:
        st.caption("No conversations yet.")

    for conv in sorted_convs:
        tid       = conv["id"]
        title     = conv.get("title", "New Chat")
        is_active = (tid == st.session_state.active_thread_id)

        display_title = title if len(title) <= 32 else title[:32] + "…"
        label = ("● " if is_active else "   ") + display_title

        if st.button(label, key=f"conv_{tid}", use_container_width=True):
            if not is_active:
                switch_to(tid)
                st.rerun()

    st.markdown("---")

    # ── Per-conversation system prompt ───────────────────────────────────
    active_meta = st.session_state.conversations_meta.get(
        st.session_state.active_thread_id, {}
    )
    has_messages = len(st.session_state.display_messages) > 0

    st.markdown('<p class="sb-label">System Prompt</p>', unsafe_allow_html=True)

    if has_messages:
        sp = active_meta.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        st.markdown(f'<div class="sp-readonly">{sp}</div>', unsafe_allow_html=True)
        st.caption("🔒 Fixed for this session. Use '＋ New Chat' to set a new prompt.")
    else:
        new_sp = st.text_area(
            "System Prompt",
            value=active_meta.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            height=120,
            key="sp_editor",
            label_visibility="collapsed",
            placeholder="Describe how the assistant should behave…",
        )
        if new_sp != active_meta.get("system_prompt"):
            st.session_state.conversations_meta[st.session_state.active_thread_id]["system_prompt"] = new_sp
            save_meta()

    st.markdown("---")
    st.markdown(
        f'<p style="color:#64748b;font-size:0.72rem;text-align:center;">'
        f'{len(st.session_state.conversations_meta)} conversation(s) stored</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main Area — Header
# ═══════════════════════════════════════════════════════════════════════════════

active_meta  = st.session_state.conversations_meta.get(st.session_state.active_thread_id, {})
conv_title   = active_meta.get("title", "New Chat")
conv_created = active_meta.get("created_at", "")[:10]

st.markdown(
    f"""
    <div class="header-card">
        <h1>🤖 AI Chatbot with Tools</h1>
        <div class="conv-title-display">
            {conv_title} {" · " + conv_created if conv_created else ""}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Message History Rendering (Native Streamlit Chat)
# ═══════════════════════════════════════════════════════════════════════════════

if not st.session_state.display_messages:
    st.markdown(
        """
        <div style="text-align:center; padding:2rem 1rem;">
            <div style="font-size:2.8rem; margin-bottom:0.6rem;">💬</div>
            <p style="font-size:1.05rem; font-weight:600; color:#e2e8f0; margin-bottom: 0.3rem;">
                Ask anything or try a tool!
            </p>
            <p style="font-size:0.82rem; color:#94a3b8; max-width: 480px; margin: 0 auto 1.3rem auto;">
                Equipped with PDF document RAG (up to 200MB), live DuckDuckGo web search, exact math calculation, and real-time stock prices.
            </p>
            <div style="display:flex; justify-content:center; gap:8px; flex-wrap:wrap; font-size:0.78rem; color:#a78bfa;">
                <span style="background:rgba(139,92,246,0.15); border:1px solid rgba(139,92,246,0.25); border-radius:12px; padding:0.35rem 0.75rem;">
                    📄 "What does the uploaded PDF say about ...?"
                </span>
                <span style="background:rgba(139,92,246,0.15); border:1px solid rgba(139,92,246,0.25); border-radius:12px; padding:0.35rem 0.75rem;">
                    🌐 "How much runs did India scored today"
                </span>
                <span style="background:rgba(139,92,246,0.15); border:1px solid rgba(139,92,246,0.25); border-radius:12px; padding:0.35rem 0.75rem;">
                    📈 "What is the stock price of Apple (AAPL)?"
                </span>
                <span style="background:rgba(139,92,246,0.15); border:1px solid rgba(139,92,246,0.25); border-radius:12px; padding:0.35rem 0.75rem;">
                    🧮 "Calculate sqrt(144) * 15 + 2^8"
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.display_messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                # Render collapsible expanders for tools used in this turn
                if msg.get("tools_used"):
                    for t in msg["tools_used"]:
                        t_name = t.get("name", "tool")
                        meta = TOOL_META.get(t_name, {"icon": "⚙️", "label": t_name})
                        with st.expander(f"{meta['icon']} Used **{meta['label']}**", expanded=False):
                            if t.get("args"):
                                st.caption(f"**Arguments:** `{json.dumps(t['args'])}`")
                            if t.get("output"):
                                st.markdown("**Result:**")
                                st.code(t["output"], language="text")
                # Render the clean markdown response
                st.markdown(msg["content"])


# ═══════════════════════════════════════════════════════════════════════════════
# Chat Input & Real-Time Streaming
# ═══════════════════════════════════════════════════════════════════════════════

user_input = st.chat_input("Ask a question, search live scores, or calculate…", key="chat_input")

if user_input and user_input.strip():
    user_text = user_input.strip()
    thread_id = st.session_state.active_thread_id

    # ── Auto-title on the first message ──────────────────────────────────
    if st.session_state.first_turn:
        set_title_from_message(thread_id, user_text)

    # ── Render user message ───────────────────────────────────────────────
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_text)

    st.session_state.display_messages.append({"role": "user", "content": user_text})

    # ── Build LangGraph payload ───────────────────────────────────────────
    payload = {"messages": [HumanMessage(content=user_text)]}
    if st.session_state.first_turn:
        payload["system_prompt"] = active_meta.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        st.session_state.first_turn = False

    config = {"configurable": {"thread_id": thread_id}}

    # ── Assistant response with native live status & token streaming ──────
    with st.chat_message("assistant", avatar="🤖"):
        tool_status_container = st.container()
        text_slot = st.empty()
        text_slot.markdown("▌")

        status_widgets = {}     # tool_id -> st.status widget
        tools_executed = []
        full_reply = ""

        try:
            for event in stream_chat_response(st.session_state.app, payload, config):
                ev_type = event.get("type")

                # 1. Tool Call Triggered
                if ev_type == "tool_call":
                    t_name = event.get("name")
                    t_id = event.get("id") or f"tool_{len(tools_executed)}"
                    meta = TOOL_META.get(t_name, {"icon": "⚙️", "label": t_name})

                    with tool_status_container:
                        s_widget = st.status(
                            f"{meta['icon']} Running **{meta['label']}**...",
                            state="running",
                            expanded=True,
                        )
                        if event.get("args"):
                            s_widget.write(f"**Arguments:** `{json.dumps(event.get('args', {}))}`")
                        status_widgets[t_id] = s_widget

                    tools_executed.append({
                        "name": t_name,
                        "args": event.get("args", {}),
                        "id": t_id,
                        "output": "",
                    })

                # 2. Tool Execution Finished
                elif ev_type == "tool_result":
                    t_id = event.get("id")
                    t_name = event.get("name")
                    out_text = event.get("output", "")

                    for tc in tools_executed:
                        if (t_id and tc.get("id") == t_id) or (tc.get("name") == t_name and not tc.get("output")):
                            tc["output"] = out_text
                            matched_id = tc.get("id")
                            if matched_id in status_widgets:
                                meta = TOOL_META.get(tc["name"], {"icon": "⚙️", "label": tc["name"]})
                                w = status_widgets[matched_id]
                                w.write("**Result:**")
                                w.code(out_text, language="text")
                                w.update(
                                    label=f"{meta['icon']} Completed **{meta['label']}**",
                                    state="complete",
                                    expanded=False,
                                )
                            break

                # 3. Text Token Arriving
                elif ev_type == "token":
                    full_reply += event.get("content", "")
                    text_slot.markdown(full_reply + "▌")

            # Finalize reply without cursor
            text_slot.markdown(full_reply if full_reply else "(No response)")

        except Exception as exc:
            full_reply = f"⚠️ Error: {exc}"
            text_slot.markdown(full_reply)

    # Persist to display history
    st.session_state.display_messages.append({
        "role": "assistant",
        "content": full_reply,
        "tools_used": list(tools_executed),
    })

    # Rerun to update sidebar conversation title and stored count
    st.rerun()
