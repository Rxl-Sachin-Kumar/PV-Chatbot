"""
app/streamlit_app.py — PharmaCog Streamlit UI

Features
--------
• Persistent multi-conversation history (SQLite)
• ChatGPT-style sidebar with search, switch, delete
• Streaming token-by-token responses
• Adjustable top-k retrieval slider
• Source citations with book page + FAERS drug/reaction
• Intent badge + model tag on every assistant message
"""
from __future__ import annotations

import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import streamlit as st

# ── DB boot — must happen before any DB call ──────────────────────────────────
from src.db.sqlite_db import (
    create_conversation,
    delete_conversation,
    init_db,
    load_conversations,
    load_messages,
    save_message,
    search_conversations,
    update_conversation_title,
)

init_db()   # idempotent — safe to call every run

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaCog — PV Chatbot",
    page_icon="💊",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Source cards ── */
.source-card {
    background: #0f1626;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin: 0.3rem 0;
    border-left: 3px solid #f7a34f;
    font-size: 0.81rem;
    color: #a0b4d0;
}
.source-card.book { border-left-color: #4fc9f7; }
.source-meta { font-size: 0.68rem; color: #3a5070; margin-top: 0.2rem; }
.badge {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.68rem;
    margin-right: 3px;
}
.badge-book  { background: #0a2030; color: #4fc9f7; }
.badge-faers { background: #220f00; color: #f7a34f; }
.badge-page  { background: #0a1f0a; color: #4fd070; }

/* ── Sidebar chat item ── */
div[data-testid="stSidebarContent"] .stButton button {
    text-align: left !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE BOOTSTRAP
# ═════════════════════════════════════════════════════════════════════════════

def _ensure_session() -> None:
    """
    Called once per run.  Initialises session state from DB on first load.
    """
    # Chatbot singleton
    if "chatbot" not in st.session_state:
        from app.chatbot import PVChatbot
        st.session_state.chatbot = PVChatbot(top_k=6)

    # Settings
    if "use_streaming" not in st.session_state:
        st.session_state.use_streaming = True
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "top_k" not in st.session_state:
        st.session_state.top_k = 6
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    # Active conversation
    if "current_session_id" not in st.session_state:
        convs = load_conversations()
        if convs:
            # Resume latest conversation
            st.session_state.current_session_id = convs[0]["id"]
        else:
            # First ever launch — create a conversation
            cid = create_conversation()
            st.session_state.current_session_id = cid

    # Message cache (avoid re-querying DB on every rerun)
    if "messages_cache" not in st.session_state:
        st.session_state.messages_cache = {}

_ensure_session()


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def current_sid() -> str:
    return st.session_state.current_session_id


def get_messages(sid: str | None = None) -> list[dict]:
    """Return messages for sid, using an in-process cache."""
    sid = sid or current_sid()
    if sid not in st.session_state.messages_cache:
        st.session_state.messages_cache[sid] = load_messages(sid)
    return st.session_state.messages_cache[sid]


def push_message(sid: str, msg: dict) -> None:
    """Append to cache and persist to DB."""
    cache = st.session_state.messages_cache.setdefault(sid, [])
    cache.append(msg)
    save_message(
        conversation_id = sid,
        role            = msg["role"],
        content         = msg["content"],
        sources         = msg.get("sources"),
        model           = msg.get("model", ""),
        intent          = msg.get("intent", ""),
    )


def switch_conversation(sid: str) -> None:
    st.session_state.current_session_id = sid
    st.session_state.chatbot.history = []
    st.session_state.chatbot.turn    = 0


def new_conversation() -> str:
    cid = create_conversation()
    st.session_state.current_session_id = cid
    st.session_state.messages_cache[cid] = []
    st.session_state.chatbot.reset()
    return cid


def remove_conversation(sid: str) -> None:
    delete_conversation(sid)
    st.session_state.messages_cache.pop(sid, None)
    if current_sid() == sid:
        convs = load_conversations()
        if convs:
            switch_conversation(convs[0]["id"])
        else:
            new_conversation()


def auto_title(sid: str, text: str) -> None:
    """Set conversation title from first user message (once)."""
    convs = load_conversations()
    for c in convs:
        if c["id"] == sid and c["title"] == "New conversation":
            title = text[:60] + ("…" if len(text) > 60 else "")
            update_conversation_title(sid, title)
            break


def fmt_time(ts: float) -> str:
    dt  = datetime.fromtimestamp(ts)
    now = datetime.now()
    if dt.date() == now.date():
        return dt.strftime("Today %H:%M")
    if (now - dt).days == 1:
        return dt.strftime("Yesterday %H:%M")
    return dt.strftime("%d %b")


# ── Source citation renderer ──────────────────────────────────────────────────

def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} sources", expanded=False):
        for i, s in enumerate(sources, 1):
            is_book  = s.get("source") == "book"
            file_    = s.get("file", "")
            page     = s.get("page", "")
            drug     = s.get("drug", "")
            reaction = s.get("reaction", "")
            score    = s.get("score", 0.0)
            text     = s.get("text", "")[:240]

            badges = ""
            if is_book:
                badges += f'<span class="badge badge-book">📖 {file_}</span>'
                if page:
                    badges += f'<span class="badge badge-page">pg.{page}</span>'
            else:
                if drug:
                    badges += f'<span class="badge badge-faers">💊 {drug}</span>'
                if reaction:
                    badges += f'<span class="badge badge-faers">⚠️ {reaction}</span>'

            card_cls = "source-card book" if is_book else "source-card"
            st.markdown(
                f'<div class="{card_cls}">'
                f'<b>#{i} {"📖 BOOK" if is_book else "📋 FAERS"}</b>'
                f'&nbsp;{badges}<br>'
                f'<span style="color:#7080a0">{text}{"…" if len(s.get("text",""))>240 else ""}</span>'
                f'<div class="source-meta">relevance {score:.4f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Render one message bubble ─────────────────────────────────────────────────

def render_message(msg: dict) -> None:
    role = msg["role"]
    with st.chat_message(role, avatar="👤" if role == "user" else "💊"):
        st.markdown(msg["content"])
        if role == "assistant":
            parts = []
            if msg.get("model"):
                parts.append(f"model: {msg['model'].split(':')[0]}")
            if msg.get("intent"):
                parts.append(f"intent: {msg['intent']}")
            if parts:
                st.caption("  ·  ".join(parts))
            if st.session_state.show_sources and msg.get("sources"):
                render_sources(msg["sources"])


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 💊 PharmaCog")

    # ── New chat ──────────────────────────────────────────────────────────────
    if st.button("➕  New Chat", use_container_width=True, type="primary"):
        new_conversation()
        st.rerun()

    st.divider()

    # ── Search ────────────────────────────────────────────────────────────────
    search = st.text_input(
        "🔍 Search chats",
        value=st.session_state.search_query,
        placeholder="Search message content…",
        label_visibility="collapsed",
    )
    st.session_state.search_query = search

    # ── Conversation list ─────────────────────────────────────────────────────
    if search.strip():
        conv_list = search_conversations(search.strip())
        if not conv_list:
            st.caption("No results found.")
    else:
        conv_list = load_conversations()

    for conv in conv_list:
        sid        = conv["id"]
        title      = conv["title"]
        updated_at = conv["updated_at"]
        n_msgs     = conv.get("message_count", 0)
        is_active  = sid == current_sid()

        col_btn, col_del = st.columns([5, 1])

        with col_btn:
            label = f"{'▌ ' if is_active else ''}{title}"
            if st.button(
                label,
                key=f"conv_{sid}",
                use_container_width=True,
                help=f"{fmt_time(updated_at)} · {n_msgs} messages",
            ):
                if not is_active:
                    switch_conversation(sid)
                    st.rerun()

            st.caption(f"{fmt_time(updated_at)}  ·  {n_msgs} msg")

        with col_del:
            if st.button("🗑", key=f"del_{sid}", help="Delete"):
                remove_conversation(sid)
                st.rerun()

    st.divider()

    # ── Settings ─────────────────────────────────────────────────────────────
    st.markdown("**Settings**")

    top_k = st.slider("Top-K retrieval", 1, 20, st.session_state.top_k)
    if top_k != st.session_state.top_k:
        st.session_state.top_k = top_k
        st.session_state.chatbot.top_k = top_k

    st.session_state.use_streaming = st.toggle(
        "⚡ Streaming", value=st.session_state.use_streaming
    )
    st.session_state.show_sources = st.toggle(
        "📚 Citations", value=st.session_state.show_sources
    )

    st.divider()

    # ── Example queries ───────────────────────────────────────────────────────
    st.markdown("**Examples**")
    for ex in [
        "What is pharmacovigilance?",
        "Side effects of aspirin",
        "Adverse events for ibuprofen",
        "Naranjo algorithm explained",
        "Signal detection methods",
        "ICH E2A guidelines",
    ]:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:18]}"):
            st.session_state["pending"] = ex
            st.rerun()

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.divider()
    all_convs = load_conversations()
    st.caption(f"💬 {len(all_convs)} conversations stored")
    st.caption(f"Session: `{current_sid()[:8]}…`")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═════════════════════════════════════════════════════════════════════════════

# ── Active conversation header ────────────────────────────────────────────────
convs = load_conversations()
active_title = next(
    (c["title"] for c in convs if c["id"] == current_sid()),
    "New conversation"
)

st.title("💊 PharmaCog")
st.caption(
    f"Pharmacovigilance AI · AWS Bedrock · "
    f"**{active_title}**  ·  Top-K: {st.session_state.top_k}"
)
st.divider()

# ── Render conversation messages ──────────────────────────────────────────────
messages = get_messages()

if not messages:
    st.markdown(
        "<div style='text-align:center;color:#2a4060;padding:4rem 0'>"
        "<div style='font-size:3rem'>💊</div>"
        "<div style='font-size:1.1rem;margin-top:0.6rem;color:#4a7090'>"
        "Ask anything about drug safety, ADRs or PV guidelines</div>"
        "<div style='font-size:0.8rem;margin-top:0.3rem;color:#1a3050'>"
        "Powered by AWS Bedrock + FAERS + PV Literature</div>"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    for msg in messages:
        render_message(msg)

# ── Chat input ────────────────────────────────────────────────────────────────
pending    = st.session_state.pop("pending", None)
user_input = st.chat_input("Ask about drug safety, ADRs, PV guidelines…")
query      = (pending or user_input or "").strip()

# ═════════════════════════════════════════════════════════════════════════════
# QUERY HANDLING
# ═════════════════════════════════════════════════════════════════════════════

if query:
    sid = current_sid()

    # Auto-title from first message
    auto_title(sid, query)

    # ── Show user bubble ──────────────────────────────────────────────────────
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    # ── Generate assistant response ───────────────────────────────────────────
    full_answer = ""
    sources     = []
    intent      = ""
    model_used  = ""

    with st.chat_message("assistant", avatar="💊"):
        try:
            if st.session_state.use_streaming:
                # ── Streaming path ────────────────────────────────────────────
                token_iter, meta = st.session_state.chatbot.stream_chat(query)

                placeholder = st.empty()
                streamed    = ""

                for token in token_iter:
                    streamed += token
                    placeholder.markdown(streamed + "▌")   # typing cursor

                placeholder.markdown(streamed)             # final — no cursor

                full_answer = streamed
                sources     = meta.sources
                intent      = meta.intent.value
                model_used  = meta.model_used

            else:
                # ── Non-streaming path ────────────────────────────────────────
                with st.spinner("Thinking…"):
                    response = st.session_state.chatbot.chat(query)

                st.markdown(response.answer)
                full_answer = response.answer
                sources     = response.sources
                intent      = response.intent.value
                model_used  = response.model_used

            # ── Caption + citations ───────────────────────────────────────────
            parts = []
            if model_used:
                parts.append(f"model: {model_used.split(':')[0]}")
            if intent:
                parts.append(f"intent: {intent}")
            if parts:
                st.caption("  ·  ".join(parts))

            if st.session_state.show_sources and sources:
                render_sources(sources)

        except FileNotFoundError:
            st.error(
                "Vector store not found.  "
                "Run `python scripts/build_index.py` to build the index first."
            )
            st.stop()

        except RuntimeError as exc:
            st.error(
                f"**Bedrock error:** {exc}\n\n"
                "Check:\n"
                "- `aws configure` — valid credentials\n"
                "- Bedrock console — model access enabled\n"
                "- `AWS_REGION` in `.env` matches enabled region"
            )
            st.stop()

        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            st.stop()

    # ── Persist both messages to SQLite ──────────────────────────────────────
    push_message(sid, {"role": "user", "content": query})
    push_message(sid, {
        "role":    "assistant",
        "content": full_answer,
        "sources": sources,
        "model":   model_used,
        "intent":  intent,
    })

    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "PharmaCog · MiniLM-L6-v2 embeddings · FAISS IndexFlatIP · "
    "BM25 + Dense + CrossEncoder retrieval · "
    "AWS Bedrock Nova Pro → Lite → Claude fallback"
)