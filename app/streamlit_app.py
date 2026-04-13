"""
app/streamlit_app.py — PharmaCog Streamlit UI
Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaCog — PV Chatbot",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.source-card {
    background: #12172a;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    margin: 0.35rem 0;
    border-left: 3px solid #f7a34f;
    font-size: 0.82rem;
    color: #a0b4d0;
}
.source-card.book { border-left-color: #4fc9f7; }
.source-meta { font-size: 0.72rem; color: #4a5a80; margin-top: 0.25rem; }
.page-badge {
    background: #1a2a4a; color: #4fc9f7;
    border-radius: 4px; padding: 1px 6px; font-size: 0.7rem; margin-right: 4px;
}
.drug-badge {
    background: #2a1a0a; color: #f7a34f;
    border-radius: 4px; padding: 1px 6px; font-size: 0.7rem; margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "chatbot" not in st.session_state:
    from app.chatbot import PVChatbot
    st.session_state.chatbot = PVChatbot(top_k=6)

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role", "content", "sources"?, "intent"?, "model"?}

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True

if "use_streaming" not in st.session_state:
    st.session_state.use_streaming = True


# ── Helper: render source citations ──────────────────────────────────────────
def render_sources(sources: list[dict]) -> None:
    with st.expander(f"📚 {len(sources)} sources", expanded=False):
        for i, src in enumerate(sources, 1):
            stype    = src.get("source", "unknown")
            drug     = src.get("drug", "")
            reaction = src.get("reaction", "")
            file_    = src.get("file", "")
            page     = src.get("page", "")
            score    = src.get("score", 0.0)
            text     = src.get("text", "")[:250]
            is_book  = stype == "book"

            badges = ""
            if is_book:
                if file_:
                    badges += f'<span class="page-badge">📖 {file_}</span>'
                if page:
                    badges += f'<span class="page-badge">pg.{page}</span>'
            else:
                if drug:
                    badges += f'<span class="drug-badge">💊 {drug}</span>'
                if reaction:
                    badges += f'<span class="drug-badge">⚠️ {reaction}</span>'

            card_cls = "source-card book" if is_book else "source-card"
            icon = "📖" if is_book else "📋"

            st.markdown(
                f'<div class="{card_cls}">'
                f'<b>#{i} {icon} {"BOOK" if is_book else "FAERS"}</b> &nbsp;{badges}<br>'
                f'<span style="color:#8090b0">{text}{"…" if len(src.get("text",""))>250 else ""}</span>'
                f'<div class="source-meta">score: {score:.4f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    new_top_k = st.slider("Top-k chunks", 3, 15, 6)
    if new_top_k != st.session_state.chatbot.top_k:
        st.session_state.chatbot.top_k = new_top_k

    st.session_state.use_streaming = st.toggle("⚡ Streaming", value=st.session_state.use_streaming)
    st.session_state.show_sources  = st.toggle("📚 Show citations", value=st.session_state.show_sources)

    st.divider()
    bot = st.session_state.chatbot
    st.caption(f"Session: `{bot.session_id[:8]}…`")
    st.caption(f"Turn: {bot.turn}  |  History: {len(bot.history)}")

    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.chatbot.reset()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Examples**")
    for ex in [
        "What is pharmacovigilance?",
        "Side effects of aspirin",
        "Adverse events for ibuprofen in FAERS",
        "What is the Naranjo algorithm?",
        "Explain signal detection methods",
        "What are ICH E2A guidelines?",
    ]:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state["pending_query"] = ex
            st.rerun()


# ── Main header ───────────────────────────────────────────────────────────────
st.title("💊 PharmaCog")
st.caption("Pharmacovigilance AI · cites book page numbers & FAERS records")
st.divider()

# ── Render full conversation history ─────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role, avatar="👤" if role == "user" else "💊"):
        st.markdown(msg["content"])
        if role == "user" and msg.get("intent"):
            st.caption(f"Intent: {msg['intent']}")
        if role == "assistant":
            if msg.get("model"):
                st.caption(f"Model: {msg['model']}  |  {msg.get('latency_ms', 0):.0f}ms")
            if st.session_state.show_sources and msg.get("sources"):
                render_sources(msg["sources"])

# ── Chat input ────────────────────────────────────────────────────────────────
pending    = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Ask about drug safety, ADRs, PV guidelines…")
query      = (pending or user_input or "").strip()

# ── Handle query ──────────────────────────────────────────────────────────────
if query:
    # 1. Show user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    # 2. Generate response
    with st.chat_message("assistant", avatar="💊"):
        try:
            if st.session_state.use_streaming:
                token_iter, meta = st.session_state.chatbot.stream_chat(query)
                full_answer = st.write_stream(token_iter)
                sources     = meta.sources
                intent      = meta.intent.value
                model_short = meta.model_used.split(".")[-1]
                latency_ms  = 0
            else:
                with st.spinner("Thinking…"):
                    response = st.session_state.chatbot.chat(query)
                st.markdown(response.answer)
                full_answer = response.answer
                sources     = response.sources
                intent      = response.intent.value
                model_short = response.model_used.split(".")[-1]
                latency_ms  = response.latency_ms

            st.caption(f"Model: {model_short}" + (f"  |  {latency_ms:.0f}ms" if latency_ms else ""))

            if st.session_state.show_sources and sources:
                render_sources(sources)

        except FileNotFoundError:
            st.error("Vector store not found. Run `python scripts/build_index.py` first.")
            st.stop()
        except RuntimeError as exc:
            if "exhausted" in str(exc) or "Bedrock" in str(exc):
                st.error(
                    f"AWS Bedrock error: {exc}\n\n"
                    "Check: aws configure · Bedrock model access · AWS_REGION in .env"
                )
            else:
                st.error(f"Error: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            st.stop()

    # 3. Save to history
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "intent": intent if "intent" in dir() else "",
    })
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer if "full_answer" in dir() else "",
        "sources": sources if "sources" in dir() else [],
        "model": model_short if "model_short" in dir() else "",
        "latency_ms": latency_ms if "latency_ms" in dir() else 0,
    })

    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("PharmaCog · MiniLM-L6-v2 · FAISS · AWS Bedrock · citations: book page + FAERS drug/reaction")