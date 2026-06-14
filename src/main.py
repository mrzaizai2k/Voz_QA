import sys
sys.path.append("")

import asyncio
import os
from datetime import datetime

import streamlit as st

from src.providers import get_openai_models, get_anthropic_models
from src.chat_history import (
    load_history,
    save_history,
    create_chat,
    get_chat_by_id,
    upsert_chat,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VOZ Thread Analyst",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark theme styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0f0f0f !important;
    color: #e0e0e0 !important;
}

.stApp {
    background: #0f0f0f !important;
}
.main .block-container {
    background: #0f0f0f !important;
    max-width: 820px;
    padding-top: 1.5rem;
    padding-bottom: 200px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161616 !important;
    border-right: 1px solid #2a2a2a !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}
section[data-testid="stSidebar"] * {
    color: #d0d0d0 !important;
}
section[data-testid="stSidebar"] label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #666 !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input {
    background: #1e1e1e !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    color: #e0e0e0 !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
section[data-testid="stSidebar"] .stTextInput input:focus,
section[data-testid="stSidebar"] .stNumberInput input:focus {
    border-color: #ff4500 !important;
    box-shadow: 0 0 0 2px rgba(255,69,0,0.15) !important;
}

section[data-testid="stSidebar"] .streamlit-expanderHeader {
    background: #1e1e1e !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    color: #d0d0d0 !important;
}
section[data-testid="stSidebar"] .streamlit-expanderContent {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-top: none !important;
}

/* ── History chat buttons — 80% width ── */
section[data-testid="stSidebar"] .stButton > button {
    background: #1e1e1e !important;
    border: 1px solid #2a2a2a !important;
    color: #c0c0c0 !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
    width: 80% !important;
    display: block !important;
    transition: border-color 0.15s, background 0.15s !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 4px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #1f1208 !important;
    border-color: #ff4500 !important;
    color: #ff6a33 !important;
}

/* Active history chat */
.active-chat-btn .stButton > button {
    background: #1f1208 !important;
    border-color: #ff4500 !important;
    color: #ff6a33 !important;
    width: 80% !important;
}

/* New Chat button — full width, accent */
.new-chat-btn .stButton > button {
    background: #ff4500 !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    width: 80% !important;
    margin-bottom: 4px;
}
.new-chat-btn .stButton > button:hover {
    background: #e03d00 !important;
    color: #fff !important;
}

/* Fetch models button */
.fetch-btn .stButton > button {
    background: #1e1e1e !important;
    border: 1px solid #444 !important;
    color: #d0d0d0 !important;
    font-size: 0.82rem !important;
    width: 100% !important;
}
.fetch-btn .stButton > button:hover {
    border-color: #ff4500 !important;
    color: #ff6a33 !important;
}

/* ── App header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.2rem;
}
.app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #f0f0f0;
    letter-spacing: -0.02em;
}
.app-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #ff4500;
    border: 1.5px solid #ff4500;
    padding: 2px 6px;
    border-radius: 3px;
}
.app-sub {
    color: #666;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
}

/* ── Chat bubbles ── */
.chat-turn { margin-bottom: 1.25rem; }

.bubble-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 0.4rem;
}
.bubble-user > .bubble-inner {
    background: #c93800;
    color: #fff;
    border-radius: 14px 14px 3px 14px;
    padding: 10px 16px;
    max-width: 72%;
    font-size: 0.9rem;
    line-height: 1.55;
}
.bubble-url-tag {
    font-size: 0.72rem;
    opacity: 0.8;
    margin-bottom: 5px;
    font-family: 'IBM Plex Mono', monospace;
    word-break: break-all;
    border-bottom: 1px solid rgba(255,255,255,0.2);
    padding-bottom: 4px;
}

.bubble-ai {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 0.4rem;
}
.bubble-avatar {
    width: 28px;
    height: 28px;
    background: #ff4500;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    font-weight: 700;
    color: #fff;
    flex-shrink: 0;
    margin-top: 2px;
}
.bubble-ai > .bubble-inner {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 3px 14px 14px 14px;
    padding: 10px 16px;
    max-width: 85%;
    font-size: 0.9rem;
    line-height: 1.65;
    color: #e0e0e0;
}
.turn-meta {
    font-size: 0.7rem;
    color: #444;
    text-align: right;
    margin-top: 2px;
}

/* ── Bottom input area ── */
.input-area-spacer { height: 180px; }

/* URL input row */
div[data-testid="stTextInput"] input {
    background: #1a1a1a !important;
    border: 1.5px solid #2e2e2e !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    color: #e0e0e0 !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #ff4500 !important;
    box-shadow: 0 0 0 2px rgba(255,69,0,0.12) !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #555 !important;
}

/* Textarea (expandable question box) */
div[data-testid="stTextArea"] textarea {
    background: #1a1a1a !important;
    border: 1.5px solid #2e2e2e !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    color: #e0e0e0 !important;
    resize: vertical !important;
    min-height: 60px !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #ff4500 !important;
    box-shadow: 0 0 0 2px rgba(255,69,0,0.12) !important;
}
div[data-testid="stTextArea"] textarea::placeholder {
    color: #555 !important;
}

/* ── Send button ── */
.send-col .stButton > button {
    background: #ff4500 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 10px 22px !important;
    width: 100% !important;
    /* Align with textarea bottom */
    margin-top: 22px !important;
    transition: background 0.15s !important;
}
.send-col .stButton > button:hover  { background: #e03d00 !important; }
.send-col .stButton > button:active { background: #c43600 !important; }

/* ── Section labels ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #555;
    margin: 1rem 0 0.5rem;
    padding: 0 2px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #444;
}
.empty-state .icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.empty-state p { font-size: 0.9rem; color: #555; }

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid #222;
    margin: 0.75rem 0;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #e0e0e0 !important;
    border-radius: 8px !important;
}

.stCaption, small { color: #555 !important; }

div[data-baseweb="select"] > div {
    background: #1e1e1e !important;
    border-color: #333 !important;
    color: #e0e0e0 !important;
}
[data-baseweb="menu"] { background: #1e1e1e !important; }
[data-baseweb="option"]:hover { background: #2a2a2a !important; }

div[data-testid="stNumberInput"] input {
    background: #1e1e1e !important;
    border-color: #333 !important;
    color: #e0e0e0 !important;
}

#MainMenu, footer { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #555; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "all_chats": load_history(),
        "active_chat_id": None,
        "streaming": False,
        "provider": "OpenAI",
        "available_models": [],
        "selected_model": "",
        # None means crawl all pages
        "max_pages": None,
        "max_posts": 300,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def active_chat() -> dict | None:
    return get_chat_by_id(st.session_state.all_chats, st.session_state.active_chat_id)


def do_new_chat():
    chat = create_chat(
        provider=st.session_state.provider,
        model=st.session_state.selected_model,
        max_pages=st.session_state.max_pages,
        max_posts=st.session_state.max_posts,
    )
    st.session_state.all_chats.insert(0, chat)
    st.session_state.active_chat_id = chat["id"]


def refresh_models():
    p = st.session_state.provider
    if p == "OpenAI":
        key = os.getenv("OPENAI_API_KEY", "")
        models = get_openai_models(key)
    else:
        key = os.getenv("ANTHROPIC_API_KEY", "")
        models = get_anthropic_models(key)
    st.session_state.available_models = models
    if models:
        st.session_state.selected_model = models[0]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:

    with st.expander("⚙️  Config", expanded=False):
        provider = st.selectbox(
            "Provider",
            ["OpenAI", "Anthropic"],
            index=["OpenAI", "Anthropic"].index(st.session_state.provider),
            key="provider_select",
        )
        st.session_state.provider = provider

        st.markdown('<div class="fetch-btn">', unsafe_allow_html=True)
        if st.button("🔄 Fetch models", key="fetch_models_btn"):
            with st.spinner("Fetching…"):
                refresh_models()
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.available_models:
            idx = 0
            if st.session_state.selected_model in st.session_state.available_models:
                idx = st.session_state.available_models.index(st.session_state.selected_model)
            model = st.selectbox(
                "Model",
                st.session_state.available_models,
                index=idx,
                key="model_select",
            )
            st.session_state.selected_model = model
        else:
            st.caption("Click 'Fetch models' to load available models.")

        # Max pages: None = crawl all
        crawl_all = st.checkbox(
            "Crawl all pages",
            value=(st.session_state.max_pages is None),
            key="crawl_all_checkbox",
        )
        if crawl_all:
            st.session_state.max_pages = None
            st.caption("All pages will be crawled.")
        else:
            val = st.session_state.max_pages if st.session_state.max_pages is not None else 9
            st.session_state.max_pages = st.number_input(
                "Max pages to crawl",
                min_value=1, max_value=100,
                value=val,
                step=1, key="max_pages_input",
            )

        st.session_state.max_posts = st.number_input(
            "Max posts in context",
            min_value=50, max_value=500,
            value=st.session_state.max_posts,
            step=50, key="max_posts_input",
        )

    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("＋  New Chat", key="new_chat_btn"):
        do_new_chat()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Chat history</div>', unsafe_allow_html=True)

    saved_chats = [c for c in st.session_state.all_chats if c.get("messages")]

    if not saved_chats:
        st.markdown(
            '<p style="font-size:0.8rem;color:#444;padding:0 4px;">No history yet.</p>',
            unsafe_allow_html=True,
        )
    else:
        for chat_item in saved_chats[:30]:
            label = chat_item.get("title", "Untitled")[:38]
            is_active = chat_item["id"] == st.session_state.active_chat_id
            btn_label = f"{'▸ ' if is_active else ''}{label}"

            if is_active:
                st.markdown('<div class="active-chat-btn">', unsafe_allow_html=True)

            if st.button(btn_label, key=f"chat_{chat_item['id']}"):
                st.session_state.active_chat_id = chat_item["id"]
                st.rerun()

            if is_active:
                st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Bootstrap active chat
# ---------------------------------------------------------------------------

if st.session_state.active_chat_id is None or active_chat() is None:
    do_new_chat()

chat = active_chat()

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="app-header">
    <span class="app-title">VOZ Thread Analyst</span>
    <span class="app-badge">Beta</span>
</div>
<p class="app-sub">Paste a VOZ thread URL, ask a question — AI reads the whole thread and answers.</p>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat message display
# ---------------------------------------------------------------------------

if not chat["messages"]:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🔍</div>
        <p>Enter a VOZ thread URL and ask a question below to get started.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in chat["messages"]:
        role = msg["role"]
        content = msg["content"]
        ts = msg.get("ts", "")

        if role == "user":
            url_part = msg.get("url", "")
            url_html = (
                f'<div class="bubble-url-tag">🔗 {url_part}</div>'
                if url_part else ""
            )
            st.markdown(f"""
<div class="chat-turn">
    <div class="bubble-user">
        <div class="bubble-inner">
            {url_html}
            {content}
        </div>
    </div>
    <div class="turn-meta">{ts}</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="chat-turn">
    <div class="bubble-ai">
        <div class="bubble-avatar">AI</div>
        <div class="bubble-inner">{content}</div>
    </div>
    <div class="turn-meta">{ts}</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input area — URL on top, question + button on same row below
# ---------------------------------------------------------------------------

st.markdown('<div class="input-area-spacer"></div>', unsafe_allow_html=True)

# Restore last used URL for this chat so user doesn't need to re-enter it
last_voz_url = chat.get("voz_url", "")

with st.container():
    # Row 1: URL input (full width)
    thread_url = st.text_input(
        "voz_url_label",
        value=last_voz_url,
        placeholder="https://voz.vn/t/… (paste thread URL here)",
        label_visibility="collapsed",
        key="thread_url_widget",
    )

    # Row 2: expandable question textarea + Analyse button
    col_q, col_send = st.columns([5, 1])

    with col_q:
        question = st.text_area(
            "question_label",
            placeholder="Ask something about this thread…",
            label_visibility="collapsed",
            key="question_widget",
            height=68,        # starts ~2 lines tall; user can drag to expand
        )

    with col_send:
        st.markdown('<div class="send-col">', unsafe_allow_html=True)
        send = st.button("Analyse →", key="send_btn")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# LLM builder
# ---------------------------------------------------------------------------

def build_llm():
    provider = st.session_state.provider
    model = st.session_state.selected_model or "gpt-4o"

    if provider == "Anthropic":
        from src.answer import AnthropicLLM
        return AnthropicLLM(model=model)
    else:
        from src.answer import OpenAILLM
        return OpenAILLM(model=model)


def run_analysis(url: str, q: str) -> str:
    from src.answer import ThreadQA

    llm = build_llm()
    # Pass None to crawl all pages, or an int to limit
    max_pages = st.session_state.max_pages
    qa = ThreadQA(llm=llm, max_pages=max_pages)

    original_build = qa._build_context
    max_p = st.session_state.max_posts
    qa._build_context = lambda df: original_build(df, max_posts=max_p)

    answer_ph = st.empty()
    final_answer = ""

    async def drive():
        nonlocal final_answer
        chunks = []
        async for chunk in qa.answer_stream(url, q):
            chunks.append(chunk)
            final_answer = "".join(chunks)
            answer_ph.markdown(f"""
<div class="chat-turn">
    <div class="bubble-ai">
        <div class="bubble-avatar">AI</div>
        <div class="bubble-inner">{final_answer}▌</div>
    </div>
</div>
""", unsafe_allow_html=True)
        answer_ph.empty()
        return final_answer

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(drive())
    loop.close()
    return final_answer


# ---------------------------------------------------------------------------
# Handle send
# ---------------------------------------------------------------------------

if send:
    if not thread_url.strip():
        st.warning("Please enter a VOZ thread URL.")
    elif not question.strip():
        st.warning("Please enter a question.")
    elif not st.session_state.selected_model:
        st.warning("Please open ⚙️ Config, click 'Fetch models' and select a model.")
    else:
        ts = datetime.now().strftime("%H:%M")

        # Always keep the latest URL on the chat object
        chat["voz_url"] = thread_url.strip()

        chat["messages"].append({
            "role": "user",
            "url": thread_url.strip(),
            "content": question.strip(),
            "ts": ts,
        })

        if len(chat["messages"]) == 1:
            chat["title"] = question.strip()[:40]

        st.session_state.all_chats = upsert_chat(st.session_state.all_chats, chat)
        save_history(st.session_state.all_chats)

        with st.spinner("Crawling thread & thinking…"):
            try:
                answer = run_analysis(thread_url.strip(), question.strip())
                chat["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "ts": datetime.now().strftime("%H:%M"),
                })
                st.session_state.all_chats = upsert_chat(st.session_state.all_chats, chat)
                save_history(st.session_state.all_chats)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")