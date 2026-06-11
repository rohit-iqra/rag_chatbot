import streamlit as st
import pdfplumber
import requests
import datetime

# -------------------- BASIC SETUP --------------------

st.set_page_config(
    page_title="Rohit's Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
PDF_PATH = "data/Rohit_Chatbot_Company_Manual.pdf"
MODEL_NAME = "llama-3.1-8b-instant"

# -------------------- CUSTOM STYLING --------------------

st.markdown("""
<style>

/* App background */
.stApp {
    background: linear-gradient(135deg, #0b1020 0%, #111827 35%, #1f2937 100%);
    color: #f9fafb;
    font-family: 'Inter', sans-serif;
}

/* Main container */
.main > div {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(17, 24, 39, 0.85);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {
    color: #f9fafb !important;
}

/* Header card */
.hero-card {
    background: linear-gradient(135deg, rgba(239,68,68,0.22), rgba(251,146,60,0.18));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    padding: 28px 30px;
    margin-bottom: 20px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.30);
    backdrop-filter: blur(14px);
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 1rem;
    color: #d1d5db;
    line-height: 1.6;
}

/* Chat message glass effect */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 12px 14px;
    margin-bottom: 12px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.20);
}

/* Assistant message accent */
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(251,146,60,0.10));
    border: 1px solid rgba(239,68,68,0.35);
}

/* User message accent */
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(251,146,60,0.16), rgba(253,186,116,0.08));
    border: 1px solid rgba(251,146,60,0.28);
}

/* Chat input */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.06);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 6px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.22);
}
[data-testid="stChatInput"] textarea {
    color: white !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #ef4444, #f97316);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.6rem 1rem;
    box-shadow: 0 8px 18px rgba(0,0,0,0.25);
}
.stButton > button:hover {
    transform: translateY(-1px);
    transition: 0.2s ease;
    filter: brightness(1.05);
}

/* Small cards in sidebar */
.info-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    margin: 4px 6px 0 0;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    background: rgba(239,68,68,0.18);
    border: 1px solid rgba(239,68,68,0.35);
    color: #e5e7eb;
}

/* Hide default top decoration if any */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Markdown text */
p, li, div {
    color: #f3f4f6;
}

/* Spinner text */
[data-testid="stSpinner"] * {
    color: #e5e7eb !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------- LOAD + CHUNK PDF --------------------

@st.cache_data
def load_chunks(max_chars: int = 600):
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            tx = page.extract_text()
            if tx:
                text += tx + "\n"
    raw_parts = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    buf = ""
    for part in raw_parts:
        if len(buf) + len(part) <= max_chars:
            buf += " " + part
        else:
            chunks.append(buf.strip())
            buf = part
    if buf:
        chunks.append(buf.strip())
    return chunks

pdf_chunks = load_chunks()

# -------------------- SIMPLE RETRIEVAL --------------------

def retrieve_context(query: str, top_k: int = 3):
    q_words = set(query.lower().split())
    scored = []
    for ch in pdf_chunks:
        ch_words = set(ch.lower().split())
        score = len(q_words & ch_words)
        if score > 0:
            scored.append((score, ch))
    if not scored:
        return ""
    scored.sort(reverse=True, key=lambda x: x[0])
    return "\n\n".join([c for _, c in scored[:top_k]])

# -------------------- GROQ API CALL --------------------

def llama_chat(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.4,
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Groq API Error:\n{str(e)}"

# -------------------- RAG + UPDATED INFO --------------------

def get_answer(question: str, history):
    context = retrieve_context(question)
    today = datetime.datetime.now().strftime("%d %B %Y (%Y)")
    pdf_strength = len(context.strip())

    if pdf_strength < 50:
        system_prompt = f"""
You are Rohit's Chatbot — a smart, friendly, and professional AI assistant built exclusively for Rohit.

Rules:
- Give clear and direct answers in English.
- Use your updated general knowledge (today = {today}).
- Do NOT say anything about "searching", "checking", "researching", or "not knowing".
- Never restrict information to the year 2023.
- Reply in a friendly, helpful, and professional style.
- Always refer to yourself as "Rohit's Chatbot" if asked about your identity.
"""
    else:
        system_prompt = f"""
You are Rohit's Chatbot — a smart, friendly, and professional AI assistant built exclusively for Rohit.

Use the following PDF text as your main reference.
If updated information (today = {today}) is needed, include it naturally.

PDF Context:
---------------------
{context}
---------------------

Rules:
- Provide confident and direct answers in English.
- Do NOT say "I am searching" or "I am researching".
- Never limit your knowledge to only 2023.
- Keep the reply natural, professional, and user-friendly.
- Always refer to yourself as "Rohit's Chatbot" if asked about your identity.
"""

    messages = [{"role": "system", "content": system_prompt}]
    for m in history[-6:]:
        messages.append(m)
    messages.append({"role": "user", "content": question})
    return llama_chat(messages)

# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("## ⚡ Rohit's Assistant")
    st.markdown("""
    <div class="info-card">
        <b>Your Personal AI Chat Experience</b><br><br>
        Ask anything related to your PDF manual or any general queries.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Features")
    st.markdown("""
    <span class="badge">PDF RAG</span>
    <span class="badge">Groq Powered</span>
    <span class="badge">Fast Replies</span>
    <span class="badge">Stylish UI</span>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="info-card">
        <b>Model:</b> llama-3.1-8b-instant<br>
        <b>Owner:</b> Rohit 👑<br>
        <b>Status:</b> Ready to help 💬
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hey Rohit! 👋 I'm your personal AI chatbot. Ask me anything — I'm here to help!"
            }
        ]
        st.rerun()

# -------------------- HEADER --------------------

st.markdown("""
<div class="hero-card">
    <div class="hero-title">🤖 Rohit's Chatbot</div>
    <div class="hero-subtitle">
        Welcome, Rohit! Your personal AI assistant — powered by Groq & LLaMA 3.1.
        Ask me anything from your PDF manual or any general question.
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hey Rohit! 👋 I'm your personal AI chatbot. Ask me anything — I'm here to help!"
        }
    ]

# -------------------- DISPLAY CHAT --------------------

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# -------------------- USER INPUT --------------------

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            answer = get_answer(user_input, st.session_state.messages)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
