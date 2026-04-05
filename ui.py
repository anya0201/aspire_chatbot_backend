import streamlit as st
import matplotlib.pyplot as plt
import requests
import speech_recognition as sr
import pyttsx3
from pathlib import Path

st.set_page_config(page_title="ASPIRE AI", page_icon="🧠", layout="wide")

# --- Sidebar Login ---
st.sidebar.title("🔐 Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_status = st.sidebar.empty()
current_user = None

if st.sidebar.button("Login"):
    res = requests.get(
        "http://127.0.0.1:8000/login",
        params={"username": username, "password": password}
    ).json()
    login_status.success(res['message'])
    if "Welcome" in res['message']:
        current_user = username

# --- Dark mode styling ---
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 ASPIRE - AI Study Assistant")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_files" not in st.session_state:
    st.session_state.rag_files = []

# --- Chat display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- User Input ---
user_input = st.chat_input("Ask me anything... or try 'Quiz me' 😏")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    response = requests.get(
        "http://127.0.0.1:8000/chat",
        params={"user_input": user_input}
    ).json()

    bot_reply = response["response"]

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)

# --- Performance Dashboard ---
st.divider()
st.subheader("📊 Performance Dashboard")
if st.button("Show My Performance"):
    if not current_user:
        st.warning("Login first 😏")
    else:
        perf = requests.get("http://127.0.0.1:8000/performance").json()
        score = perf["score"]
        total = perf["total"]
        weak_topics = perf["weak_topics"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Score", f"{score} / {total}")
        with col2:
            st.metric("Accuracy", f"{(score/total)*100:.1f}%" if total else "0%")

        if weak_topics:
            st.subheader("📉 Weak Areas")
            topics = list(weak_topics.keys())
            values = list(weak_topics.values())
            fig, ax = plt.subplots()
            ax.bar(topics, values)
            ax.set_ylabel("Mistakes")
            ax.set_title("Weak Topics Analysis")
            st.pyplot(fig)

        if perf.get("recommendation"):
            st.warning(perf["recommendation"])
        else:
            st.success("You're doing great 😏")

# --- Leaderboard ---
st.divider()
st.subheader("🏆 Leaderboard")
if st.button("Show Leaderboard"):
    data = requests.get("http://127.0.0.1:8000/leaderboard").json()
    for i, user in enumerate(data["leaderboard"], start=1):
        st.write(f"{i}. {user['user']} — Score: {user['score']} | Accuracy: {user['accuracy']}%")

# --- File Upload for RAG ---
st.divider()
st.subheader("📚 Upload Files for AI Q&A")
uploaded_files = st.file_uploader("Upload PDF/Image", accept_multiple_files=True)
if uploaded_files:
    st.session_state.rag_files.extend(uploaded_files)
    st.success(f"{len(uploaded_files)} files uploaded")

# --- RAG Query ---
rag_query = st.text_input("Ask your uploaded files something 😏")
if st.button("Query Files") and rag_query:
    if not st.session_state.rag_files:
        st.warning("Upload files first 😅")
    else:
        payload = {
            "query": rag_query,
            "files": [f.name for f in st.session_state.rag_files]
        }
        res = requests.post("http://127.0.0.1:8000/rag_query", json=payload).json()
        st.info(res.get("answer", "No answer returned"))

# --- Voice Interaction ---
recognizer = sr.Recognizer()
if st.button("🎤 Speak"):
    with sr.Microphone() as source:
        st.info("Listening... speak now 😏")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        st.success(f"You said: {text}")

        res = requests.get(
            "http://127.0.0.1:8000/chat",
            params={"user_input": text}
        ).json()

        bot_reply = res["response"]
        st.write("🤖:", bot_reply)

        engine = pyttsx3.init()
        engine.say(bot_reply)
        engine.runAndWait()

    except:
        st.error("Couldn't understand you… try again 😅")
