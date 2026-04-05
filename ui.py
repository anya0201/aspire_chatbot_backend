import streamlit as st
import matplotlib.pyplot as plt
import requests
import speech_recognition as sr
import pyttsx3

st.set_page_config(page_title="ASPIRE AI", page_icon="🧠", layout="wide")

# 🔐 Sidebar Login
st.sidebar.title("🔐 Login")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    res = requests.get(
        "http://127.0.0.1:8000/login",
        params={"username": username, "password": password}
    ).json()

    st.sidebar.success(res.get("message", "Login failed"))

# 🌙 Dark UI
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

# 🧠 Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# 💬 Show chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 🎤 + 💬 Controls
col1, col2 = st.columns([4, 1])

user_input = col1.chat_input("Ask me anything... or try 'Quiz me' 😏")

recognizer = sr.Recognizer()

# 🎤 Voice input
if col2.button("🎤"):
    with sr.Microphone() as source:
        st.info("Listening... speak now 😏")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)

        st.session_state.messages.append({"role": "user", "content": text})
        with st.chat_message("user"):
            st.write(text)

        res = requests.get(
            "http://127.0.0.1:8000/chat",
            params={"user_input": text}
        ).json()

        bot_reply = res.get("response", "Something went wrong 😅")

        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.write(bot_reply)

        # 🔊 Voice reply
        engine = pyttsx3.init()
        engine.say(bot_reply)
        engine.runAndWait()

    except:
        st.error("Couldn't understand you… try again 😅")

# 💬 Text input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    try:
        response = requests.get(
            "http://127.0.0.1:8000/chat",
            params={"user_input": user_input}
        ).json()

        bot_reply = response.get("response", "Something broke 😅")

    except:
        bot_reply = "Server not responding… check backend 😏"

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)

# 📊 Performance Dashboard
st.divider()
st.subheader("📊 Performance Dashboard")

if st.button("Show My Performance"):
    try:
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

    except:
        st.error("Couldn't load performance… backend sleeping? 😴")

# 🏆 Leaderboard
st.divider()
st.subheader("🏆 Leaderboard")

if st.button("Show Leaderboard"):
    try:
        data = requests.get("http://127.0.0.1:8000/leaderboard").json()

        for i, user in enumerate(data["leaderboard"], start=1):
            st.write(f"{i}. {user['user']} — Score: {user['score']} | Accuracy: {user['accuracy']}%")

    except:
        st.error("Leaderboard failed… dramatic, I know 😏")