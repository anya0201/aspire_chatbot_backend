import streamlit as st
import matplotlib.pyplot as plt
import requests
import speech_recognition as sr
import pyttsx3
from PIL import Image

# Page config
st.set_page_config(page_title="ASPIRE AI Tutor", page_icon="🧠", layout="wide")

# 🎨 Sidebar: Login
st.sidebar.title("🔐 Login / Session")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    res = requests.get(
        "https://aspire-chatbot.onrender.com",  
        params={"username": username, "password": password}
    ).json()
    st.sidebar.success(res["message"])
    if "Welcome" in res["message"]:
        st.session_state.user = username

# Dark mode styling
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.stChatMessage { border-radius: 15px; padding: 10px; }
.stButton button { background-color: #1f2937; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 ASPIRE AI Tutor")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_input = st.chat_input("Ask me anything... or 'Quiz me' 😏")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    response = requests.get(
        "https://YOUR_DEPLOYED_BACKEND_URL/chat",
        params={"user_input": user_input}
    ).json()
    bot_reply = response["response"]
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)

# 📊 Performance Dashboard
st.divider()
st.subheader("📊 Performance Dashboard")

if st.button("Show My Performance"):
    perf = requests.get(f"https://YOUR_DEPLOYED_BACKEND_URL/performance").json()
    score, total = perf["score"], perf["total"]
    weak_topics = perf["weak_topics"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score", f"{score} / {total}")
    with col2:
        st.metric("Accuracy", f"{(score/total)*100:.1f}%" if total else "0%")

    # Weak topics chart
    if weak_topics:
        st.subheader("📉 Weak Topics")
        topics, values = list(weak_topics.keys()), list(weak_topics.values())
        fig, ax = plt.subplots()
        ax.bar(topics, values, color="tomato")
        ax.set_ylabel("Mistakes")
        ax.set_title("Weak Topic Analysis")
        st.pyplot(fig)

    # Recommendation
    if perf.get("recommendation"):
        st.warning(perf["recommendation"])
    else:
        st.success("You're doing great 😏")

# 🏆 Leaderboard
st.divider()
st.subheader("🏆 Leaderboard")
if st.button("Show Leaderboard"):
    data = requests.get("https://YOUR_DEPLOYED_BACKEND_URL/leaderboard").json()
    for i, user in enumerate(data["leaderboard"], start=1):
        st.write(f"{i}. {user['user']} — Score: {user['score']} | Accuracy: {user['accuracy']}%")

# 🎤 Voice Interaction
recognizer = sr.Recognizer()
st.divider()
st.subheader("🎤 Talk to ASPIRE AI")

if st.button("Speak"):
    with sr.Microphone() as source:
        st.info("Listening... speak now 😏")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        st.success(f"You said: {text}")

        res = requests.get(
            "https://YOUR_DEPLOYED_BACKEND_URL/chat",
            params={"user_input": text}
        ).json()
        bot_reply = res["response"]
        st.write("🤖:", bot_reply)

        engine = pyttsx3.init()
        engine.say(bot_reply)
        engine.runAndWait()

    except:
        st.error("Couldn't understand you… try again 😅")

# 📚 RAG File Upload
st.divider()
st.subheader("📂 Upload Files for AI Q&A")
uploaded_file = st.file_uploader("Upload PDF / Image", type=["pdf","png","jpg","jpeg"])
if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("https://YOUR_DEPLOYED_BACKEND_URL/upload", files=files).json()
    st.success(response.get("message","File uploaded successfully 😏"))

st.info("Everything is connected! Ask questions or take quizzes and watch the AI tutor flex 😎")
