import streamlit as st
import matplotlib.pyplot as plt
import requests
import speech_recognition as sr
import pyttsx3

st.set_page_config(page_title="ASPIRE AI", page_icon="🧠", layout="wide")
st.sidebar.title("🔐 Login / Profile")

# === Login ===
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
if st.sidebar.button("Login"):
    res = requests.get(
        "https://YOUR_BACKEND_URL/login",
        params={"username": username, "password": password}
    ).json()
    st.sidebar.success(res["message"])

# === Dark mode styling ===
st.markdown("""
<style>
body {background-color: #0e1117; color: white;}
.stChatMessage {border-radius: 15px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 ASPIRE - AI Study Assistant")

# === Chat memory ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === Display chat messages ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# === Chat / quiz / RAG query ===
user_input = st.chat_input("Ask me anything… or type 'Quiz me' 😏")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = requests.get(
        "https://YOUR_BACKEND_URL/chat",
        params={"user_input": user_input}
    ).json()
    bot_reply = response["response"]
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)

# === Performance dashboard ===
st.divider()
st.subheader("📊 Performance Dashboard")
if st.button("Show My Performance"):
    perf = requests.get("https://YOUR_BACKEND_URL/performance").json()
    score = perf["score"]
    total = perf["total"]
    weak_topics = perf["weak_topics"]

    col1, col2 = st.columns(2)
    with col1: st.metric("Score", f"{score} / {total}")
    with col2: st.metric("Accuracy", f"{(score/total)*100:.1f}%" if total else "0%")

    if weak_topics:
        st.subheader("📉 Weak Areas")
        topics = list(weak_topics.keys())
        values = list(weak_topics.values())
        fig, ax = plt.subplots()
        ax.bar(topics, values, color="tomato")
        ax.set_ylabel("Mistakes")
        ax.set_title("Weak Topics Analysis")
        st.pyplot(fig)

    if perf["recommendation"]: st.warning(perf["recommendation"])
    else: st.success("You're doing great 😏")

# === Leaderboard ===
st.divider()
st.subheader("🏆 Leaderboard")
if st.button("Show Leaderboard"):
    data = requests.get("https://YOUR_BACKEND_URL/leaderboard").json()
    for i, user in enumerate(data["leaderboard"], start=1):
        st.write(f"{i}. {user['user']} — Score: {user['score']} | Accuracy: {user['accuracy']}%")

# === RAG: Upload file & ask questions ===
st.divider()
st.subheader("📚 Upload Study Materials (PDF / Image)")
uploaded_file = st.file_uploader("Upload your file", type=["pdf", "png", "jpg", "jpeg"])
if uploaded_file:
    files_res = requests.post(
        "https://YOUR_BACKEND_URL/upload_file",
        files={"file": uploaded_file}
    ).json()
    st.success(files_res["message"])

st.subheader("❓ Ask Questions from Uploaded Files")
rag_query = st.text_input("Type your question here…")
if st.button("Get Answer from RAG"):
    if rag_query:
        answer_res = requests.post(
            "https://YOUR_BACKEND_URL/query_rag",
            params={"question": rag_query}
        ).json()
        st.info(f"RAG Answer: {answer_res['answer']}")

# === Voice Interaction ===
st.divider()
st.subheader("🎤 Talk to ASPIRE AI")

recognizer = sr.Recognizer()
engine = pyttsx3.init()

if st.button("🎙 Speak"):
    with sr.Microphone() as source:
        st.info("Listening… speak now 😏")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        st.success(f"You said: {text}")

        res = requests.get(
            "https://YOUR_BACKEND_URL/chat",
            params={"user_input": text}
        ).json()

        bot_reply = res["response"]
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.write("🤖:", bot_reply)

        # 🔊 Speak response
        engine.say(bot_reply)
        engine.runAndWait()

    except:
        st.error("Couldn't understand you… try again 😅")
