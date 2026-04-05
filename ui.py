import streamlit as st
import matplotlib.pyplot as plt
import requests
from PIL import Image

# --- CONFIGURATION ---
# Replace this if your Render URL is different!
BACKEND_URL = "https://aspire-chatbot.onrender.com"

# Page config
st.set_page_config(page_title="ASPIRE AI Tutor", page_icon="🧠", layout="wide")

# Dark mode styling
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.stChatMessage { border-radius: 15px; padding: 10px; }
.stButton button { background-color: #1f2937; color: white; }
</style>
""", unsafe_allow_html=True)

# 🎨 Sidebar: Login
st.sidebar.title("🔐 Login / Session")

if "user" not in st.session_state:
    st.session_state.user = None

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    try:
        # FIXED: Added /login endpoint
        res = requests.get(f"{BACKEND_URL}/login", params={"username": username, "password": password})
        data = res.json()
        
        if res.status_code == 200 and "Welcome" in data.get("message", ""):
            st.session_state.user = username
            st.sidebar.success(data["message"])
        else:
            st.sidebar.error(data.get("message", "Invalid credentials"))
    except Exception as e:
        st.sidebar.error("Backend is currently offline or waking up...")

st.title("🧠 ASPIRE AI Tutor")

# --- Landing Page (Visible when NOT logged in) ---
if not st.session_state.user:
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Welcome to ASPIRE AI 🚀</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #A0AEC0;'>Your Personal, Data-Driven Smart Tutor</h4>", unsafe_allow_html=True)
    st.divider()
    
    # Feature Showcase
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🧠 **Smart Chat & Quizzes**\n\nAsk questions or type 'Quiz me' to test your knowledge dynamically.")
    with col2:
        st.warning("📊 **Performance Tracking**\n\nASPIRE analyzes your weak points and graphs your learning progress.")
    with col3:
        st.success("📚 **Document Chat (RAG)**\n\nUpload your PDFs and let the AI teach you from your own notes.")
        
    st.divider()
    st.markdown("<p style='text-align: center; font-size: 18px;'>👈 <b>Log in using the sidebar to access your dashboard!</b></p>", unsafe_allow_html=True)
    
    # Stop the app here so unauthenticated users don't see the chat UI
    st.stop()

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

    with st.spinner("ASPIRE is thinking..."):
        try:
            # FIXED: Added username to params and used the actual backend URL
            response = requests.get(
                f"{BACKEND_URL}/chat",
                params={"username": st.session_state.user, "user_input": user_input}
            )
            bot_reply = response.json().get("response", "Error connecting to AI.")
        except:
            bot_reply = "Backend connection failed. Is Render awake?"

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)

# 📊 Performance Dashboard
st.divider()
st.subheader("📊 Performance Dashboard")

if st.button("Show My Performance"):
    # FIXED: Added username to params
    res = requests.get(f"{BACKEND_URL}/performance", params={"username": st.session_state.user})
    if res.status_code == 200:
        perf = res.json()
        score, total = perf.get("score", 0), perf.get("total", 0)
        weak_topics = perf.get("weak_topics", {})

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Score", f"{score} / {total}")
        with col2:
            st.metric("Accuracy", f"{(score/total)*100:.1f}%" if total > 0 else "0%")

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
    res = requests.get(f"{BACKEND_URL}/leaderboard")
    if res.status_code == 200:
        data = res.json()
        for i, user in enumerate(data.get("leaderboard", []), start=1):
            st.write(f"{i}. {user['user']} — Score: {user['score']} | Accuracy: {user['accuracy']}%")

# 📚 RAG File Upload
st.divider()
st.subheader("📂 Upload Files for AI Q&A")
uploaded_file = st.file_uploader("Upload PDF / Image", type=["pdf","png","jpg","jpeg"])
if uploaded_file and st.button("Upload to Backend"):
    with st.spinner("Uploading..."):
        # FIXED: Pass username and correct file formatting for FastAPI
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(
            f"{BACKEND_URL}/upload_file", # FIXED: correct endpoint name
            params={"username": st.session_state.user},
            files=files
        )
        st.success(response.json().get("message","File uploaded successfully 😏"))

st.info("Everything is connected! Ask questions or take quizzes and watch the AI tutor flex 😎")

# --- NOTE ON VOICE ---
# pyttsx3 and speech_recognition require local system hardware. 
# They will crash a cloud server. To do voice in Streamlit Cloud, 
# look into libraries like 'streamlit-mic-recorder' or standard HTML5 audio components.
