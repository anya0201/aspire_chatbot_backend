import os
import json
import random
import streamlit as st
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
from pathlib import Path


# 📦 HuggingFace / LLM imports
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# 🌐 FastAPI setup
app = FastAPI(title="ASPIRE AI Cloud Tutor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 🔑 HuggingFace API
HF_API_KEY = os.getenv("HF_API_KEY")
HUGGINGFACE_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# 🗂 Paths
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

QUIZ_FILE = DATA_DIR / "quiz_data.json"
USERS_FILE = DATA_DIR / "users.json"

# 🧠 Load quiz and users
with open(QUIZ_FILE) as f:
    quiz_data = json.load(f)

if USERS_FILE.exists():
    with open(USERS_FILE) as f:
        users = json.load(f)
else:
    users = {}

# 🔹 In-memory RAG storage (FAISS per user)
user_rag = {}

# ----------------- HELPERS -----------------
def save_users():
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def detect_topic(user_input: str):
    user_input = user_input.lower()
    for topic in quiz_data.keys():
        if topic in user_input:
            return topic
    return "general"

def generate_explanation(question, answer, user_id):
    """
    Calls HF model to generate explanation
    """
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    prompt = f"Question: {question}\nCorrect Answer: {answer}\nExplain simply for a student."
    payload = {"inputs": prompt}
    try:
        response = requests.post(API_URL, headers=HUGGINGFACE_HEADERS, json=payload)
        return response.json()[0]["generated_text"]
    except Exception:
        return f"(AI fallback) Correct answer: {answer}"

# ----------------- ENDPOINTS -----------------

@app.get("/")
def home():
    return {"message": "ASPIRE AI Cloud Tutor Running 😏"}

# 🔐 Login
@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    if username in users and users[username]["password"] == password:
        return {"message": f"Welcome {username} 😏"}
    elif username not in users:
        # create new user
        users[username] = {
            "password": password,
            "score": 0,
            "total": 0,
            "weak_topics": {},
        }
        save_users()
        return {"message": f"New account created for {username} 😎"}
    else:
        return JSONResponse(status_code=401, content={"message": "Invalid credentials"})

# 🎯 Chat / Quiz
@app.post("/chat")
async def chat(user_input: str = Form(...), username: str = Form(...)):
    if username not in users:
        return JSONResponse(status_code=401, content={"response": "Login first 😏"})

    user_data = users[username]

    # if answering a question
    if "current_question" in user_data and user_data["current_question"]:
        user_data["total"] += 1
        correct_answer = user_data["current_answer"]
        if user_input.lower() in correct_answer.lower():
            user_data["score"] += 1
            response = "Correct 😏 You're getting sharp!"
        else:
            topic = user_data["current_topic"]
            user_data["weak_topics"][topic] = user_data["weak_topics"].get(topic, 0) + 1
            response = generate_explanation(user_data["current_question"], correct_answer, username)

        # reset
        user_data["current_question"] = None
        user_data["current_answer"] = None
        user_data["current_topic"] = None
        save_users()
        return {"response": response, "score": user_data["score"], "total": user_data["total"]}

    # normal flow
    intent = "start_quiz" if "quiz" in user_input.lower() else "chat"

    if intent == "start_quiz":
        topic = detect_topic(user_input)
        question_obj = random.choice(quiz_data.get(topic, quiz_data["general"]))
        user_data["current_question"] = question_obj["question"]
        user_data["current_answer"] = question_obj["answer"]
        user_data["current_topic"] = topic
        save_users()
        return {"response": f"{topic.upper()} QUIZ 🎯\n\n{question_obj['question']}"}

    # RAG query fallback
    if username in user_rag:
        retriever = user_rag[username]["retriever"]
        qa_chain = RetrievalQA.from_chain_type(
            llm=HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0.3}),
            retriever=retriever
        )
        answer = qa_chain.run(user_input)
        return {"response": answer}

    return {"response": f"🤖 {user_input} (chat fallback)"}

# 📚 Upload files for RAG
@app.post("/upload")
async def upload_file(username: str = Form(...), files: List[UploadFile] = File(...)):
    if username not in users:
        return JSONResponse(status_code=401, content={"message": "Login first 😏"})

    docs_text = []
    for f in files:
        path = UPLOAD_DIR / f"{username}_{f.filename}"
        content = await f.read()
        with open(path, "wb") as out:
            out.write(content)
        docs_text.append(content.decode("utf-8", errors="ignore"))

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(" ".join(docs_text))

    # Create embeddings
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    user_rag[username] = {"vectorstore": vectorstore, "retriever": vectorstore.as_retriever()}

    return {"message": f"Uploaded {len(files)} files and ready to answer questions 😏"}

# 📊 Performance
@app.get("/performance")
def performance(username: str):
    if username not in users:
        return JSONResponse(status_code=401, content={"message": "Login first 😏"})
    user_data = users[username]
    recommendation = None
    if user_data["weak_topics"]:
        weakest = max(user_data["weak_topics"], key=user_data["weak_topics"].get)
        recommendation = f"You should revise {weakest} first… you're slipping there 😏"
    return {
        "score": user_data["score"],
        "total": user_data["total"],
        "weak_topics": user_data["weak_topics"],
        "recommendation": recommendation
    }

# 🏆 Leaderboard
@app.get("/leaderboard")
def leaderboard():
    ranking = []
    for username, data in users.items():
        score = data["score"]
        total = data["total"]
        accuracy = (score / total) * 100 if total else 0
        ranking.append({"user": username, "score": score, "accuracy": round(accuracy, 2)})
    ranking = sorted(ranking, key=lambda x: (x["score"], x["accuracy"]), reverse=True)
    return {"leaderboard": ranking}

# ----------------- END -----------------
