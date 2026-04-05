import os
import random
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import requests
from pathlib import Path

# Cloud-ready HF API key
HF_API_KEY = os.getenv("HF_API_KEY")

app = FastAPI(title="ASPIRE AI Cloud")

# CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load quiz data
QUIZ_FILE = Path("quiz_data.json")
USER_FILE = Path("users.json")
with open(QUIZ_FILE) as f:
    quiz_data = json.load(f)

if USER_FILE.exists():
    with open(USER_FILE) as f:
        users = json.load(f)
else:
    users = {}

# Memory (cloud-safe, per session can be improved with DB)
sessions = {}

# ---------------------------
# Pydantic models
# ---------------------------
class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    username: str
    message: str

# ---------------------------
# Helpers
# ---------------------------
def save_users():
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

def detect_topic(text):
    text = text.lower()
    for topic in quiz_data.keys():
        if topic in text:
            return topic
    return "economics"

def hf_generate_explanation(question, answer):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    prompt = f"Question: {question}\nAnswer: {answer}\nExplain this in simple terms for a student."
    payload = {"inputs": prompt}
    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        res = r.json()
        return res[0]["generated_text"] if isinstance(res, list) else f"Answer: {answer}"
    except Exception:
        return f"(AI fallback) Correct answer: {answer}"

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def home():
    return {"message": "ASPIRE AI Cloud is running"}

@app.post("/login")
def login(req: LoginRequest):
    username = req.username
    password = req.password
    if username in users:
        if users[username]["password"] == password:
            sessions[username] = {}
            return {"message": f"Welcome back, {username} 😏"}
        else:
            return {"error": "Invalid credentials"}
    # New user
    users[username] = {"password": password, "score": 0, "total": 0, "weak_topics": {}}
    save_users()
    sessions[username] = {}
    return {"message": f"Account created! Welcome {username} 😏"}

@app.post("/chat")
def chat(req: ChatRequest):
    username = req.username
    text = req.message
    if username not in users:
        return {"error": "Login first 😏"}

    session = sessions.get(username, {})

    # If answering question
    if session.get("current_answer"):
        answer = session["current_answer"]
        topic = session["current_topic"]
        users[username]["total"] += 1
        if text.lower() in answer.lower():
            users[username]["score"] += 1
            response = "Correct 😏 You're getting sharp!"
        else:
            users[username]["weak_topics"][topic] = users[username]["weak_topics"].get(topic, 0) + 1
            response = hf_generate_explanation(session["current_question"], answer)
        save_users()
        session.clear()
        return {"response": response, "score": users[username]["score"], "total": users[username]["total"]}

    # Normal flow
    if "quiz" in text.lower():
        topic = detect_topic(text)
        q_obj = random.choice(quiz_data[topic])
        session["current_question"] = q_obj["question"]
        session["current_answer"] = q_obj["answer"]
        session["current_topic"] = topic
        return {"response": f"{topic.upper()} QUIZ 🎯\n\n{q_obj['question']}"}

    # RAG / AI query
    if "upload" in text.lower():
        return {"response": "Send file via /upload endpoint"}

    return {"response": f"Echo: {text} 😏 (normal chat fallback)"}

@app.post("/upload")
async def upload_file(username: str = Form(...), file: UploadFile = File(...)):
    if username not in users:
        return {"error": "Login first 😏"}
    folder = Path(f"user_files/{username}")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file.filename
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"message": f"File {file.filename} uploaded successfully 😏"}

@app.get("/performance")
def performance(username: str):
    if username not in users:
        return {"error": "Login first 😏"}
    user = users[username]
    rec = None
    if user["weak_topics"]:
        weakest = max(user["weak_topics"], key=user["weak_topics"].get)
        rec = f"You should revise {weakest} first… you're slipping there 😏"
    return {"score": user["score"], "total": user["total"], "weak_topics": user["weak_topics"], "recommendation": rec}

@app.get("/leaderboard")
def leaderboard():
    ranking = []
    for uname, data in users.items():
        total = data["total"]
        score = data["score"]
        acc = round((score/total)*100,2) if total>0 else 0
        ranking.append({"user": uname, "score": score, "accuracy": acc})
    ranking.sort(key=lambda x: (x["score"], x["accuracy"]), reverse=True)
    return {"leaderboard": ranking}

# ---------------------------
# Main entry (Render auto detects $PORT)
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
