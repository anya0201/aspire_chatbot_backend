import os
import json
import random
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import requests

# === Cloud API Keys ===
HF_API_KEY = os.getenv("HF_API_KEY")  # HuggingFace LLM & embeddings

# === App setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load data ===
with open("quiz_data.json") as f:
    quiz_data = json.load(f)

with open("users.json") as f:
    users = json.load(f)

# === Memory ===
current_question = None
current_answer = None
current_topic = None
current_user = None

# === Pydantic models ===
class QuestionResponse(BaseModel):
    user_input: str

# === Utilities ===
def detect_topic(user_input):
    for topic in quiz_data.keys():
        if topic in user_input.lower():
            return topic
    return "economics"  # default fallback

def generate_explanation(question, answer):
    """Call HF LLM for explanation using RAG/QA"""
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    prompt = f"Question: {question}\nAnswer: {answer}\nExplain this clearly for a student."
    payload = {"inputs": prompt}
    try:
        resp = requests.post(API_URL, headers=headers, json=payload).json()
        return resp[0]["generated_text"]
    except Exception:
        return f"(AI fallback) Correct answer: {answer}"

# === Login system ===
@app.get("/login")
def login(username: str, password: str):
    global current_user
    if username in users and users[username]["password"] == password:
        current_user = username
        return {"message": f"Welcome {username} 😏"}
    return {"message": "Invalid credentials"}

# === Chat / Quiz ===
@app.get("/chat")
def chat(user_input: str):
    global current_question, current_answer, current_topic, current_user
    if not current_user:
        return {"response": "Login first 😏"}

    user_data = users[current_user]

    # ✅ Answering a question
    if current_answer:
        user_data["total"] += 1
        if user_input.lower() in current_answer.lower():
            user_data["score"] += 1
            response = "Correct 😏 You're getting sharp!"
        else:
            topic = current_topic
            user_data["weak_topics"][topic] = user_data["weak_topics"].get(topic, 0) + 1
            response = generate_explanation(current_question, current_answer)

        # reset
        current_question = None
        current_answer = None
        current_topic = None

        # save users
        with open("users.json", "w") as f:
            json.dump(users, f, indent=4)

        return {"response": response, "score": user_data["score"], "total": user_data["total"]}

    # 💬 Normal chatbot / start quiz
    intent = "start_quiz" if "quiz" in user_input.lower() else "chat"
    if intent == "start_quiz":
        topic = detect_topic(user_input)
        question_obj = random.choice(quiz_data[topic])
        current_question = question_obj["question"]
        current_answer = question_obj["answer"]
        current_topic = topic
        return {"response": f"{topic.upper()} QUIZ 🎯\n\n{current_question}"}

    return {"response": f"Chat response placeholder: {user_input}"}

# === Performance & recommendations ===
@app.get("/performance")
def performance():
    if not current_user:
        return {"message": "Login first"}
    user_data = users[current_user]
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

# === Leaderboard ===
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

# === RAG: File upload + QA ===
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    if not current_user:
        return {"message": "Login first 😏"}
    # Read file content
    content = await file.read()
    # TODO: convert PDF/images to text, chunk, embed, save vectors in FAISS or Chroma
    # placeholder
    return {"message": f"File '{file.filename}' uploaded! Processing will be done soon 😏"}

@app.post("/query_rag")
def query_rag(question: str):
    if not current_user:
        return {"message": "Login first 😏"}
    # TODO: retrieve top-k chunks from embeddings & send to HF LLM
    # placeholder
    return {"answer": f"(RAG placeholder) Answer to: {question}"}
