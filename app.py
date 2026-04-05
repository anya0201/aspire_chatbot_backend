import os
import json
import random
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import httpx

# Optional extras
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

try:
    import pyttsx3
except:
    pyttsx3 = None

try:
    import speech_recognition as sr
except:
    sr = None

# --- Configuration ---
HF_API_KEY = os.getenv("HF_API_KEY")
QUIZ_FILE = Path("quiz_data.json")
USERS_FILE = Path("users.json")

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load data ---
quiz_data = {}
if QUIZ_FILE.exists():
    with QUIZ_FILE.open() as f:
        quiz_data = json.load(f)

users = {}
if USERS_FILE.exists():
    with USERS_FILE.open() as f:
        users = json.load(f)

# --- In-memory multi-user sessions & uploads ---
sessions = {}
uploads = {}

# --- Quiz + intents (merged directly) ---
intents = [
    {"tag":"greeting","patterns":["hi","hello","hey"],"responses":["Hey there!","Hello! Ready to study?"]},
    {"tag":"start_quiz","patterns":["quiz me","test me","ask questions"],"responses":["Alright, let's start!"]},
    {"tag":"explain_concept","patterns":["what is demand?","explain profit and loss"],"responses":["Demand is the quantity people want to buy at a price."]}
]

# --- Helpers ---
def detect_topic(user_input: str):
    user_input = user_input.lower()
    for topic in quiz_data.keys():
        if topic in user_input:
            return topic
    return "economics"

def find_intent_response(user_input: str):
    user_input_lower = user_input.lower()
    for intent in intents:
        for pattern in intent["patterns"]:
            if pattern.lower() in user_input_lower:
                return random.choice(intent["responses"])
    return None

async def call_hf_llm(prompt: str):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            res = await client.post(API_URL, headers=headers, json={"inputs": prompt})
            res_json = res.json()
            return res_json[0].get("generated_text", "(AI fallback) Sorry 😅")
        except Exception:
            return "(AI fallback) Sorry 😅"

def save_users():
    with USERS_FILE.open("w") as f:
        json.dump(users, f, indent=4)

# --- Routes ---
@app.get("/")
async def home():
    return {"message": "ASPIRE AI Cloud Backend Running 😏"}

@app.get("/login")
async def login(username: str, password: str):
    if username in users and users[username]["password"] == password:
        sessions[username] = {"current_question": None, "current_answer": None, "current_topic": None}
        return {"message": f"Welcome {username} 😏"}
    return {"message": "Invalid credentials"}

@app.get("/chat")
async def chat(username: str, user_input: str):
    if username not in sessions:
        return {"response": "Login first 😏"}
    session = sessions[username]
    user_data = users.setdefault(username, {"score":0, "total":0, "weak_topics":{}, "files":[]})

    # Intent detection
    intent_resp = find_intent_response(user_input)
    if intent_resp:
        return {"response": intent_resp}

    # Quiz answering
    if session["current_answer"]:
        user_data["total"] += 1
        if user_input.lower() in session["current_answer"].lower():
            user_data["score"] += 1
            response = "Correct 😏 You're getting sharp!"
        else:
            topic = session["current_topic"]
            user_data["weak_topics"][topic] = user_data["weak_topics"].get(topic,0)+1
            prompt = f"Question: {session['current_question']}\nAnswer: {session['current_answer']}\nExplain simply for a student."
            response = await call_hf_llm(prompt)
        session["current_question"] = session["current_answer"] = session["current_topic"] = None
        save_users()
        return {"response": response, "score": user_data["score"], "total": user_data["total"]}

    # Start quiz
    if "quiz" in user_input.lower():
        topic = detect_topic(user_input)
        question_obj = random.choice(quiz_data[topic])
        session["current_question"] = question_obj["question"]
        session["current_answer"] = question_obj["answer"]
        session["current_topic"] = topic
        return {"response": f"{topic.upper()} QUIZ 🎯\n\n{session['current_question']}"}

    # Fallback AI
    response = await call_hf_llm(f"Chat: {user_input}")
    return {"response": response}

@app.get("/performance")
async def performance(username: str):
    if username not in sessions:
        return {"message":"Login first 😏"}
    data = users[username]
    recommendation = None
    if data["weak_topics"]:
        weakest = max(data["weak_topics"], key=data["weak_topics"].get)
        recommendation = f"Revise {weakest} first 😏"
    return {"score": data["score"], "total": data["total"], "weak_topics": data["weak_topics"], "recommendation": recommendation}

@app.get("/leaderboard")
async def leaderboard():
    ranking = []
    for u, data in users.items():
        score = data["score"]
        total = data["total"]
        accuracy = (score/total)*100 if total else 0
        ranking.append({"user": u, "score": score, "accuracy": round(accuracy,2)})
    ranking.sort(key=lambda x: (x["score"], x["accuracy"]), reverse=True)
    return {"leaderboard": ranking}

# --- File upload & RAG ---
@app.post("/upload_file")
async def upload_file(username: str, file: UploadFile = File(...)):
    if username not in sessions:
        return {"message":"Login first 😏"}
    content = await file.read()
    uploads.setdefault(username, []).append({"name": file.filename, "content": content})
    return {"message": f"{file.filename} uploaded successfully 😏"}

@app.post("/query_rag")
async def query_rag(username: str, question: str):
    if username not in sessions:
        return {"answer":"Login first 😏"}
    user_files = uploads.get(username, [])
    if not user_files:
        return {"answer":"Upload files first 😅"}
    combined_text = b"".join([f["content"] for f in user_files]).decode(errors="ignore")
    prompt = f"User uploaded documents:\n{combined_text}\n\nQuestion: {question}\nAnswer simply for a student:"
    answer = await call_hf_llm(prompt)
    return {"answer": answer}
