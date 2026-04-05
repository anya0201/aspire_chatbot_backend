import os
import json
import random
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import httpx

# --- HuggingFace API Key ---
HF_API_KEY = os.getenv("HF_API_KEY")

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory multi-user sessions ---
sessions = {}
uploads = {}

# --- Quiz data & intents ---
QUIZ_DATA = {
    "economics": [
        {"question": "What is demand?", "answer": "The quantity people want to buy at a price."},
        {"question": "What is supply?", "answer": "The quantity producers offer at a price."}
    ],
    "math": [
        {"question": "2 + 2 = ?", "answer": "4"},
        {"question": "5 * 3 = ?", "answer": "15"}
    ]
}

INTENTS = [
    {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey"], "responses": ["Hey there!", "Hello! Ready to study?"]},
    {"tag": "explain_concept", "patterns": ["What is demand?", "Explain profit and loss"], "responses": ["Demand is the quantity people want to buy at a price."]},
    {"tag": "start_quiz", "patterns": ["Quiz me", "Test me", "Ask questions"], "responses": ["Alright, let's start! What is 2 + 2?"]}
]

# --- Helpers ---
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

def detect_topic(user_input: str):
    user_input = user_input.lower()
    for topic in QUIZ_DATA.keys():
        if topic in user_input:
            return topic
    return "economics"

def find_intent_response(user_input: str):
    user_input = user_input.lower()
    for intent in INTENTS:
        for pattern in intent["patterns"]:
            if pattern.lower() in user_input:
                return random.choice(intent["responses"])
    return None

# --- Routes ---
@app.get("/")
async def home():
    return {"message": "ASPIRE AI Cloud Backend Running 😏"}

@app.get("/login")
async def login(username: str, password: str):
    # simple demo: password same as username for now
    if username == password:
        sessions[username] = {"current_question": None, "current_answer": None, "current_topic": None}
        return {"message": f"Welcome {username} 😏"}
    return {"message": "Invalid credentials"}

@app.get("/chat")
async def chat(username: str, user_input: str):
    if username not in sessions:
        return {"response": "Login first 😏"}

    session = sessions[username]

    # Check intents first
    intent_resp = find_intent_response(user_input)
    if intent_resp:
        return {"response": intent_resp}

    # Quiz answering
    if session["current_answer"]:
        # For simplicity, store score in session itself
        session.setdefault("score", 0)
        session.setdefault("total", 0)
        session["total"] += 1
        if user_input.lower() in session["current_answer"].lower():
            session["score"] += 1
            response = "Correct 😏 You're getting sharp!"
        else:
            topic = session["current_topic"]
            session.setdefault("weak_topics", {})
            session["weak_topics"][topic] = session["weak_topics"].get(topic, 0) + 1
            prompt = f"Question: {session['current_question']}\nAnswer: {session['current_answer']}\nExplain simply for a student."
            response = await call_hf_llm(prompt)
        # Reset
        session["current_question"] = session["current_answer"] = session["current_topic"] = None
        return {"response": response, "score": session["score"], "total": session["total"]}

    # Start quiz if triggered
    if "quiz" in user_input.lower():
        topic = detect_topic(user_input)
        question_obj = random.choice(QUIZ_DATA[topic])
        session["current_question"] = question_obj["question"]
        session["current_answer"] = question_obj["answer"]
        session["current_topic"] = topic
        return {"response": f"{topic.upper()} QUIZ 🎯\n\n{session['current_question']}"}

    # Otherwise, fallback to AI explanation
    prompt = f"Chat: {user_input}"
    response = await call_hf_llm(prompt)
    return {"response": response}

# --- RAG uploads ---
@app.post("/upload_file")
async def upload_file(username: str, file: UploadFile = File(...)):
    if username not in sessions:
        return {"message": "Login first 😏"}
    content = await file.read()
    uploads.setdefault(username, []).append({"name": file.filename, "content": content})
    return {"message": f"{file.filename} uploaded successfully 😏"}

@app.post("/query_rag")
async def query_rag(username: str, question: str):
    if username not in sessions:
        return {"answer": "Login first 😏"}
    user_files = uploads.get(username, [])
    if not user_files:
        return {"answer": "Upload files first 😅"}
    combined_text = b"".join([f["content"] for f in user_files]).decode(errors="ignore")
    prompt = f"User uploaded documents:\n{combined_text}\n\nQuestion: {question}\nAnswer simply for a student:" 
    answer = await call_hf_llm(prompt)
    return {"answer": answer}
