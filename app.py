import os
import json
import random
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import httpx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# --- Config ---
HF_API_KEY = os.getenv("HF_API_KEY")  # HuggingFace API token
USERS_FILE = Path("users.json")

# --- In-memory multi-user sessions & uploads ---
sessions = {}
uploads = {}

# --- Sample intents (merged from intents.json) ---
INTENTS = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey"],
        "responses": ["Hey there!", "Hello! Ready to study?"]
    },
    {
        "tag": "explain_concept",
        "patterns": ["What is demand?", "Explain profit and loss"],
        "responses": ["Demand is the quantity people want to buy at a price."]
    },
    {
        "tag": "start_quiz",
        "patterns": ["Quiz me", "Test me", "Ask questions"],
        "responses": ["Alright, let's start! What is 2 + 2?"]
    }
]

# --- Sample quiz data (merge your quiz_data.json here) ---
QUIZ_DATA = {
    "economics": [
        {"question": "What is supply?", "answer": "Supply is the quantity of a good or service that producers are willing to sell at a given price."},
        {"question": "What is demand?", "answer": "Demand is the quantity of a good or service that consumers are willing to buy at a given price."}
    ],
    "math": [
        {"question": "What is 2 + 2?", "answer": "4"},
        {"question": "What is 5 * 6?", "answer": "30"}
    ]
}

# --- Load or initialize users ---
if USERS_FILE.exists():
    with USERS_FILE.open() as f:
        users = json.load(f)
else:
    users = {}

def save_users():
    with USERS_FILE.open("w") as f:
        json.dump(users, f, indent=4)

# --- Simple ML intent model ---
texts, labels = [], []
for intent in INTENTS:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=200)
model.fit(X, labels)

async def predict_intent(text: str) -> str:
    loop = asyncio.get_running_loop()
    X_test = vectorizer.transform([text])
    tag = await loop.run_in_executor(None, model.predict, X_test)
    return tag[0]

async def get_response(tag: str) -> str:
    loop = asyncio.get_running_loop()
    def choose():
        for intent in INTENTS:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return await loop.run_in_executor(None, choose)

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helpers ---
def detect_topic(user_input: str):
    user_input = user_input.lower()
    for topic in QUIZ_DATA.keys():
        if topic in user_input:
            return topic
    return "economics"

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

    # Predict intent using ML model
    tag = await predict_intent(user_input)
    intent_resp = await get_response(tag)
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
        question_obj = random.choice(QUIZ_DATA[topic])
        session["current_question"] = question_obj["question"]
        session["current_answer"] = question_obj["answer"]
        session["current_topic"] = topic
        return {"response": f"{topic.upper()} QUIZ 🎯\n\n{session['current_question']}"}

    # Fallback to AI
    prompt = f"Chat: {user_input}"
    response = await call_hf_llm(prompt)
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
