import os
import json
import random
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()
HF_API_KEY = os.getenv("HF_API_KEY")  # HuggingFace token

# Allow CORS for your UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load quiz and user data
with open("quiz_data.json") as f:
    quiz_data = json.load(f)

if os.path.exists("users.json"):
    with open("users.json") as f:
        users = json.load(f)
else:
    users = {}

# In-memory session
current_question = None
current_answer = None
current_topic = None
current_user = None

uploaded_files = {}  # store file content per user for RAG


# === Helpers ===
def detect_topic(user_input):
    user_input = user_input.lower()
    for topic in quiz_data.keys():
        if topic in user_input:
            return topic
    return "economics"

def call_hf_llm(prompt):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt}
    try:
        res = requests.post(API_URL, headers=headers, json=payload).json()
        return res[0]["generated_text"]
    except:
        return "(AI fallback) Sorry, couldn't generate explanation."


# === Routes ===
@app.get("/")
def home():
    return {"message": "ASPIRE AI Cloud Backend Running 😏"}

@app.get("/login")
def login(username: str, password: str):
    global current_user
    if username in users and users[username]["password"] == password:
        current_user = username
        return {"message": f"Welcome {username} 😏"}
    return {"message": "Invalid credentials"}

@app.get("/chat")
def chat(user_input: str):
    global current_question, current_answer, current_topic, current_user
    if not current_user:
        return {"response": "Login first 😏"}

    user_data = users.get(current_user)
    if not user_data:
        users[current_user] = {"score":0, "total":0, "weak_topics":{}, "files":[]}
        user_data = users[current_user]

    # Answering quiz
    if current_answer:
        user_data["total"] += 1
        if user_input.lower() in current_answer.lower():
            user_data["score"] += 1
            response = "Correct 😏 You're getting sharp!"
        else:
            topic = current_topic
            user_data["weak_topics"][topic] = user_data["weak_topics"].get(topic,0)+1
            # AI explanation
            response = call_hf_llm(f"Question: {current_question}\nAnswer: {current_answer}\nExplain simply for a student.")

        # Reset
        current_question = None
        current_answer = None
        current_topic = None

        # Save user
        with open("users.json","w") as f:
            json.dump(users,f,indent=4)

        return {"response": response, "score": user_data["score"], "total": user_data["total"]}

    # Normal chat
    # detect quiz start
    if "quiz" in user_input.lower():
        topic = detect_topic(user_input)
        question_obj = random.choice(quiz_data[topic])
        current_question = question_obj["question"]
        current_answer = question_obj["answer"]
        current_topic = topic
        return {"response": f"{topic.upper()} QUIZ 🎯\n\n{current_question}"}

    # Otherwise, just reply using AI explanation (simple chatbot style)
    return {"response": call_hf_llm(f"Chat: {user_input}")}

@app.get("/performance")
def performance():
    if not current_user:
        return {"message":"Login first"}
    user_data = users[current_user]
    recommendation = None
    if user_data["weak_topics"]:
        weakest = max(user_data["weak_topics"], key=user_data["weak_topics"].get)
        recommendation = f"Revise {weakest} first 😏"
    return {"score": user_data["score"], "total": user_data["total"],
            "weak_topics": user_data["weak_topics"], "recommendation": recommendation}

@app.get("/leaderboard")
def leaderboard():
    ranking = []
    for u,data in users.items():
        score = data["score"]
        total = data["total"]
        accuracy = (score/total)*100 if total else 0
        ranking.append({"user":u, "score":score, "accuracy":round(accuracy,2)})
    ranking.sort(key=lambda x: (x["score"], x["accuracy"]), reverse=True)
    return {"leaderboard": ranking}

# === RAG: Upload files & query ===
@app.post("/upload_file")
def upload_file(file: UploadFile = File(...)):
    if not current_user:
        return {"message":"Login first"}
    content = file.file.read()
    if current_user not in uploaded_files:
        uploaded_files[current_user] = []
    uploaded_files[current_user].append({"name":file.filename, "content":content})
    return {"message": f"{file.filename} uploaded successfully 😏"}

@app.post("/query_rag")
def query_rag(question: str):
    if not current_user:
        return {"answer":"Login first 😏"}
    user_files = uploaded_files.get(current_user, [])
    if not user_files:
        return {"answer":"Upload files first 😅"}
    # For simplicity: concatenate text and ask LLM
    combined_text = b"".join([f["content"] for f in user_files]).decode(errors="ignore")
    prompt = f"User uploaded documents:\n{combined_text}\n\nQuestion: {question}\nAnswer simply for a student:"
    answer = call_hf_llm(prompt)
    return {"answer": answer}
