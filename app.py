import random
import json
import os
import requests
from fastapi import FastAPI
from model import predict_intent, get_response

app = FastAPI()

# 🔐 Load API Key
HF_API_KEY = os.getenv("HF_API_KEY")

# 📂 Load data
with open("quiz_data.json") as f:
    quiz_data = json.load(f)

with open("users.json") as f:
    users = json.load(f)

# 🧠 Memory (session-level for demo)
current_question = None
current_answer = None
current_topic = None
current_user = None


# 🎯 Topic detection
def detect_topic(user_input):
    user_input = user_input.lower()

    for topic in quiz_data.keys():
        if topic in user_input:
            return topic

    return "economics"


# 🤖 AI Explanation (Hugging Face)
def generate_explanation(question, answer):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }

    prompt = f"""
    Question: {question}
    Correct Answer: {answer}
    Explain this in a simple way for a student.
    """

    payload = {"inputs": prompt}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]

        return f"(AI fallback) Correct answer: {answer}"

    except Exception:
        return f"(AI fallback) Correct answer: {answer}"


# 🏠 Home
@app.get("/")
def home():
    return {"message": "ASPIRE Chatbot is running 😏"}


# 🔐 Login
@app.get("/login")
def login(username: str, password: str):
    global current_user

    if username in users and users[username]["password"] == password:
        current_user = username
        return {"message": f"Welcome {username} 😏"}

    return {"message": "Invalid credentials"}


# 💬 Chatbot
@app.get("/chat")
def chat(user_input: str):
    global current_question, current_answer, current_topic, current_user

    if not current_user:
        return {"response": "Login first 😏"}

    user_data = users[current_user]

    # 🧠 If answering a quiz
    if current_answer:
        user_data["total"] += 1

        if user_input.lower() in current_answer.lower():
            user_data["score"] += 1
            response = "Correct 😏 You're getting sharp!"
        else:
            topic = current_topic
            user_data["weak_topics"][topic] = user_data["weak_topics"].get(topic, 0) + 1

            response = generate_explanation(current_question, current_answer)

        # 💾 Save progress
        with open("users.json", "w") as f:
            json.dump(users, f, indent=4)

        # 🔄 Reset
        current_question = None
        current_answer = None
        current_topic = None

        return {
            "response": response,
            "score": user_data["score"],
            "total": user_data["total"]
        }

    # 💬 Normal intent
    intent = predict_intent(user_input)

    if intent == "start_quiz":
        topic = detect_topic(user_input)

        question_obj = random.choice(quiz_data[topic])

        current_question = question_obj["question"]
        current_answer = question_obj["answer"]
        current_topic = topic

        return {
            "response": f"{topic.upper()} QUIZ 🎯\n\n{current_question}"
        }

    return {"response": get_response(intent)}


# 📊 Performance
@app.get("/performance")
def performance():
    if not current_user:
        return {"message": "Login first 😏"}

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


# 🏆 Leaderboard
@app.get("/leaderboard")
def leaderboard():
    ranking = []

    for username, data in users.items():
        score = data["score"]
        total = data["total"]
        accuracy = (score / total) * 100 if total > 0 else 0

        ranking.append({
            "user": username,
            "score": score,
            "accuracy": round(accuracy, 2)
        })

    ranking = sorted(ranking, key=lambda x: (x["score"], x["accuracy"]), reverse=True)

    return {"leaderboard": ranking}