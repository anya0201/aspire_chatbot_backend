from fastapi import FastAPI, UploadFile, File
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory vectorstore per user
user_vectorstores = {}
# Temporary placeholder for logged-in user
current_user = "demo_user"

@app.get("/")
def root():
    return {"message": "Hello, Aspire AI!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global user_vectorstores, current_user

    if not current_user:
        return {"error": "Login first 😏"}

    file_path = os.path.join(UPLOAD_DIR, f"{current_user}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text
    text = ""
    if file.filename.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    else:
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
        except Exception:
            return {"error": "Failed to read image content"}

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Store per-user vectorstore
    user_vectorstores[current_user] = vectorstore

    return {"message": f"{file.filename} uploaded and ready for AI queries 😏"}
