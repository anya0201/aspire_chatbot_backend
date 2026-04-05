from fastapi import UploadFile, File
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory vectorstore per user
user_vectorstores = {}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global user_vectorstores

    if not current_user:
        return {"error": "Login first 😏"}

    # Save the file
    file_path = os.path.join(UPLOAD_DIR, f"{current_user}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text
    if file.filename.lower().endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    else:
        # For images, use OCR
        try:
            import pytesseract
            from PIL import Image
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
        except Exception:
            return {"error": "Failed to read image content"}

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Save vectorstore for the user
    user_vectorstores[current_user] = vectorstore

    return {"message": f"{file.filename} uploaded and ready for AI queries 😏"}
