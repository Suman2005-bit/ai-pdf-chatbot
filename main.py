from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import os
import requests

app = FastAPI()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Globals
chunks = []
chunk_embeddings = None
index = None

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load existing index and chunks if they exist
if os.path.exists("faiss_index.bin") and os.path.exists("chunks.npy"):
    print("Loading saved FAISS index and chunks...")
    index = faiss.read_index("faiss_index.bin")
    chunks = np.load("chunks.npy", allow_pickle=True).tolist()
    print(f"Loaded {len(chunks)} chunks from saved files.")

@app.get("/")
def home():
    return {"message": "Backend is working!"}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global chunks, chunk_embeddings, index

    pdf_text = ""
    reader = PyPDF2.PdfReader(file.file)

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text

    if not pdf_text:
        return {"error": "PDF contains no text."}

    # Split into chunks (500 words each)
    words = pdf_text.split()
    chunks = [" ".join(words[i:i+500]) for i in range(0, len(words), 500)]

    # Create embeddings
    chunk_embeddings = embedding_model.encode(chunks)

    # Build FAISS index
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings))

    # Save index and chunks to disk
    faiss.write_index(index, "faiss_index.bin")
    np.save("chunks.npy", np.array(chunks))

    return {"message": f"PDF processed and indexed successfully ({len(chunks)} chunks)."}

@app.post("/chat/")
async def chat(question: str):
    global chunks, chunk_embeddings, index

    if not chunks or index is None:
        return {"error": "No PDF uploaded or index available."}

    # Embed question
    question_embedding = embedding_model.encode([question])

    # Search top 3 relevant chunks
    distances, indices = index.search(np.array(question_embedding), k=3)

    relevant_chunks = "\n\n".join([chunks[i] for i in indices[0]])

    prompt = f"""
Use ONLY the following context to answer the question.

Context:
{relevant_chunks}

Question:
{question}
"""

    # Call LLM API
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()
    return {"answer": result.get("response", "No response from model.")}