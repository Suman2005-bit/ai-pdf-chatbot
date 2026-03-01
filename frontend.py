import streamlit as st
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
# import your LLaMA/Ollama function if available

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")
st.title("AI PDF Chatbot")
st.write("Upload a PDF and ask questions about its content!")

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Read PDF text
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    if not text.strip():
        st.error("No text could be extracted from this PDF.")
        st.stop()

    st.success("PDF loaded successfully!")

    # -----------------------------
    # Split text into chunks
    # -----------------------------
    def split_text(text, chunk_size=500):
        words = text.split()
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    chunks = split_text(text)
    st.write(f"PDF split into {len(chunks)} chunks.")

    # -----------------------------
    # Generate embeddings
    # -----------------------------
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_numpy=True)
    embeddings = np.array(embeddings)
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)  # ensure 2D

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    st.write("Embeddings generated and FAISS index built.")

    # -----------------------------
    # Ask questions
    # -----------------------------
    question = st.text_input("Ask a question about the PDF:")

    if question:
        question_embedding = model.encode([question], convert_to_numpy=True)
        if len(question_embedding.shape) == 1:
            question_embedding = question_embedding.reshape(1, -1)

        k = min(3, len(chunks))  # top-k relevant chunks
        D, I = index.search(question_embedding, k)
        relevant_chunks = [chunks[i] for i in I[0]]

        st.subheader("Relevant text from PDF:")
        for c in relevant_chunks:
            st.write("-", c)

        # -----------------------------
        # Optional: Call LLaMA/Ollama here to generate final answer
        # -----------------------------
        # answer = call_ollama(relevant_chunks, question)
        # st.subheader("Answer from AI:")
        # st.write(answer)
