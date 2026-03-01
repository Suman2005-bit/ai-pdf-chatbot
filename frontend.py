import streamlit as st
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
# import your LLaMA / Ollama code here

st.title("AI PDF Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Read PDF text
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    st.write("PDF loaded successfully!")

    # Split text into chunks (~500 words)
    def split_text(text, chunk_size=500):
        words = text.split()
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    chunks = split_text(text)
    
    # Create embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Ask questions
    question = st.text_input("Ask a question about your PDF:")

    if question:
        question_embedding = model.encode([question])
        k = 3  # top 3 relevant chunks
        D, I = index.search(np.array(question_embedding), k)
        relevant_chunks = [chunks[i] for i in I[0]]

        st.write("Relevant info from PDF:")
        for c in relevant_chunks:
            st.write("-", c)

        # Here you can call your LLaMA / Ollama function to generate final answer
        # answer = call_ollama(relevant_chunks, question)
        # st.write("Answer:", answer)
