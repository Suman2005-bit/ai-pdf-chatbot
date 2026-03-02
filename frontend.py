# frontend.py
import streamlit as st
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

st.set_page_config(page_title="AI PDF Chatbot", page_icon="🤖")
st.title("🤖 AI PDF Chatbot - Chat Style (Specific Answers)")

# ----------------- Session State -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------- PDF Upload -----------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Read PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = "".join([page.extract_text() for page in pdf_reader.pages])
    st.write("✅ PDF loaded successfully!")

    # Split into ~500-word chunks
    def split_text(text, chunk_size=500):
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    chunks = split_text(text)

    # Create embeddings and FAISS index
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # ----------------- Chat Input -----------------
    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Find top 3 relevant chunks
        question_embedding = model.encode([user_question])
        D, I = index.search(np.array(question_embedding), 3)
        relevant_chunks = [chunks[i] for i in I[0]]

        # ----------------- AI Summarization Placeholder -----------------
        # Replace this function with your LLaMA/Ollama call
        def generate_concise_answer(chunks, question):
            """
            Input: chunks = list of relevant PDF text
                   question = user question
            Output: concise answer (1-2 sentences)
            """
            context = " ".join(chunks)
            prompt = f"""
            You are an AI assistant. Answer the question based on the PDF content.
            PDF Content:
            {context}

            Question: {question}

            Answer in 1-2 concise sentences, do not copy full chunks.
            """
            # ----- Replace the next line with LLaMA / Ollama call -----
            # Example placeholder: just returns first sentence of first chunk (temporary)
            return chunks[0].split(".")[0] + "."  # temporary
            # -----------------------------------------------------------

        answer = generate_concise_answer(relevant_chunks, user_question)

        # Append AI answer
        st.session_state.messages.append({"role": "bot", "content": answer})

    # ----------------- Display Chat -----------------
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**AI:** {msg['content']}")

    # ----------------- Optional: Download Chat -----------------
    if st.button("Download Chat as TXT"):
        chat_text = "\n\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        st.download_button("Download Q&A", chat_text, file_name="chat.txt")
