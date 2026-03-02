# 🤖 AI PDF Chatbot

Ask questions and get answers directly from PDFs using AI — perfect for research, study, or quick document search.

---

## 🚀 How it Works

1. Upload a PDF → text is split into small chunks (~500 words).  
2. Convert chunks into **embeddings** using `SentenceTransformer`.  
3. Store embeddings in **FAISS** for fast similarity search.  
4. User asks a question → FAISS finds relevant chunks → **LLaMA / Ollama** generates concise answer.  
5. Answer is displayed in a **chat-style interface** on the website.

---

## 🛠 Tech Stack

- **Python** (backend + logic)  
- **Streamlit** (frontend)  
- **SentenceTransformers** (embedding generation)  
- **FAISS** (vector search for relevant chunks)  
- **LLaMA / Ollama 3** (AI answer generation)

---

## 🎯 Features

- Upload any PDF (up to 200MB)  
- Ask multiple questions without re-uploading  
- Chat-style conversation for clarity  
- Optional: Download Q&A as TXT  

---

## 🖥 Live Demo

[Streamlit Demo](https://ai-pdf-chatbot-seqnslke6vdkieeemjampe.streamlit.app/)

---

## 📂 Installation (Local)

1. Clone the repo:

```bash
git clone https://github.com/Suman2005-bit/ai-pdf-chatbot.git
cd ai-pdf-chatbot
