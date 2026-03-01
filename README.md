# AI PDF Chatbot
Ask questions and get answers directly from PDFs using AI.

## How it works
1. Upload a PDF → text is split into small chunks (~500 words).  
2. Convert chunks into embeddings using SentenceTransformers.  
3. Store embeddings in FAISS for fast search.  
4. User asks a question → FAISS finds relevant chunks → LLaMA 3 generates answer.  
5. Answer is shown on the website.

## Live Demo
https://ai-pdf-chatbot-seqnslke6vdkieeemjampe.streamlit.app/
