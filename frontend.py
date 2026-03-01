import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI PDF Chatbot", layout="centered")

st.title("📄 AI PDF Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{BACKEND_URL}/upload-pdf/", files=files)

    if response.status_code == 200:
        st.success("✅ PDF Processed Successfully!")
    else:
        st.error("❌ Upload Failed")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("Ask a question about your PDF"):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{BACKEND_URL}/chat/",
                params={"question": question}
            )

            if response.status_code == 200:
                answer = response.json()["answer"]
            else:
                answer = "Error getting response."

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})