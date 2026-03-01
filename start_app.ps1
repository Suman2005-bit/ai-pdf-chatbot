# start_app.ps1
# Navigate to backend folder
cd C:\Users\kashy\OneDrive\Desktop\rag-chatbot\backend

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start FastAPI backend in a new terminal
Start-Process powershell -ArgumentList '-NoExit','-Command',"uvicorn main:app --reload --host 127.0.0.1 --port 8000"

# Start Streamlit frontend in a new terminal
Start-Process powershell -ArgumentList '-NoExit','-Command',".\venv\Scripts\streamlit.exe run frontend.py"