# 🤖 AI Chatbot with RAG + Agent

A smart AI chatbot built with **FastAPI**, **Groq LLM**, and **RAG (Retrieval-Augmented Generation)**. It can answer questions from your uploaded documents, search the web, and chat normally — all automatically!

---

## ✨ Features

- 📄 **RAG Mode** — Upload PDFs/docs and ask questions from them
- 🌐 **Web Search Mode** — Auto-detects queries needing live info (news, today, latest)
- 🧠 **Normal Chat** — Smart conversation with message history
- ⚡ **Streaming Responses** — Real-time word-by-word output
- 🗄️ **Chat History** — Saved to SQLite database
- 🔍 **ChromaDB Vector Store** — For fast semantic document search

---

## 🗂️ Project Structure

```
├── app_rag.py              # Main FastAPI app (routes, chat logic)
├── llm.py                  # Groq LLM integration
├── ai_agent.py             # AI Agent with tools
├── agent_tools.py          # Agent tool definitions
├── vector_store.py         # ChromaDB vector store
├── document_processor.py   # File chunking & processing
├── db.py                   # SQLite chat history
├── chatbot-interface-agent.html  # Frontend UI
├── documents-manager.html  # Document manager UI
├── requirements.txt        # Python dependencies
└── uploaded_files/         # Uploaded documents (auto-created)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Groq API Key

Create a `.env` file in the root folder:

```
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free API key at: https://console.groq.com

### 5. Run the server

```bash
python main.py
```

Open your browser: **http://localhost:8000**

---

## 🚀 How It Works

The chatbot **automatically decides** which mode to use based on your message:

| Your Message Contains                | Mode Used                    |
| ------------------------------------ | ---------------------------- |
| "latest", "news", "today", "current" | 🌐 Web Search (Agent)        |
| "document", "file", "pdf", "my data" | 📄 RAG (your uploaded files) |
| Anything else                        | 🧠 Normal Chat               |

---

## 📤 Uploading Documents

1. Go to **http://localhost:8000**
2. Use the upload button to upload PDF, TXT, or DOCX files
3. Once uploaded, ask questions like: _"Summarize my document"_ or _"What does my PDF say about X?"_

---

## 🔗 API Endpoints

| Method | Endpoint       | Description               |
| ------ | -------------- | ------------------------- |
| GET    | `/`            | Frontend UI               |
| GET    | `/health`      | Health check              |
| POST   | `/chat`        | Chat (returns full reply) |
| POST   | `/chat-stream` | Chat with streaming       |
| POST   | `/upload`      | Upload a document         |
| GET    | `/documents`   | List uploaded documents   |

---

## 🧰 Tech Stack

- **FastAPI** — Backend API framework
- **Groq** — Fast LLM inference (Llama 3)
- **ChromaDB** — Vector database for RAG
- **SQLite** — Chat history storage
- **HTML/JS** — Frontend UI

---

## 📋 Requirements

See `requirements.txt`. Main packages:

```
fastapi
uvicorn
groq
chromadb
python-dotenv
pydantic
```

---

## 🙋 FAQ

**Q: Is an internet connection required?**
A: Only for web search queries and the Groq API. RAG and normal chat work with just the API key.

**Q: What file types can I upload?**
A: Check supported types in `document_processor.py` (PDF, TXT, DOCX, etc.)

**Q: Where is chat history stored?**
A: In `chat_history.db` (SQLite, auto-created locally).

---

## 📄 License

MIT License — free to use and modify.

---

## 🙌 Author

Made with ❤️ by **[Mohammad Faizan]**

> ⭐ If you found this helpful, give it a star on GitHub!
