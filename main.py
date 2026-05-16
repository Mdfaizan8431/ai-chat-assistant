from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import requests
from db import init_db, save_message, get_last_messages
from datetime import datetime
import shutil
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# RAG + Groq
from llm import stream_llm_response, get_llm_response
from document_processor import process_file, get_supported_extensions
from ai_agent import create_agent

app = FastAPI(title="AI Chatbot with RAG", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init
init_db()
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_files")
os.makedirs(UPLOAD_DIR, exist_ok=True)

vector_store = None

# ─────────────────────────────────────────────
# KEY CHANGE: Only load vector store when needed
# ─────────────────────────────────────────────
def get_or_init_vector_store():
    global vector_store
    if vector_store is None:
        from vector_store import get_vector_store
        vector_store = get_vector_store()
    return vector_store


system_prompt = """You are a helpful AI assistant.

Rules:
1. If you are not sure about something, say:
   "I'm not certain about this. Please verify."

2. If asked about events after early 2024, say:
   "This might be after my training data.
   Let me search for current information."

3. Never make up information you don't know.

4. MCP = Model Context Protocol by Anthropic (2024)
   - Connects AI models to external tools
   - Like a universal standard for AI integrations
"""

class Message(BaseModel):
    text: str

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        html_path = os.path.join(BASE_DIR, "chatbot-interface-agent.html")
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            "<h1>❌ chatbot-interface-agent.html not found</h1>",
            status_code=404
        )

@app.get("/documents-manager", response_class=HTMLResponse)
async def serve_documents_manager():
    try:
        html_path = os.path.join(BASE_DIR, "documents-manager.html")
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            "<h1>❌ documents-manager.html not found</h1>",
            status_code=404
        )

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "groq": "connected"
    }
@app.get("/history")
def get_history():
    """Get recent chat messages for sidebar"""
    try:
        messages = get_last_messages(limit=20)
        return {"messages": [
            {"role": role, "content": content}
            for role, content in messages
        ]}
    except Exception as e:
        return {"messages": []}

@app.delete("/history")
def clear_history():
    """Clear chat history"""
    try:
        from db import clear_history as db_clear
        db_clear()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
# ─────────────────────────────────────────────
# KEY CHANGE: /documents now returns safely
# without loading vector store if no files exist
# ─────────────────────────────────────────────
@app.get("/documents")
def list_documents():
    uploaded_files = os.listdir(UPLOAD_DIR)
    
    # Only load vector store if files exist
    if len(uploaded_files) > 0:
        try:
            vs = get_or_init_vector_store()
            total_chunks = vs.get_stats()['total_chunks']
        except Exception:
            total_chunks = 0
    else:
        total_chunks = 0

    return {
        "total_chunks": total_chunks,
        "uploaded_files": uploaded_files
    }

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        ext = file.filename.lower().split('.')[-1]
        if ext not in get_supported_extensions():
            raise HTTPException(status_code=400, detail="Unsupported file")

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        result = process_file(file_path, file.filename)

        vs = get_or_init_vector_store()
        vs.add_documents(result['chunks'], result['metadatas'])

        return {
            "success": True,
            "filename": file.filename,
            "chunks_added": result['total_chunks']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────
# RAG FUNCTION
# ─────────────────────────────────────────────
def rag_chat(query):
    try:
        vs = get_or_init_vector_store()
        results = vs.search(query, n_results=3)

        if not results:
            return {"answer": "No relevant information found.", "sources": []}

        context = "\n\n".join([r['text'][:500] for r in results])
        sources = list(set([r['metadata'].get("source", "Unknown") for r in results]))

        prompt = f"""You are a strict AI assistant.
Answer ONLY from the context below.
If answer is not found, say exactly: "I don't know".

Context:
{context}

Question:
{query}

Answer:"""

        answer = "".join([chunk for chunk in stream_llm_response(prompt)])
        return {"answer": answer, "sources": sources}

    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": []}

# ─────────────────────────────────────────────
# MAIN CHAT
# ─────────────────────────────────────────────
@app.post("/chat")
async def chat(msg: Message):
    user_text = msg.text.strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="Empty message")

    save_message("User", user_text)
    query_lower = user_text.lower()

    use_web = any(word in query_lower for word in
        ["latest", "news", "today", "current",
        "what is", "explain", "tell me about",
        "mcp", "protocol", "2024", "2025"])
    use_rag = len(os.listdir(UPLOAD_DIR)) > 0 and any(
        word in query_lower for word in ["document", "file", "pdf", "my data"]
    )

    try:
        if use_web:
            print("🌐 AUTO → WEB SEARCH")
            vs = get_or_init_vector_store()
            agent = create_agent(get_llm_response, vector_store=vs)
            result = agent.run(user_text)
            reply = result["response"]
            save_message("Assistant", reply)
            return {"reply": reply}

        elif use_rag:
            print("📄 AUTO → RAG")
            result = rag_chat(user_text)
            reply = result["answer"]
            save_message("Assistant", reply)
            return {"reply": reply}

        else:
            print("🧠 AUTO → NORMAL LLM")
            history = system_prompt + "\n\n"
            for role, content in get_last_messages(limit=4):
                history += f"{role}: {content}\n"
            prompt = history + f"User: {user_text}\nAssistant:"
            reply = "".join([chunk for chunk in stream_llm_response(prompt)])
            save_message("Assistant", reply)
            return {"reply": reply}

    except Exception as e:
        return {"reply": f"Error: {str(e)}"}

# ─────────────────────────────────────────────
# STREAM CHAT
# ─────────────────────────────────────────────
import time

@app.post("/chat-stream")
def chat_stream(msg: Message):

    def generator():
        try:
            user_text = msg.text.strip()

            if not user_text:
                yield "Error: Empty message"
                return

            query_lower = user_text.lower()

            use_web = any(word in query_lower for word in ["latest", "news", "today", "current", "price"])
            use_rag = len(os.listdir(UPLOAD_DIR)) > 0 and any(
                word in query_lower for word in ["document", "file", "pdf", "my data"]
            )

            if use_web:
                print("🌐 STREAM → WEB SEARCH")
                vs = get_or_init_vector_store()
                agent = create_agent(get_llm_response, vector_store=vs)
                result = agent.run(user_text)
                reply = result["response"]
                for word in reply.split():
                    yield word + " "
                    time.sleep(0.02)
                return

            elif use_rag:
                print("📄 STREAM → RAG")
                result = rag_chat(user_text)
                reply = result["answer"]
                for word in reply.split():
                    yield word + " "
                    time.sleep(0.02)
                return

            else:
                print("🧠 STREAM → NORMAL LLM")
                history = system_prompt + "\n\n"
                for role, content in get_last_messages(limit=6):
                    history += f"{role}: {content}\n"
                prompt = history + f"User: {user_text}\nAssistant:"
                for chunk in stream_llm_response(prompt):
                    yield chunk

        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(
    generator(),
    media_type="text/plain",
    headers={
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
        "Transfer-Encoding": "chunked"
    }
)

# ─────────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("🚀 Running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)