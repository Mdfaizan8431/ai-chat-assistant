from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import requests
from db import init_db, save_message, get_last_messages
from datetime import datetime
import shutil
import os

# RAG + Groq
from vector_store import get_vector_store
from llm import stream_llm_response, get_llm_response  # ✅ added get_llm_response
from document_processor import process_file, get_supported_extensions

# Agent
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
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

vector_store = None

# ─────────────────────────────────────────────
# ✅ RAG FUNCTION (Groq)
# ─────────────────────────────────────────────
def rag_chat(query):
    try:
        vs = get_or_init_vector_store()
        results = vs.search(query, n_results=3)

        if not results:
            return {
                "answer": "No relevant information found.",
                "sources": []
            }

        context = "\n\n".join([r['text'][:500] for r in results])
        sources = list(set([r['metadata'].get("source", "Unknown") for r in results]))

        prompt = f"""
        You are a strict AI assistant.

        Answer ONLY from the context below.
        If answer is not found, say exactly: "I don't know".

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        # ⚠️ Streaming → convert to string
        answer = "".join([chunk for chunk in stream_llm_response(prompt)])

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }

def get_or_init_vector_store():
    global vector_store
    if vector_store is None:
        vector_store = get_vector_store()
    return vector_store


system_prompt = """You are a helpful AI assistant."""

# class Message(BaseModel):
#     text: str
#     use_rag: bool = False
#     use_agent: bool = False
class Message(BaseModel):
    text: str

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("chatbot-interface-agent.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except:
        return HTMLResponse("<h1>UI not found</h1>", status_code=404)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "groq": "connected"
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
# DOCUMENT LIST
# ─────────────────────────────────────────────
@app.get("/documents")
def list_documents():
    vs = get_or_init_vector_store()
    return {
        "total_chunks": vs.get_stats()['total_chunks'],
        "uploaded_files": os.listdir(UPLOAD_DIR)
    }


# ─────────────────────────────────────────────
# MAIN CHAT (FIXED)
# ─────────────────────────────────────────────
# @app.post("/chat")
# async def chat(msg: Message):
#     user_text = msg.text.strip()

#     if not user_text:
#         raise HTTPException(status_code=400, detail="Empty message")

#     save_message("User", user_text)

#     # 🔥 AGENT MODE (FIXED → Groq + tools)
#     if msg.use_agent:
#         try:
#             vs = get_or_init_vector_store()

#             # ✅ Use Groq LLM instead of Ollama
#             agent = create_agent(get_llm_response, vector_store=vs)

#             result = agent.run(user_text)

#             reply = result["response"]
#             save_message("Assistant", reply)

#             return {
#                 "reply": reply,
#                 "used_agent": True
#             }

#         except Exception as e:
#             return {"reply": f"Agent error: {str(e)}"}

#     # ───────── RAG MODE
#     if msg.use_rag:
#         result = rag_chat(user_text)

#         reply = result["answer"]
#         sources = result["sources"]

#         save_message("Assistant", reply)

#         return {
#             "reply": reply,
#             "used_rag": True,
#             "sources": sources
#         }

#     # ───────── NORMAL CHAT
#     try:
#         history = system_prompt + "\n\n"
#         for role, content in get_last_messages(limit=4):
#             history += f"{role}: {content}\n"

#         prompt = history + f"User: {user_text}\nAssistant:"

#         reply = "".join([chunk for chunk in stream_llm_response(prompt)])

#         save_message("Assistant", reply)

#         return {
#             "reply": reply,
#             "used_rag": False
#         }

#     except Exception as e:
#         return {"reply": f"LLM error: {str(e)}"}

@app.post("/chat")
async def chat(msg: Message):
    user_text = msg.text.strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="Empty message")

    save_message("User", user_text)

    query_lower = user_text.lower()

    # 🔥 AUTO DECISION RULES
    use_web = any(word in query_lower for word in ["latest", "news", "today", "current"])
    use_rag = len(os.listdir(UPLOAD_DIR)) > 0 and any(
        word in query_lower for word in ["document", "file", "pdf", "my data"]
    )

    try:
        # 🔥 CASE 1: WEB SEARCH (Agent)
        if use_web:
            print("🌐 AUTO → WEB SEARCH")

            vs = get_or_init_vector_store()
            agent = create_agent(get_llm_response, vector_store=vs)

            result = agent.run(user_text)
            reply = result["response"]

            save_message("Assistant", reply)

            return {"reply": reply}

        # 🔥 CASE 2: RAG
        elif use_rag:
            print("📄 AUTO → RAG")

            result = rag_chat(user_text)
            reply = result["answer"]

            save_message("Assistant", reply)

            return {"reply": reply}

        # 🔥 CASE 3: NORMAL CHAT
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
# STREAMING
# ─────────────────────────────────────────────
from fastapi.responses import StreamingResponse
import time
import os

# ─────────────────────────────────────────────
# STREAM CHAT (AUTO MODE - NO TOGGLES)
# ─────────────────────────────────────────────
@app.post("/chat-stream")
def chat_stream(msg: Message):

    def generator():
        try:
            user_text = msg.text.strip()

            if not user_text:
                yield "Error: Empty message"
                return

            query_lower = user_text.lower()

            # 🔥 AUTO DECISION
            use_web = any(word in query_lower for word in ["latest", "news", "today", "current", "price"])
            use_rag = len(os.listdir(UPLOAD_DIR)) > 0 and any(
                word in query_lower for word in ["document", "file", "pdf", "my data"]
            )

            # ─────────────────────────────
            # 🌐 WEB SEARCH (AGENT)
            # ─────────────────────────────
            if use_web:
                print("🌐 STREAM → WEB SEARCH")

                vs = get_or_init_vector_store()
                agent = create_agent(get_llm_response, vector_store=vs)

                result = agent.run(user_text)
                reply = result["response"]

                # 🔥 stream word-by-word (fake streaming)
                for word in reply.split():
                    yield word + " "
                    time.sleep(0.02)

                return

            # ─────────────────────────────
            # 📄 RAG MODE
            # ─────────────────────────────
            elif use_rag:
                print("📄 STREAM → RAG")

                result = rag_chat(user_text)
                reply = result["answer"]

                for word in reply.split():
                    yield word + " "
                    time.sleep(0.02)

                return

            # ─────────────────────────────
            # 🧠 NORMAL CHAT (REAL STREAM)
            # ─────────────────────────────
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

    return StreamingResponse(generator(), media_type="text/plain")


# ─────────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print("🚀 Running on http://localhost:8000")

    uvicorn.run(app, host="0.0.0.0", port=8000)