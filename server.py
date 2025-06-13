from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from backend.agent import chatbot_agent 
from langchain_community.chat_message_histories import RedisChatMessageHistory
from dotenv import load_dotenv
import os
import redis.asyncio as redis

app = FastAPI(
    title="RAG Chatbot API",
    description="An API for querying a Retrieval-Augmented Generation chatbot using LangChain and Groq LLMs.",
    version="1.0.0"
)

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.from_url(REDIS_URL, decode_responses=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: str

class QueryResponse(BaseModel):
    response: str

@app.get("/messages")
def get_messages(session_id: str):
    try:
        history = RedisChatMessageHistory(
            url=REDIS_URL,
            session_id=session_id
        )
        messages = history.messages
        formatted = []
        for msg in messages:
            if msg.type == "human":
                formatted.append({"sender": "user", "message": msg.content})
            elif msg.type == "ai":
                formatted.append({"sender": "bot", "message": msg.content})
        return {"messages": formatted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve messages: {str(e)}")

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    try:
        answer = chatbot_agent(request.query, session_id=request.session_id)
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.delete("/clear_messages")
async def clear_messages(session_id: str = Query(...)):
    key = f"message_store:{session_id}"
    try:
        deleted = await r.delete(key)
        if deleted:
            return {"message": "Chat history cleared."}
        raise HTTPException(status_code=404, detail="Session not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear messages: {str(e)}")
