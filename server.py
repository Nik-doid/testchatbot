from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.agent import chatbot_agent 

app = FastAPI(
    title="RAG Chatbot API",
    description="An API for querying a Retrieval-Augmented Generation chatbot using LangChain and Hugging Face LLMs.",
    version="1.0.0"
)

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and Response Schemas
class QueryRequest(BaseModel):
    query: str
    session_id: str

class QueryResponse(BaseModel):
    response: str
    

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    try:
        answer = chatbot_agent(request.query, session_id=request.session_id)  
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
