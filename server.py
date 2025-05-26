from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import chatbot_agent 

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

class QueryResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Chatbot API!"}

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    """
    Accepts a user query and returns a response generated using
    the Retrieval-Augmented Generation (RAG) pipeline with fallback
    to general language model knowledge.
    """
    try:
        answer = chatbot_agent(request.query)
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
