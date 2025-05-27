from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from .documents_loader import load_documents

_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        docs = load_documents()
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"},encode_kwargs={'normalize_embeddings': True})
        _vector_store = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./chroma_db")
    return _vector_store