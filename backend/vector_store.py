from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from .documents_loader import load_documents

_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:

        docs = load_documents()

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={'normalize_embeddings': True}
        )

        milvus_connection_args = {
            "host": "localhost",
            "port": "19530"
        }

        _vector_store = Milvus.from_documents(
            documents=docs,
            embedding=embeddings,
            connection_args=milvus_connection_args,
            collection_name="classic_vector_collection",

        )
        
    return _vector_store
