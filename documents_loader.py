import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents():
    loader = DirectoryLoader("documents", glob="manual.txt")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    return text_splitter.split_documents(docs)
