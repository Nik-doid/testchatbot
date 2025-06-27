import os
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
}

def load_documents():
    try:
        # Load from multiple file types
        loaders = []
        for ext, loader_class in SUPPORTED_EXTENSIONS.items():
            loaders.append(DirectoryLoader(
                "backend/documents",
                glob=f"**/*{ext}",
                loader_cls=loader_class,
                show_progress=True
            ))
        
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_documents(docs)
    
    except Exception as e:
        raise RuntimeError(f"Document loading failed: {str(e)}")