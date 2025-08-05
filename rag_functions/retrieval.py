# retrieval.py - Updated to use Gemma embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

def setup_vector_db(reference_texts, reference_meta=None):
    """Setup vector database with reference documents using Gemma embeddings"""
    # Use Ollama embeddings with a Gemma model
    embeddings = OllamaEmbeddings(
        model="gemma:2b",  # Using smaller model for embeddings
        base_url="http://localhost:11434"
    )
    vectorstore = FAISS.from_texts(reference_texts, embeddings, metadatas=reference_meta)
    return vectorstore

def retrieve_references(vectorstore, parsed, k=5):
    """Retrieve similar reference documents based on parsed content"""
    docs = vectorstore.similarity_search(str(parsed), k=k)
    return [doc.page_content for doc in docs]