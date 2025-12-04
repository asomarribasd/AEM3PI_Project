"""
Vector store and document loader setup for HR, Courses, and IT domains using LangChain and Chroma.
"""
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

DATA_DIRS = {
    "hr": os.path.join(os.path.dirname(__file__), "..", "data", "hr_docs"),
    "courses": os.path.join(os.path.dirname(__file__), "..", "data", "courses_docs"),
    "it": os.path.join(os.path.dirname(__file__), "..", "data", "it_docs"),
}

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", ".chroma_store")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(domain: str):
    """Load and chunk all documents for a given domain."""
    docs = []
    data_dir = DATA_DIRS[domain]
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for fname in os.listdir(data_dir):
        if fname.endswith(".txt"):
            loader = TextLoader(os.path.join(data_dir, fname), encoding="utf-8")
            file_docs = loader.load()
            docs.extend(splitter.split_documents(file_docs))
    return docs


def build_vector_store(domain: str, persist: bool = True):
    """Build or load a Chroma vector store for a domain."""
    docs = load_documents(domain)
    embeddings = OpenAIEmbeddings()
    persist_dir = os.path.join(CHROMA_DIR, domain)
    os.makedirs(persist_dir, exist_ok=True)
    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    # Note: Chroma auto-persists when persist_directory is set
    return vectordb


def get_vector_store(domain: str):
    """Load an existing Chroma vector store for a domain."""
    embeddings = OpenAIEmbeddings()
    persist_dir = os.path.join(CHROMA_DIR, domain)
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
