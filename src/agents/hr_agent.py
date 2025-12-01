"""
HR RAG Agent: Answers HR-related queries using HR vector store and OpenAI LLM.
"""
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from src.vector_store import get_vector_store

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = get_vector_store("hr").as_retriever()

hr_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

def answer_hr_query(question: str) -> dict:
    """Answer an HR question with sources."""
    return hr_qa_chain({"query": question})
