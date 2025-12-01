"""
Courses RAG Agent: Answers course-related queries using Courses vector store and OpenAI LLM.
"""
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from src.vector_store import get_vector_store

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = get_vector_store("courses").as_retriever()

courses_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

def answer_courses_query(question: str) -> dict:
    """Answer a Courses question with sources."""
    return courses_qa_chain({"query": question})
