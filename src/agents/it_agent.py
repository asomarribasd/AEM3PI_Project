"""
IT RAG Agent: Answers IT-related queries using IT vector store and OpenAI LLM.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.vector_store import get_vector_store

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = get_vector_store("it").as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

it_qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def answer_it_query(question: str) -> dict:
    """Answer an IT question with sources."""
    answer = it_qa_chain.invoke(question)
    source_docs = retriever.invoke(question)
    return {
        "result": answer,
        "source_documents": source_docs
    }

