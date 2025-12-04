"""
HR RAG Agent: Answers HR-related queries using HR vector store and OpenAI LLM.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langfuse.langchain import CallbackHandler
from src.vector_store import get_vector_store

load_dotenv()

# Initialize Langfuse callback handler (uses LANGFUSE_* env vars)
langfuse_handler = CallbackHandler()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = get_vector_store("hr").as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

hr_qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def answer_hr_query(question: str) -> dict:
    """Answer an HR question with sources."""
    answer = hr_qa_chain.invoke(
        question,
        config={"callbacks": [langfuse_handler]}
    )
    source_docs = retriever.invoke(question)
    return {
        "result": answer,
        "source_documents": source_docs
    }

