"""
Orchestrator agent: Classifies user queries into HR, Courses, IT, or Other using LangChain.
"""
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

CATEGORIES = ["HR", "Courses", "IT", "Other"]

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert support query classifier for an educational company. Classify the following user question into one of these categories: HR, Courses, IT, or Other. Only output the category name."),
    ("user", "{question}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
output_parser = StrOutputParser()

classifier_chain = CLASSIFY_PROMPT | llm | output_parser

def classify_query(question: str) -> str:
    """Classify a user query and return the category."""
    category = classifier_chain.invoke({"question": question}).strip()
    if category not in CATEGORIES:
        return "Other"
    return category
