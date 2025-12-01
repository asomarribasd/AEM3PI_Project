"""
Orchestrator agent: Classifies user queries into HR, Courses, IT, or Other using LangChain LLMChain.
"""
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

CATEGORIES = ["HR", "Courses", "IT", "Other"]

CLASSIFY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
You are an expert support query classifier for an educational company. Classify the following user question into one of these categories: HR, Courses, IT, or Other. Only output the category name.

Question: {question}
Category:
"""
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

classifier_chain = LLMChain(
    llm=llm,
    prompt=CLASSIFY_PROMPT,
    output_key="category"
)

def classify_query(question: str) -> str:
    """Classify a user query and return the category."""
    result = classifier_chain({"question": question})
    category = result["category"].strip()
    if category not in CATEGORIES:
        return "Other"
    return category
