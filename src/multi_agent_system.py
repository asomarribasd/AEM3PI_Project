"""
Main multi-agent orchestration system: classifies and routes queries to the correct RAG agent.
"""
from src.agents.orchestrator import classify_query
from src.agents.hr_agent import answer_hr_query
from src.agents.courses_agent import answer_courses_query
from src.agents.it_agent import answer_it_query


def route_query(question: str) -> dict:
    """Classify and route the query to the correct agent."""
    category = classify_query(question)
    trace = {"question": question, "category": category}
    if category == "HR":
        result = answer_hr_query(question)
    elif category == "Courses":
        result = answer_courses_query(question)
    elif category == "IT":
        result = answer_it_query(question)
    else:
        result = {"result": "Sorry, I could not classify your question into HR, Courses, or IT."}
    trace["result"] = result
    return trace

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.multi_agent_system 'Your question here'")
        exit(1)
    question = sys.argv[1]
    response = route_query(question)
    import json
    print(json.dumps(response, indent=2, ensure_ascii=False))
