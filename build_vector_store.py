"""
Build vector stores for all domains (HR, Courses, IT).
Run this script once to initialize the vector stores before using the multi-agent system.
"""
import os
from dotenv import load_dotenv
from src.vector_store import build_vector_store

# Load environment variables
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found in environment variables.")
    print("Please set it in your .env file or export it in your shell.")
    exit(1)

print("Building vector stores for all domains...")
print("This may take a few minutes depending on document size.\n")

domains = ["hr", "courses", "it"]

for domain in domains:
    print(f"Building vector store for {domain.upper()}...")
    try:
        vectordb = build_vector_store(domain)
        # Get collection size
        collection_size = vectordb._collection.count()
        print(f"✓ {domain.upper()} vector store created with {collection_size} chunks\n")
    except Exception as e:
        print(f"✗ Error building {domain.upper()} vector store: {e}\n")
        exit(1)

print("All vector stores built successfully!")
print("You can now run queries using: python -m src.multi_agent_system \"Your question here\"")