# Multi-Agent Support System for Educational Company

This project implements a production-grade multi-agent system for routing and answering support queries in an educational company. It uses LangChain, OpenAI, and Chroma for RAG (Retrieval-Augmented Generation), with modular agents for HR, Courses, and IT.

## Features
- **Orchestrator agent**: Classifies queries into HR, Courses, IT, or Other
- **Specialized RAG agents**: Each domain uses its own vector store and documentation
- **Vector search**: Fast, accurate retrieval from internal docs
- **Easy extensibility**: Add more domains/agents as needed

## Project Structure
```
AEM3PI_Project/
├── src/
│   ├── multi_agent_system.py
│   ├── vector_store.py
│   └── agents/
│       ├── orchestrator.py
│       ├── hr_agent.py
│       ├── courses_agent.py
│       └── it_agent.py
├── data/
│   ├── hr_docs/
│   ├── courses_docs/
│   └── it_docs/
├── requirements.txt
├── .env.example
└── README.md
```

## Setup Instructions

1. **Clone the repository**

# Create virtual environment

python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
- Copy `.env.example` to `.env` and fill in your OpenAI and Langfuse keys.

4. **Build vector stores**
Run the build script to initialize vector stores for all domains:
```bash
python build_vector_store.py
```

5. **Run the system**
```bash
python -m src.multi_agent_system "How do I request paid time off?"
```

## Usage Example
- Ask any question about HR, courses, or IT. The system will classify and route it to the correct agent, returning an answer with sources.

## Configuration Notes
- All document data is in `data/` folders.
- Vector stores are persisted in `.chroma_store/`.
- Uses OpenAI GPT-4o-mini by default (configurable in code).

## Known Limitations
- Only supports .txt files for document ingestion.
- No web UI (CLI only).
- Requires valid OpenAI API key.

## Extending the System
- Add new domain docs in `data/` and create a new agent in `src/agents/`.
- Update orchestrator prompt to include new categories.

## License
MIT
