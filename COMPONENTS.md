# System Components and Relationships

## 1. Orchestrator Agent
- **File:** `src/agents/orchestrator.py`
- **Role:** Classifies user queries into HR, Courses, IT, or Other using a prompt-based LLM chain.
- **Input:** User question
- **Output:** Category label (HR, Courses, IT, Other)
- **Dependency:** OpenAI LLM (via LangChain)

## 2. Specialized RAG Agents
- **Files:**
  - HR: `src/agents/hr_agent.py`
  - Courses: `src/agents/courses_agent.py`
  - IT: `src/agents/it_agent.py`
- **Role:** Each agent retrieves relevant documents from its domain vector store and generates an answer using OpenAI LLM.
- **Input:** User question (routed by orchestrator)
- **Output:** Answer + source documents
- **Dependency:** Chroma vector store, OpenAI LLM

## 3. Vector Store & Document Loader
- **File:** `src/vector_store.py`
- **Role:** Loads, chunks, and indexes domain documents using LangChain loaders and Chroma. Provides retrievers for RAG agents.
- **Input:** Domain name (hr, courses, it)
- **Output:** Vector store retriever
- **Dependency:** Chroma, LangChain, OpenAI Embeddings

## 4. Multi-Agent System Entrypoint
- **File:** `src/multi_agent_system.py`
- **Role:** Main script that receives user queries, calls orchestrator, routes to correct agent, and returns the answer.
- **Input:** User question
- **Output:** JSON with answer, sources, and trace
- **Dependency:** All agents

## 5. Data Folders
- **Folders:** `data/hr_docs/`, `data/courses_docs/`, `data/it_docs/`
- **Role:** Store domain-specific documents for retrieval.
- **Format:** Plain text files (`.txt`)

## 6. Configuration & Environment
- **Files:** `requirements.txt`, `.env.example`
- **Role:** Manage dependencies and API keys for OpenAI and (optionally) Langfuse.

## Relationships
- The orchestrator agent determines which RAG agent to use.
- Each RAG agent uses its own vector store for retrieval.
- All agents rely on OpenAI LLM for classification and answer generation.
- The entrypoint script ties all components together for end-to-end execution.
