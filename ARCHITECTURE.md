# System Architecture

## Overview
This project implements a multi-agent orchestration system for support queries in an imaginary educational company. The system uses LangChain for workflow, OpenAI for LLMs, and Chroma for vector search.

## Architecture Diagram

```
+-------------------+         +-------------------+         +-------------------+
|                   |         |                   |         |                   |
|   User Query      +-------->+   Orchestrator    +-------->+   Specialized     |
|   (CLI/Script)    |         |   Agent           |         |   RAG Agent       |
|                   |         |  (Intent Class.)  |         |  (HR/Courses/IT)  |
+-------------------+         +-------------------+         +-------------------+
                                                                |
                                                                v
                                                    +-------------------------+
                                                    |  Vector Store (Chroma)  |
                                                    |  + Domain Documents     |
                                                    +-------------------------+
                                                                |
                                                                v
                                                    +-------------------------+
                                                    |   OpenAI LLM (RAG)      |
                                                    +-------------------------+
                                                                |
                                                                v
                                                    +-------------------------+
                                                    |   Final Answer +        |
                                                    |   Source Documents      |
                                                    +-------------------------+
```

## System Flow
1. **User submits a query** via CLI or script.
2. **Orchestrator agent** classifies the query as HR, Courses, IT, or Other using an LLM.
3. The query is **routed to the appropriate RAG agent** (HR, Courses, or IT).
4. The RAG agent retrieves relevant documents from its **domain-specific vector store** (Chroma).
5. The agent uses **OpenAI LLM** to generate a grounded answer, citing sources.
6. The system returns the **final answer and sources** to the user.

## Extensibility
- Add new domains by creating new document folders, vector stores, and agent modules.
- Update the orchestrator to recognize new categories.
