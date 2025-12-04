# Test Suite Documentation

## Overview
This test suite provides comprehensive coverage for the multi-agent support system, including unit tests, integration tests, and end-to-end tests.

## Test Structure

### 1. `test_vector_store.py`
Tests for document loading and vector store functionality:
- Verifies data directories exist
- Tests document loading for all domains (HR, Courses, IT)
- Validates chunk sizes and counts (minimum 50 chunks per domain)
- Integration test for vector store building (requires API key)

### 2. `test_orchestrator.py`
Tests for the orchestrator agent:
- Validates category definitions
- Tests query classification for HR, Courses, IT queries
- Tests with mocked LLM responses
- Validates fallback to "Other" for invalid categories

### 3. `test_rag_agents.py`
Tests for specialized RAG agents:
- Tests HR, Courses, and IT agents individually
- Validates response structure (result + source_documents)
- Verifies source document metadata

### 4. `test_multi_agent_system.py`
End-to-end integration tests:
- Tests full query routing through orchestrator to RAG agents
- Tests all query types (HR, Courses, IT, Other)
- Validates response structure and source inclusion
- Tests sequential query handling

### 5. `conftest.py`
Test configuration and fixtures:
- Sample questions for each domain
- Mock response fixtures
- Custom pytest markers

## Running Tests

### Run all tests:
```bash
pytest tests/ -v
```

### Run specific test file:
```bash
pytest tests/test_vector_store.py -v
```

### Run without API key (unit tests only):
```bash
pytest tests/ -v -m "not integration"
```

### Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Requirements
- Most tests are **unit tests** and can run without API keys (using mocks)
- **Integration tests** require `OPENAI_API_KEY` environment variable
- Tests automatically skip integration tests when API key is not present

## Adding New Tests
1. Create test file in `tests/` with `test_` prefix
2. Use pytest fixtures from `conftest.py`
3. Mark integration tests that require API keys appropriately
4. Follow naming convention: `test_<functionality>_<scenario>`

## Best Practices
- Keep tests isolated and independent
- Use mocks for external dependencies when possible
- Validate both success and error cases
- Include descriptive assertion messages
