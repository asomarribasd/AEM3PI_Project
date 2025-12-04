"""
Tests for specialized RAG agents
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestRAGAgents:
    """Test individual RAG agents"""
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_hr_agent_response(self):
        """Test HR agent returns proper response structure"""
        from src.agents.hr_agent import answer_hr_query
        
        result = answer_hr_query("How many vacation days do employees get?")
        
        assert "result" in result, "Response should contain 'result' key"
        assert "source_documents" in result, "Response should contain source documents"
        assert len(result["source_documents"]) > 0, "Should return at least one source document"
        assert isinstance(result["result"], str), "Result should be a string"
        assert len(result["result"]) > 10, "Result should be a meaningful answer"
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_courses_agent_response(self):
        """Test Courses agent returns proper response structure"""
        from src.agents.courses_agent import answer_courses_query
        
        result = answer_courses_query("How do I enroll in a course?")
        
        assert "result" in result, "Response should contain 'result' key"
        assert "source_documents" in result, "Response should contain source documents"
        assert len(result["source_documents"]) > 0, "Should return at least one source document"
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_it_agent_response(self):
        """Test IT agent returns proper response structure"""
        from src.agents.it_agent import answer_it_query
        
        result = answer_it_query("How do I reset my password?")
        
        assert "result" in result, "Response should contain 'result' key"
        assert "source_documents" in result, "Response should contain source documents"
        assert len(result["source_documents"]) > 0, "Should return at least one source document"
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_source_documents_have_metadata(self):
        """Verify source documents contain proper metadata"""
        from src.agents.hr_agent import answer_hr_query
        
        result = answer_hr_query("What is the remote work policy?")
        sources = result["source_documents"]
        
        for source in sources:
            assert hasattr(source, 'page_content'), "Source should have page_content"
            assert hasattr(source, 'metadata'), "Source should have metadata"
            assert 'source' in source.metadata, "Metadata should include source file"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
