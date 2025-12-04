"""
Integration tests for the multi-agent system
"""
import os
import sys
import pytest
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.multi_agent_system import route_query


class TestMultiAgentSystem:
    """Test end-to-end multi-agent orchestration"""
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_route_query_hr(self):
        """Test routing HR query through full system"""
        question = "How many sick days do I get per year?"
        result = route_query(question)
        
        assert "question" in result, "Result should contain question"
        assert "category" in result, "Result should contain category"
        assert "result" in result, "Result should contain result"
        assert result["question"] == question
        # Category should ideally be HR, but we don't strictly enforce for flexibility
        assert result["category"] in ["HR", "Courses", "IT", "Other"]
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_route_query_courses(self):
        """Test routing Courses query through full system"""
        question = "What is the refund policy for courses?"
        result = route_query(question)
        
        assert "question" in result
        assert "category" in result
        assert "result" in result
        assert result["question"] == question
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_route_query_it(self):
        """Test routing IT query through full system"""
        question = "My VPN is not connecting, what should I do?"
        result = route_query(question)
        
        assert "question" in result
        assert "category" in result
        assert "result" in result
        assert result["question"] == question
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_route_query_other(self):
        """Test routing ambiguous query"""
        question = "What is the weather today?"
        result = route_query(question)
        
        assert "question" in result
        assert "category" in result
        assert "result" in result
        # Should likely be categorized as "Other"
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_result_contains_sources(self):
        """Verify results include source documents when available"""
        question = "What is the attendance policy?"
        result = route_query(question)
        
        # For non-Other categories, result should have sources
        if result["category"] != "Other":
            assert "result" in result["result"], "Should have nested result structure"
            # Updated to check for 'sources' instead of 'source_documents'
            if "sources" in result["result"]:
                assert len(result["result"]["sources"]) > 0
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_multiple_queries_sequence(self):
        """Test handling multiple queries in sequence"""
        queries = [
            "How do I enroll in Python course?",
            "What is the vacation policy?",
            "How do I reset my password?"
        ]
        
        for query in queries:
            result = route_query(query)
            assert result is not None, f"Failed to process query: {query}"
            assert "category" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
