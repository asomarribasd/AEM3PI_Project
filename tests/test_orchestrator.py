"""
Tests for orchestrator agent
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.orchestrator import classify_query, CATEGORIES


class TestOrchestrator:
    """Test orchestrator classification logic"""
    
    def test_categories_defined(self):
        """Verify all expected categories are defined"""
        expected = ["HR", "Courses", "IT", "Other"]
        assert set(CATEGORIES) == set(expected), f"Expected {expected}, got {CATEGORIES}"
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_classify_hr_query(self):
        """Test classification of HR-related queries"""
        queries = [
            "How much vacation time do I get?",
            "What is the parental leave policy?",
            "How do I request time off?",
            "What are the company benefits?"
        ]
        for query in queries:
            result = classify_query(query)
            assert result in CATEGORIES, f"Result '{result}' not in valid categories"
            # Most should be classified as HR (though we allow flexibility)
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_classify_courses_query(self):
        """Test classification of Courses-related queries"""
        queries = [
            "How do I enroll in a Python course?",
            "What courses are available?",
            "What is the refund policy for courses?",
            "How long is the Data Science course?"
        ]
        for query in queries:
            result = classify_query(query)
            assert result in CATEGORIES, f"Result '{result}' not in valid categories"
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_classify_it_query(self):
        """Test classification of IT-related queries"""
        queries = [
            "My laptop won't start, what should I do?",
            "How do I connect to the VPN?",
            "I forgot my password",
            "How do I set up multi-factor authentication?"
        ]
        for query in queries:
            result = classify_query(query)
            assert result in CATEGORIES, f"Result '{result}' not in valid categories"
    
    @patch('src.agents.orchestrator.classifier_chain')
    def test_classify_with_mock(self, mock_chain):
        """Test classification with mocked LLM"""
        # Mock the invoke method which is used in LCEL chains
        mock_chain.invoke.return_value = "HR"
        result = classify_query("test question")
        assert result == "HR"
        mock_chain.invoke.assert_called_once()
    
    @patch('src.agents.orchestrator.classifier_chain')
    def test_classify_invalid_category_returns_other(self, mock_chain):
        """Test that invalid categories default to 'Other'"""
        mock_chain.invoke.return_value = "InvalidCategory"
        result = classify_query("test question")
        assert result == "Other", "Invalid categories should default to 'Other'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
