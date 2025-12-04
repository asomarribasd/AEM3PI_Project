"""
Test configuration and fixtures
"""
import os
import sys
import pytest

# Add src to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def sample_hr_question():
    """Sample HR question for testing"""
    return "How many vacation days do employees get?"


@pytest.fixture
def sample_courses_question():
    """Sample Courses question for testing"""
    return "How do I enroll in a Python course?"


@pytest.fixture
def sample_it_question():
    """Sample IT question for testing"""
    return "How do I connect to the VPN?"


@pytest.fixture
def sample_other_question():
    """Sample ambiguous question for testing"""
    return "What is the meaning of life?"


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    return {
        "result": "This is a test answer from the system.",
        "source_documents": []
    }


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires API keys)"
    )
