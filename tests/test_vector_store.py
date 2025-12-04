"""
Tests for vector store functionality
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.vector_store import load_documents, build_vector_store, get_vector_store, DATA_DIRS


class TestVectorStore:
    """Test vector store document loading and retrieval"""
    
    def test_data_dirs_exist(self):
        """Verify all data directories are properly configured"""
        for domain, path in DATA_DIRS.items():
            assert os.path.exists(path), f"Data directory for {domain} does not exist: {path}"
    
    def test_load_documents_hr(self):
        """Test loading HR documents"""
        docs = load_documents("hr")
        assert len(docs) > 0, "HR documents should be loaded"
        assert all(hasattr(doc, 'page_content') for doc in docs), "All docs should have page_content"
        assert all(hasattr(doc, 'metadata') for doc in docs), "All docs should have metadata"
    
    def test_load_documents_courses(self):
        """Test loading Courses documents"""
        docs = load_documents("courses")
        assert len(docs) > 0, "Courses documents should be loaded"
    
    def test_load_documents_it(self):
        """Test loading IT documents"""
        docs = load_documents("it")
        assert len(docs) > 0, "IT documents should be loaded"
    
    def test_load_documents_chunks(self):
        """Verify documents are properly chunked"""
        docs = load_documents("hr")
        # Check that we have sufficient chunks (at least 30 chunks)
        assert len(docs) >= 30, f"Expected at least 30 chunks, got {len(docs)}"
        
        # Verify chunk sizes are reasonable
        for doc in docs[:5]:  # Check first 5
            assert len(doc.page_content) > 0, "Chunks should not be empty"
            assert len(doc.page_content) <= 1000, "Chunks should not be too large"
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_build_vector_store_integration(self):
        """Integration test for building vector store (requires API key)"""
        # This test is skipped if no API key is present
        vectordb = build_vector_store("hr", persist=False)
        assert vectordb is not None
        
        # Test retrieval
        results = vectordb.similarity_search("vacation policy", k=3)
        assert len(results) > 0, "Should retrieve documents"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
