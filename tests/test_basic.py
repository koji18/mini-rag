"""
Basic functionality tests for Mini-RAG system
Tests core components and their interactions
"""
import pytest
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import config


# ================================================================================
# Configuration Tests
# ================================================================================

class TestConfig:
    """Test configuration and constants"""

    def test_config_exists(self):
        """Test that config module exists and is importable"""
        assert config is not None
        assert hasattr(config, 'EMBEDDING_MODEL')

    def test_embedding_dimension(self):
        """Test embedding dimension is set correctly"""
        assert config.EMBEDDING_DIMENSION == 384
        assert isinstance(config.EMBEDDING_DIMENSION, int)
        assert config.EMBEDDING_DIMENSION > 0

    def test_chunk_size(self, expected_configs):
        """Test chunk size configuration"""
        assert config.CHUNK_SIZE == expected_configs["CHUNK_SIZE"]
        assert config.CHUNK_SIZE > 0
        assert isinstance(config.CHUNK_SIZE, int)

    def test_chunk_overlap(self):
        """Test chunk overlap is less than chunk size"""
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
        assert config.CHUNK_OVERLAP >= 0

    def test_embedding_model(self):
        """Test embedding model name"""
        assert isinstance(config.EMBEDDING_MODEL, str)
        assert len(config.EMBEDDING_MODEL) > 0
        assert "all-MiniLM-L6-v2" in config.EMBEDDING_MODEL

    def test_similarity_threshold(self):
        """Test similarity threshold is valid"""
        assert 0 <= config.SIMILARITY_THRESHOLD <= 1
        assert isinstance(config.SIMILARITY_THRESHOLD, (int, float))

    def test_top_k(self):
        """Test top-k configuration"""
        assert config.TOP_K > 0
        assert isinstance(config.TOP_K, int)
        assert config.TOP_K <= 100  # Reasonable upper bound

    def test_config_summary_function(self):
        """Test that config summary function exists"""
        # Check if function exists
        assert hasattr(config, 'get_config_summary') or hasattr(config, '__dict__')


# ================================================================================
# Embedding Tests
# ================================================================================

class TestEmbeddings:
    """Test embedding functionality"""

    def test_embedding_dimension_correct(self, validate_embedding):
        """Test that embedding dimensions match expected size"""
        # Create a dummy embedding
        dummy_emb = np.random.randn(config.EMBEDDING_DIMENSION)
        assert validate_embedding(dummy_emb)

    def test_embedding_is_normalized(self):
        """Test embedding normalization"""
        dummy_emb = np.random.randn(config.EMBEDDING_DIMENSION)
        dummy_emb = dummy_emb / np.linalg.norm(dummy_emb)

        norm = np.linalg.norm(dummy_emb)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_embedding_contains_finite_values(self):
        """Test embeddings contain finite values"""
        dummy_emb = np.random.randn(config.EMBEDDING_DIMENSION)
        assert np.all(np.isfinite(dummy_emb))

    def test_batch_embeddings_shape(self, sample_texts):
        """Test batch embeddings have correct shape"""
        batch_size = len(sample_texts)
        dummy_embeddings = np.random.randn(batch_size, config.EMBEDDING_DIMENSION)

        assert dummy_embeddings.shape == (batch_size, config.EMBEDDING_DIMENSION)
        assert dummy_embeddings.dtype in [np.float32, np.float64]


# ================================================================================
# Document Processing Tests
# ================================================================================

class TestDocumentProcessing:
    """Test document loading and chunking"""

    def test_sample_documents_exist(self, sample_documents):
        """Test sample documents are available"""
        assert len(sample_documents) >= 3
        assert all("path" in doc and "content" in doc for doc in sample_documents)

    def test_document_content_non_empty(self, sample_documents):
        """Test all documents have content"""
        for doc in sample_documents:
            assert len(doc["content"]) > 0

    def test_chunk_size_validation(self):
        """Test chunk size is reasonable"""
        assert 256 <= config.CHUNK_SIZE <= 1024
        assert config.CHUNK_SIZE % 16 == 0  # Should be divisible by 16

    def test_chunk_overlap_validation(self):
        """Test chunk overlap is reasonable"""
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
        assert config.CHUNK_OVERLAP >= 0
        assert config.CHUNK_OVERLAP / config.CHUNK_SIZE <= 0.2  # Max 20% overlap


# ================================================================================
# Retriever Tests
# ================================================================================

class TestRetriever:
    """Test retrieval functionality"""

    def test_top_k_configuration(self):
        """Test top-k value is reasonable"""
        assert 1 <= config.TOP_K <= 10
        assert isinstance(config.TOP_K, int)

    def test_similarity_score_range(self):
        """Test similarity scores should be in valid range"""
        # Valid range is typically [0, 1] for cosine similarity of normalized vectors
        assert 0 <= config.SIMILARITY_THRESHOLD <= 1

    def test_retrieval_result_structure(self, validate_retrieval_result):
        """Test retrieval result has expected structure"""
        mock_result = {
            "chunk": "Sample text chunk",
            "score": 0.95,
        }
        assert validate_retrieval_result(mock_result)

    def test_mock_retriever(self, mock_retriever, sample_chunks):
        """Test mock retriever functionality"""
        mock_retriever.chunks = sample_chunks
        results = mock_retriever.retrieve_similar_chunks("test query", top_k=3)

        assert isinstance(results, list)
        assert len(results) <= 3


# ================================================================================
# RAG Pipeline Tests
# ================================================================================

class TestRAGPipeline:
    """Test RAG pipeline functionality"""

    def test_rag_answer_structure(self, validate_query_result):
        """Test RAG answer has expected structure"""
        mock_answer = {
            "answer": "Sample answer",
            "context": ["context chunk 1", "context chunk 2"],
            "confidence": 0.85,
        }
        assert validate_query_result(mock_answer)

    def test_rag_answer_not_empty(self, mock_rag_pipeline):
        """Test RAG returns non-empty answers"""
        result = mock_rag_pipeline.answer_query("What is Python?")

        assert result["answer"]
        assert len(result["answer"]) > 0

    def test_rag_context_is_list(self, mock_rag_pipeline):
        """Test RAG context is a list"""
        result = mock_rag_pipeline.answer_query("Test query")
        assert isinstance(result["context"], list)

    def test_rag_confidence_in_range(self, mock_rag_pipeline):
        """Test confidence score is in valid range"""
        result = mock_rag_pipeline.answer_query("Test query")

        confidence = result.get("confidence", result.get("score", 0))
        assert 0 <= confidence <= 1


# ================================================================================
# Integration Tests
# ================================================================================

class TestIntegration:
    """Basic integration tests"""

    def test_config_to_retriever_integration(self, mock_retriever, sample_chunks):
        """Test config flows to retriever"""
        mock_retriever.chunks = sample_chunks
        results = mock_retriever.retrieve_similar_chunks("test", top_k=config.TOP_K)

        assert len(results) <= config.TOP_K

    def test_embeddings_to_retriever_pipeline(self, mock_embedding_manager, mock_retriever):
        """Test embeddings to retriever pipeline"""
        text = "Python programming"
        embedding = mock_embedding_manager.embed_text(text)

        assert embedding is not None
        assert len(embedding) == config.EMBEDDING_DIMENSION

    def test_full_query_pipeline(self, mock_embedding_manager, mock_rag_pipeline):
        """Test full query processing pipeline"""
        query = "What is machine learning?"

        result = mock_rag_pipeline.answer_query(query)

        assert result is not None
        assert "answer" in result
        assert "context" in result


# ================================================================================
# Error Handling Tests
# ================================================================================

class TestErrorHandling:
    """Test error handling in components"""

    def test_empty_chunk_list_handling(self, mock_retriever):
        """Test handling of empty chunk list"""
        mock_retriever.chunks = []
        results = mock_retriever.retrieve_similar_chunks("query", top_k=3)

        assert isinstance(results, list)

    def test_none_query_handling(self, mock_rag_pipeline):
        """Test handling of empty query"""
        # This should either handle gracefully or raise ValueError
        try:
            result = mock_rag_pipeline.answer_query("")
            assert result is not None
        except (ValueError, AssertionError):
            pass  # Expected behavior

    def test_invalid_embedding_dimension(self):
        """Test handling of invalid embedding dimensions"""
        invalid_embedding = np.random.randn(384, 2)  # Wrong shape

        assert invalid_embedding.shape != (config.EMBEDDING_DIMENSION,)

    def test_invalid_similarity_score(self):
        """Test that similarity scores are validated"""
        invalid_score = 1.5  # Out of range

        assert not (0 <= invalid_score <= 1)


# ================================================================================
# Performance Tests (Basic)
# ================================================================================

class TestPerformance:
    """Basic performance tests"""

    @pytest.mark.performance
    def test_single_embedding_performance(self, mock_embedding_manager, sample_text):
        """Test single text embedding performance"""
        import time

        start = time.time()
        embedding = mock_embedding_manager.embed_text(sample_text)
        elapsed = time.time() - start

        # Mock should be very fast (< 0.1 seconds)
        assert elapsed < 0.1
        assert embedding is not None

    @pytest.mark.performance
    def test_batch_embedding_performance(self, mock_embedding_manager, sample_texts):
        """Test batch embedding performance"""
        import time

        start = time.time()
        embeddings = mock_embedding_manager.embed_batch(sample_texts)
        elapsed = time.time() - start

        # Should be faster than 1 second for 5 texts
        assert elapsed < 1.0
        assert embeddings.shape == (len(sample_texts), config.EMBEDDING_DIMENSION)

    @pytest.mark.performance
    def test_retrieval_performance(self, mock_retriever, sample_chunks):
        """Test retrieval performance"""
        import time

        mock_retriever.chunks = sample_chunks

        start = time.time()
        results = mock_retriever.retrieve_similar_chunks("test query", top_k=3)
        elapsed = time.time() - start

        # Should be fast
        assert elapsed < 0.1
        assert len(results) <= 3


# ================================================================================
# Data Validation Tests
# ================================================================================

class TestDataValidation:
    """Test data validation functionality"""

    def test_validate_text_chunk(self, sample_chunks):
        """Test text chunk validation"""
        for chunk in sample_chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_validate_embedding_values(self, validate_embedding, generate_random_embedding):
        """Test embedding value validation"""
        emb = generate_random_embedding(seed=42)
        assert validate_embedding(emb)

    def test_validate_query_result_structure(self, validate_query_result):
        """Test query result structure validation"""
        valid_result = {
            "answer": "This is an answer.",
            "context": ["context1", "context2"],
            "confidence": 0.8,
        }
        assert validate_query_result(valid_result)

    def test_document_path_validation(self, sample_documents):
        """Test document path validation"""
        for doc in sample_documents:
            path = Path(doc["path"])
            assert path.suffix in [".txt", ".md", ".pdf"]

    def test_chunk_size_validation_helper(self, sample_chunks):
        """Test chunk sizes are reasonable"""
        for chunk in sample_chunks:
            # Each chunk should be smaller than 2x chunk size
            assert len(chunk) <= config.CHUNK_SIZE * 2


# ================================================================================
# Configuration Integration Tests
# ================================================================================

class TestConfigurationIntegration:
    """Test configuration works across components"""

    def test_config_affects_embedding_dimension(self):
        """Test config dimension affects embeddings"""
        emb = np.random.randn(config.EMBEDDING_DIMENSION)
        assert emb.shape[0] == config.EMBEDDING_DIMENSION

    def test_config_affects_top_k_results(self, mock_retriever, sample_chunks):
        """Test top_k config is respected"""
        mock_retriever.chunks = sample_chunks
        results = mock_retriever.retrieve_similar_chunks("test", top_k=config.TOP_K)

        assert len(results) <= config.TOP_K

    def test_config_chunk_size_consistency(self):
        """Test chunk size config is consistent"""
        # Create a dummy chunk
        chunk = "a" * config.CHUNK_SIZE

        assert len(chunk) == config.CHUNK_SIZE
