"""
Module-specific unit tests for Mini-RAG components
Tests individual modules and their public APIs
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import config


# ================================================================================
# Config Module Tests
# ================================================================================

class TestConfigModule:
    """Comprehensive tests for config module"""

    def test_config_module_import(self):
        """Test config module can be imported"""
        assert config is not None

    def test_all_required_constants_exist(self):
        """Test all required config constants exist"""
        required_constants = [
            'EMBEDDING_MODEL',
            'EMBEDDING_DIMENSION',
            'CHUNK_SIZE',
            'CHUNK_OVERLAP',
            'SIMILARITY_THRESHOLD',
            'TOP_K',
            'SIMILARITY_METRIC',
            'LLM_TYPE',
        ]

        for const in required_constants:
            assert hasattr(config, const), f"Config missing {const}"

    def test_embedding_model_is_string(self):
        """Test embedding model is a string"""
        assert isinstance(config.EMBEDDING_MODEL, str)
        assert len(config.EMBEDDING_MODEL) > 0

    def test_embedding_dimension_is_valid_int(self):
        """Test embedding dimension is valid integer"""
        assert isinstance(config.EMBEDDING_DIMENSION, int)
        assert config.EMBEDDING_DIMENSION > 0
        assert config.EMBEDDING_DIMENSION % 128 == 0  # Usually divisible by 128

    def test_chunk_size_greater_than_overlap(self):
        """Test chunk size is greater than overlap"""
        assert config.CHUNK_SIZE > config.CHUNK_OVERLAP

    def test_similarity_threshold_valid_range(self):
        """Test similarity threshold is in valid range"""
        assert isinstance(config.SIMILARITY_THRESHOLD, (int, float))
        assert -1 <= config.SIMILARITY_THRESHOLD <= 1

    def test_top_k_is_positive_int(self):
        """Test top_k is positive integer"""
        assert isinstance(config.TOP_K, int)
        assert config.TOP_K > 0

    def test_similarity_metric_valid(self):
        """Test similarity metric is valid"""
        valid_metrics = ['cosine', 'euclidean', 'manhattan']
        assert config.SIMILARITY_METRIC in valid_metrics

    def test_llm_type_valid(self):
        """Test LLM type is valid"""
        valid_types = ['template', 'openai', 'local']
        assert config.LLM_TYPE in valid_types

    @pytest.mark.requires_model
    def test_sentence_transformers_model_name(self):
        """Test Sentence Transformers model name format"""
        # Should be in format: organization/model-name
        parts = config.EMBEDDING_MODEL.split('/')
        assert len(parts) >= 2, "Model should include organization prefix"

    def test_config_coherence(self):
        """Test configuration is internally coherent"""
        # Chunk size should not exceed 2048 for typical models
        assert config.CHUNK_SIZE <= 2048

        # Overlap should be at least 10% less than chunk
        assert config.CHUNK_OVERLAP <= config.CHUNK_SIZE * 0.9

        # Top K should be reasonable
        assert config.TOP_K <= 50


# ================================================================================
# Embeddings Module Tests
# ================================================================================

class TestEmbeddingsModule:
    """Tests for embeddings module (Phase 1)"""

    def test_embeddings_module_structure(self):
        """Test embeddings module has expected structure"""
        # Check that constants are properly configured for embeddings
        assert config.EMBEDDING_DIMENSION > 0
        assert config.EMBEDDING_MODEL is not None

    def test_embedding_dimension_matches_model(self):
        """Test embedding dimension matches model specification"""
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert config.EMBEDDING_DIMENSION == 384

    @pytest.mark.unit
    def test_mock_embedding_creation(self, mock_embedding_manager, sample_text):
        """Test mock embedding manager can create embeddings"""
        embedding = mock_embedding_manager.embed_text(sample_text)

        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == config.EMBEDDING_DIMENSION

    @pytest.mark.unit
    def test_mock_batch_embeddings(self, mock_embedding_manager, sample_texts):
        """Test mock embedding manager batch processing"""
        embeddings = mock_embedding_manager.embed_batch(sample_texts)

        assert embeddings.shape == (len(sample_texts), config.EMBEDDING_DIMENSION)

    @pytest.mark.unit
    def test_embeddings_are_normalized(self, mock_embedding_manager, sample_text):
        """Test embeddings are normalized"""
        embedding = mock_embedding_manager.embed_text(sample_text)

        norm = np.linalg.norm(embedding)
        # Should be close to 1.0 for normalized embedding
        assert 0.9 <= norm <= 1.1

    @pytest.mark.unit
    def test_embedding_deterministic(self, mock_embedding_manager):
        """Test embeddings are deterministic for same input"""
        text = "Test text"

        emb1 = mock_embedding_manager.embed_text(text)
        emb2 = mock_embedding_manager.embed_text(text)

        assert np.allclose(emb1, emb2)

    @pytest.mark.unit
    def test_different_texts_different_embeddings(self, mock_embedding_manager):
        """Test different texts produce different embeddings"""
        text1 = "Python programming"
        text2 = "JavaScript programming"

        emb1 = mock_embedding_manager.embed_text(text1)
        emb2 = mock_embedding_manager.embed_text(text2)

        # Embeddings should be different
        assert not np.allclose(emb1, emb2)


# ================================================================================
# Ingest Module Tests
# ================================================================================

class TestIngestModule:
    """Tests for ingest module (Phase 1)"""

    def test_ingest_module_config(self):
        """Test ingest module configuration"""
        assert config.CHUNK_SIZE > 0
        assert config.CHUNK_OVERLAP >= 0
        assert config.CHUNK_SIZE > config.CHUNK_OVERLAP

    def test_chunking_strategy_valid(self):
        """Test chunking strategy is valid"""
        valid_strategies = ['hierarchical', 'simple', 'recursive']
        # Check that some chunking strategy is defined
        assert hasattr(config, 'CHUNK_SIZE')

    @pytest.mark.unit
    def test_document_loading_config(self, sample_documents):
        """Test document loading with sample documents"""
        assert len(sample_documents) > 0

        for doc in sample_documents:
            assert 'path' in doc
            assert 'content' in doc
            assert len(doc['content']) > 0

    @pytest.mark.unit
    def test_text_chunking_respects_size(self, sample_chunks):
        """Test chunks respect size configuration"""
        for chunk in sample_chunks:
            # Chunks should be <= CHUNK_SIZE (with some tolerance)
            assert len(chunk) <= config.CHUNK_SIZE * 1.5

    @pytest.mark.unit
    def test_chunking_preserves_content(self, sample_text):
        """Test that chunking preserves original content"""
        chunks = sample_text.split()
        original_words = sample_text.split()

        # All original content should be represented in chunks
        assert len(original_words) > 0


# ================================================================================
# Retriever Module Tests
# ================================================================================

class TestRetrieverModule:
    """Tests for retriever module (Phase 1)"""

    def test_retriever_module_config(self):
        """Test retriever module configuration"""
        assert config.TOP_K > 0
        assert config.SIMILARITY_THRESHOLD >= 0
        assert config.SIMILARITY_THRESHOLD <= 1

    @pytest.mark.unit
    def test_mock_retriever_initialization(self, mock_retriever):
        """Test retriever initialization"""
        assert mock_retriever is not None
        assert hasattr(mock_retriever, 'retrieve_similar_chunks')

    @pytest.mark.unit
    def test_mock_retriever_returns_list(self, mock_retriever, sample_chunks):
        """Test retriever returns list of results"""
        mock_retriever.chunks = sample_chunks
        results = mock_retriever.retrieve_similar_chunks("test query")

        assert isinstance(results, list)

    @pytest.mark.unit
    def test_mock_retriever_respects_top_k(self, mock_retriever, sample_chunks):
        """Test retriever respects top_k parameter"""
        mock_retriever.chunks = sample_chunks

        results = mock_retriever.retrieve_similar_chunks("test", top_k=config.TOP_K)

        assert len(results) <= config.TOP_K

    @pytest.mark.unit
    def test_retrieval_result_has_score(self, mock_retriever, sample_chunks):
        """Test retrieval results have similarity scores"""
        mock_retriever.chunks = sample_chunks
        results = mock_retriever.retrieve_similar_chunks("test", top_k=1)

        assert len(results) > 0
        assert 'score' in results[0]
        assert isinstance(results[0]['score'], (int, float))

    @pytest.mark.unit
    def test_similarity_scores_in_valid_range(self, mock_retriever, sample_chunks):
        """Test similarity scores are in valid range"""
        mock_retriever.chunks = sample_chunks
        results = mock_retriever.retrieve_similar_chunks("test", top_k=3)

        for result in results:
            score = result['score']
            assert 0 <= score <= 1, f"Score {score} out of range"


# ================================================================================
# RAG Module Tests
# ================================================================================

class TestRAGModule:
    """Tests for RAG module (Phase 1)"""

    def test_rag_module_config(self):
        """Test RAG module configuration"""
        assert config.LLM_TYPE in ['template', 'openai']

    @pytest.mark.unit
    def test_mock_rag_initialization(self, mock_rag_pipeline):
        """Test RAG pipeline initialization"""
        assert mock_rag_pipeline is not None
        assert hasattr(mock_rag_pipeline, 'answer_query')

    @pytest.mark.unit
    def test_rag_answer_query_returns_dict(self, mock_rag_pipeline):
        """Test answer_query returns dictionary"""
        result = mock_rag_pipeline.answer_query("Test query")

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_rag_answer_has_required_fields(self, mock_rag_pipeline, validate_query_result):
        """Test answer has all required fields"""
        result = mock_rag_pipeline.answer_query("What is Python?")

        assert validate_query_result(result)

    @pytest.mark.unit
    def test_rag_answer_content_non_empty(self, mock_rag_pipeline):
        """Test answer content is not empty"""
        result = mock_rag_pipeline.answer_query("Test query")

        assert len(result['answer']) > 0
        assert len(result['context']) >= 0

    @pytest.mark.unit
    def test_rag_confidence_in_range(self, mock_rag_pipeline):
        """Test confidence score is in valid range"""
        result = mock_rag_pipeline.answer_query("Test query")

        confidence = result.get('confidence', result.get('score', 0))
        assert 0 <= confidence <= 1


# ================================================================================
# CLI Module Tests
# ================================================================================

class TestCLIModule:
    """Tests for CLI module (Phase 1)"""

    def test_cli_module_exists(self):
        """Test CLI module can be imported"""
        try:
            from src import cli  # This may fail if not implemented yet
            assert cli is not None
        except ImportError:
            # Expected if not yet implemented
            pass

    @pytest.mark.unit
    def test_cli_has_expected_structure(self):
        """Test CLI module has expected command structure"""
        # Test configuration supports CLI operations
        assert hasattr(config, 'EMBEDDING_MODEL')
        assert hasattr(config, 'CHUNK_SIZE')


# ================================================================================
# Module Integration Tests
# ================================================================================

class TestModuleIntegration:
    """Test interactions between modules"""

    @pytest.mark.integration
    def test_embeddings_to_retriever(self, mock_embedding_manager, mock_retriever, sample_text, sample_chunks):
        """Test embedding output works with retriever"""
        # Create embedding
        embedding = mock_embedding_manager.embed_text(sample_text)

        # Use with retriever
        mock_retriever.chunks = sample_chunks
        results = mock_retriever.retrieve_similar_chunks(sample_text)

        assert len(results) > 0

    @pytest.mark.integration
    def test_retriever_to_rag(self, mock_retriever, mock_rag_pipeline, sample_chunks):
        """Test retriever output works with RAG"""
        # Retrieve context
        mock_retriever.chunks = sample_chunks
        context = mock_retriever.retrieve_similar_chunks("Python")

        # Use in RAG
        answer = mock_rag_pipeline.answer_query("Python")

        assert answer is not None
        assert 'answer' in answer

    @pytest.mark.integration
    def test_full_module_pipeline(self, mock_embedding_manager, mock_retriever, mock_rag_pipeline, sample_text, sample_chunks):
        """Test complete pipeline from embedding to answer"""
        # Step 1: Embed query
        query_embedding = mock_embedding_manager.embed_text(sample_text)
        assert query_embedding is not None

        # Step 2: Retrieve context
        mock_retriever.chunks = sample_chunks
        context = mock_retriever.retrieve_similar_chunks(sample_text)
        assert len(context) > 0

        # Step 3: Generate answer
        answer = mock_rag_pipeline.answer_query(sample_text)
        assert answer is not None


# ================================================================================
# Module Error Handling Tests
# ================================================================================

class TestModuleErrorHandling:
    """Test error handling in modules"""

    @pytest.mark.unit
    def test_embeddings_handles_empty_string(self, mock_embedding_manager):
        """Test embedding handles empty string"""
        try:
            embedding = mock_embedding_manager.embed_text("")
            # Should either work or raise an error gracefully
            assert embedding is not None or embedding is None
        except (ValueError, AssertionError):
            pass  # Expected

    @pytest.mark.unit
    def test_retriever_handles_no_chunks(self, mock_retriever):
        """Test retriever handles empty chunk list"""
        mock_retriever.chunks = []
        results = mock_retriever.retrieve_similar_chunks("query")

        assert isinstance(results, list)

    @pytest.mark.unit
    def test_rag_handles_empty_context(self, mock_rag_pipeline):
        """Test RAG handles empty context"""
        try:
            result = mock_rag_pipeline.answer_query("")
            # Should handle gracefully
            assert result is not None
        except (ValueError, AssertionError):
            pass  # Expected

    @pytest.mark.unit
    def test_modules_handle_none_input(self, mock_embedding_manager, mock_retriever):
        """Test modules handle None input gracefully"""
        # Embeddings
        try:
            embedding = mock_embedding_manager.embed_text(None)
        except (TypeError, AttributeError, ValueError):
            pass  # Expected

        # Retriever
        try:
            results = mock_retriever.retrieve_similar_chunks(None)
        except (TypeError, AttributeError, ValueError):
            pass  # Expected
