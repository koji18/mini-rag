"""
Shared pytest fixtures and utilities for all test modules
"""
import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_DIMENSION,
    EMBEDDING_MODEL, TOP_K, SIMILARITY_THRESHOLD
)


# ================================================================================
# Session-level Fixtures (shared across all tests)
# ================================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            "path": "doc1.txt",
            "content": """
            Python is a high-level programming language.
            It emphasizes code readability and simplicity.
            Python supports multiple programming paradigms.

            The Python interpreter can be used interactively.
            Many libraries extend Python's functionality.
            """
        },
        {
            "path": "doc2.txt",
            "content": """
            Machine Learning is a subset of Artificial Intelligence.
            Machine Learning algorithms learn from data.
            Supervised learning requires labeled data.

            Unsupervised learning finds patterns in unlabeled data.
            Deep learning uses neural networks.
            """
        },
        {
            "path": "doc3.txt",
            "content": """
            Natural Language Processing handles human language.
            NLP techniques include tokenization and stemming.
            Text classification assigns categories to documents.

            Named Entity Recognition identifies entities in text.
            Sentiment analysis determines emotional tone.
            """
        }
    ]


@pytest.fixture(scope="session")
def sample_queries():
    """Sample queries for testing"""
    return [
        "What is Python?",
        "Explain machine learning",
        "How does NLP work?",
        "What are embeddings?",
        "Describe deep learning",
        "What is text classification?",
    ]


@pytest.fixture(scope="session")
def expected_configs():
    """Expected configuration values"""
    return {
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "EMBEDDING_DIMENSION": EMBEDDING_DIMENSION,
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "TOP_K": TOP_K,
        "SIMILARITY_THRESHOLD": SIMILARITY_THRESHOLD,
    }


# ================================================================================
# Module-level Fixtures (shared within test module)
# ================================================================================

@pytest.fixture(scope="module")
def temp_index_dir(test_data_dir):
    """Temporary directory for index files"""
    index_dir = Path(test_data_dir) / "index"
    index_dir.mkdir(exist_ok=True)
    return index_dir


@pytest.fixture(scope="module")
def temp_cache_dir(test_data_dir):
    """Temporary directory for cache files"""
    cache_dir = Path(test_data_dir) / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


@pytest.fixture(scope="module")
def docs_dir(test_data_dir, sample_documents):
    """Create a directory with sample documents"""
    docs_path = Path(test_data_dir) / "docs"
    docs_path.mkdir(exist_ok=True)

    for doc in sample_documents:
        (docs_path / doc["path"]).write_text(doc["content"])

    return docs_path


# ================================================================================
# Function-level Fixtures (fresh for each test)
# ================================================================================

@pytest.fixture
def sample_text():
    """A sample text for embedding tests"""
    return "Python is a versatile programming language used in data science and web development."


@pytest.fixture
def sample_texts():
    """Multiple sample texts"""
    return [
        "Python is a programming language.",
        "Machine learning models learn from data.",
        "Natural language processing analyzes text.",
        "Deep learning uses neural networks.",
        "Data science combines statistics and programming.",
    ]


@pytest.fixture
def sample_chunks():
    """Sample text chunks"""
    return [
        "Python is a high-level programming language used for various applications.",
        "Machine learning involves training models on data to make predictions.",
        "Natural language processing focuses on understanding and generating human language.",
        "Neural networks are inspired by biological neurons in the human brain.",
        "Data preprocessing is a crucial step in any machine learning pipeline.",
        "Embeddings represent text as numerical vectors in a high-dimensional space.",
        "Clustering groups similar data points together without labeled examples.",
        "Classification assigns data points to predefined categories or classes.",
        "Regression predicts continuous numerical values based on input features.",
        "Cross-validation assesses model performance on unseen data.",
    ]


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings (simulated)"""
    np.random.seed(42)
    # Create embeddings that are normalized
    embeddings = np.random.randn(10, EMBEDDING_DIMENSION)
    # Normalize each embedding to unit length
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


@pytest.fixture
def related_text():
    """Text related to a sample query"""
    return "Python is a powerful programming language that is widely used in machine learning and data science applications."


@pytest.fixture
def unrelated_text():
    """Text unrelated to a sample query"""
    return "The capital of France is Paris. Paris is located in northern France along the Seine River."


@pytest.fixture
def config_dict():
    """Sample configuration dictionary"""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "cache_enabled": True,
        "cache_size": 1000,
    }


# ================================================================================
# Helper Validators
# ================================================================================

@pytest.fixture
def validate_embedding():
    """Function to validate embedding structure and properties"""
    def _validate(embedding):
        # Check type
        assert isinstance(embedding, (np.ndarray, list)), "Embedding should be numpy array or list"

        # Check shape
        if isinstance(embedding, np.ndarray):
            assert len(embedding.shape) == 1, "Embedding should be 1-dimensional"
            assert embedding.shape[0] == EMBEDDING_DIMENSION, f"Embedding should have {EMBEDDING_DIMENSION} dimensions"
        else:
            assert len(embedding) == EMBEDDING_DIMENSION, f"Embedding should have {EMBEDDING_DIMENSION} dimensions"

        # Check values are numeric
        embedding_array = np.array(embedding)
        assert np.all(np.isfinite(embedding_array)), "Embedding should contain finite values"

        return True

    return _validate


@pytest.fixture
def validate_chunks():
    """Function to validate chunk structure"""
    def _validate(chunks):
        assert isinstance(chunks, list), "Chunks should be a list"
        assert len(chunks) > 0, "Chunks list should not be empty"

        for chunk in chunks:
            assert isinstance(chunk, str), "Each chunk should be a string"
            assert len(chunk) > 0, "Each chunk should not be empty"
            assert len(chunk) <= CHUNK_SIZE * 2, f"Chunk should not exceed reasonable size"

        return True

    return _validate


@pytest.fixture
def validate_query_result():
    """Function to validate query result structure"""
    def _validate(result):
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "answer" in result, "Result should have 'answer' field"
        assert "context" in result, "Result should have 'context' field"
        assert "confidence" in result or "score" in result, "Result should have confidence/score"

        assert isinstance(result["answer"], str), "Answer should be a string"
        assert isinstance(result["context"], list), "Context should be a list"

        return True

    return _validate


@pytest.fixture
def validate_retrieval_result():
    """Function to validate retrieval result structure"""
    def _validate(result):
        assert isinstance(result, dict), "Retrieval result should be a dictionary"
        assert "chunk" in result, "Result should have 'chunk' field"
        assert "score" in result, "Result should have 'score' field"

        assert isinstance(result["chunk"], str), "Chunk should be a string"
        assert isinstance(result["score"], (int, float)), "Score should be numeric"
        assert 0 <= result["score"] <= 1, "Score should be between 0 and 1"

        return True

    return _validate


# ================================================================================
# Mock Objects
# ================================================================================

@pytest.fixture
def mock_embedding_manager():
    """Mock EmbeddingManager for testing"""
    class MockEmbeddingManager:
        def __init__(self):
            self.cached = {}

        def embed_text(self, text):
            if text not in self.cached:
                np.random.seed(hash(text) % 2**32)
                embedding = np.random.randn(EMBEDDING_DIMENSION)
                embedding = embedding / np.linalg.norm(embedding)
                self.cached[text] = embedding
            return self.cached[text]

        def embed_batch(self, texts):
            embeddings = [self.embed_text(t) for t in texts]
            return np.array(embeddings)

    return MockEmbeddingManager()


@pytest.fixture
def mock_retriever():
    """Mock Retriever for testing"""
    class MockRetriever:
        def __init__(self, chunks=None):
            self.chunks = chunks or []
            self.index = None

        def retrieve_similar_chunks(self, query, top_k=TOP_K):
            # Return mock results
            results = [
                {"chunk": chunk, "score": 0.8 - (i * 0.1), "index": i}
                for i, chunk in enumerate(self.chunks[:top_k])
            ]
            return results

        def load_index(self, path):
            self.index = {"path": str(path)}
            return True

    return MockRetriever()


@pytest.fixture
def mock_rag_pipeline():
    """Mock RAGPipeline for testing"""
    class MockRAGPipeline:
        def __init__(self):
            self.config = {}

        def answer_query(self, query):
            return {
                "answer": "This is a mock answer based on the provided context.",
                "context": [f"Context chunk related to: {query}"],
                "confidence": 0.85,
                "sources": ["doc1.txt"],
            }

        def initialize(self, config):
            self.config = config
            return True

    return MockRAGPipeline()


# ================================================================================
# Test Data Generators
# ================================================================================

@pytest.fixture
def generate_random_embedding():
    """Function to generate random embeddings"""
    def _generate(seed=None):
        if seed is not None:
            np.random.seed(seed)
        embedding = np.random.randn(EMBEDDING_DIMENSION)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    return _generate


@pytest.fixture
def generate_related_chunks(sample_chunks):
    """Function to generate chunks related to a query"""
    def _generate(query, count=3):
        # Simple mock: return first N chunks as "related"
        return sample_chunks[:count]

    return _generate


# ================================================================================
# Marker Registration
# ================================================================================

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring embedding model"
    )
    config.addinivalue_line(
        "markers", "edge_case: mark test as testing edge cases"
    )
