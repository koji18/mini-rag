"""
Embedding module for Mini-RAG system.

Handles text-to-vector conversion using Sentence Transformers.
Provides EmbeddingManager for single/batch embeddings and EmbeddingCache for caching.
"""

import logging
import time
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_NORMALIZE,
    EMBEDDING_CACHE_SIZE,
    EMBEDDING_CACHE_TTL,
    CACHE_DIR,
    VALIDATE_EMBEDDINGS,
)

# Setup logging
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages text embedding using Sentence Transformers.

    Converts text strings into 384-dimensional vectors using the
    all-MiniLM-L6-v2 model. Supports single text, batch processing,
    and cosine similarity calculations.

    Attributes:
        model (SentenceTransformer): Loaded embedding model
        cache (EmbeddingCache): Embedding cache instance
    """

    def __init__(self, use_cache: bool = True):
        """
        Initialize EmbeddingManager with model and cache.

        Args:
            use_cache (bool): Whether to enable caching. Default: True
        """
        self.model: Optional[SentenceTransformer] = None
        self.use_cache = use_cache
        self.cache = EmbeddingCache() if use_cache else None

        logger.info(f"EmbeddingManager initialized (model: {EMBEDDING_MODEL})")

    def _load_model(self) -> None:
        """Lazy load the embedding model on first use."""
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
                self.model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    device=EMBEDDING_DEVICE
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Converts a text string into a 384-dimensional vector.
        Uses cache if enabled.

        Args:
            text (str): Input text to embed

        Returns:
            np.ndarray: Embedding vector of shape (384,) with dtype float32

        Raises:
            TypeError: If text is not a string
            ValueError: If text is empty
        """
        # Input validation
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        if not text.strip():
            raise ValueError("Input text cannot be empty")

        # Check cache
        if self.use_cache and self.cache is not None:
            cached_embedding = self.cache.get(text)
            if cached_embedding is not None:
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return cached_embedding

        # Load model if not already loaded
        if self.model is None:
            self._load_model()

        # Embed text
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            # Validate embedding
            if VALIDATE_EMBEDDINGS:
                self._validate_embedding(embedding)

            # Ensure correct dtype
            embedding = embedding.astype(np.float32)

            # Cache the embedding
            if self.use_cache and self.cache is not None:
                self.cache.set(text, embedding)

            logger.debug(f"Embedded text: {text[:50]}... (shape: {embedding.shape})")
            return embedding

        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts as a batch.

        Efficiently embeds multiple texts. Checks cache for each text first,
        then batches the remaining texts for embedding.

        Args:
            texts (List[str]): List of text strings to embed

        Returns:
            np.ndarray: Embeddings of shape (n, 384) with dtype float32
                       where n is the number of input texts

        Raises:
            TypeError: If texts is not a list of strings
            ValueError: If texts is empty
        """
        # Input validation
        if not isinstance(texts, list):
            raise TypeError(f"Expected list, got {type(texts).__name__}")

        if not texts:
            raise ValueError("Input list cannot be empty")

        if not all(isinstance(t, str) for t in texts):
            raise TypeError("All items in texts must be strings")

        logger.info(f"Embedding batch of {len(texts)} texts")

        # Load model if not already loaded
        if self.model is None:
            self._load_model()

        # Separate cached and uncached texts
        embeddings_list = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"Text at index {i} is empty")

            # Check cache
            if self.use_cache and self.cache is not None:
                cached_embedding = self.cache.get(text)
                if cached_embedding is not None:
                    embeddings_list.append((i, cached_embedding))
                    continue

            uncached_texts.append(text)
            uncached_indices.append(i)

        # Embed uncached texts in batches
        if uncached_texts:
            logger.debug(f"Embedding {len(uncached_texts)} uncached texts")

            for batch_start in range(0, len(uncached_texts), EMBEDDING_BATCH_SIZE):
                batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, len(uncached_texts))
                batch = uncached_texts[batch_start:batch_end]

                try:
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )

                    # Process each embedding
                    for j, embedding in enumerate(batch_embeddings):
                        # Validate
                        if VALIDATE_EMBEDDINGS:
                            self._validate_embedding(embedding)

                        # Convert dtype
                        embedding = embedding.astype(np.float32)

                        # Cache
                        if self.use_cache and self.cache is not None:
                            self.cache.set(batch[j], embedding)

                        # Store with original index
                        idx = uncached_indices[batch_start + j]
                        embeddings_list.append((idx, embedding))

                except Exception as e:
                    logger.error(f"Failed to embed batch: {e}")
                    raise

        # Sort by original index and create output array
        embeddings_list.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings_list], dtype=np.float32)

        logger.info(f"Batch embedding complete. Shape: {result.shape}")
        return result

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Computes: (A·B) / (||A|| * ||B||)

        For normalized vectors, this is equivalent to the dot product.

        Args:
            vec1 (np.ndarray): First vector (shape: (d,))
            vec2 (np.ndarray): Second vector (shape: (d,))

        Returns:
            float: Cosine similarity in range [-1.0, 1.0]
                  (typically [0.0, 1.0] for normalized vectors)

        Raises:
            TypeError: If inputs are not numpy arrays
            ValueError: If vectors have incompatible shapes
        """
        # Input validation
        if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
            raise TypeError("Both inputs must be numpy arrays")

        if vec1.ndim != 1 or vec2.ndim != 1:
            raise ValueError("Vectors must be 1-dimensional")

        if vec1.shape[0] != vec2.shape[0]:
            raise ValueError(
                f"Vector dimensions must match: {vec1.shape[0]} vs {vec2.shape[0]}"
            )

        # Calculate cosine similarity
        # For normalized vectors: cos_sim = dot product
        dot_product = np.dot(vec1, vec2)

        # Clamp to [-1, 1] to avoid numerical errors
        similarity = np.clip(dot_product, -1.0, 1.0)

        return float(similarity)

    @staticmethod
    def _validate_embedding(embedding: np.ndarray) -> None:
        """
        Validate embedding vector properties.

        Args:
            embedding (np.ndarray): Embedding vector to validate

        Raises:
            ValueError: If embedding is invalid
        """
        if embedding.shape[0] != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, "
                f"got {embedding.shape[0]}"
            )

        if not np.all(np.isfinite(embedding)):
            raise ValueError("Embedding contains non-finite values (NaN or Inf)")


class EmbeddingCache:
    """
    Cache for embeddings with LRU eviction and TTL support.

    Stores embeddings in memory with a maximum size limit.
    Uses Least Recently Used (LRU) eviction policy when cache is full.
    Supports Time-To-Live (TTL) for automatic expiration.

    Attributes:
        cache (Dict): In-memory cache {text: embedding}
        timestamps (Dict): Creation timestamps for TTL {text: timestamp}
        max_size (int): Maximum cache size
        ttl (int): Time-to-live in seconds
    """

    def __init__(self,
                 max_size: int = EMBEDDING_CACHE_SIZE,
                 ttl: int = EMBEDDING_CACHE_TTL):
        """
        Initialize embedding cache.

        Args:
            max_size (int): Maximum number of cached embeddings. Default: 10,000
            ttl (int): Time-to-live in seconds. Default: 86,400 (24 hours)
        """
        self.cache: Dict[str, np.ndarray] = {}
        self.timestamps: Dict[str, float] = {}
        self.access_times: Dict[str, float] = {}

        self.max_size = max_size
        self.ttl = ttl

        logger.info(f"EmbeddingCache initialized (size: {max_size}, TTL: {ttl}s)")

    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache.

        Args:
            key (str): Cache key (usually the input text)

        Returns:
            np.ndarray: Cached embedding if valid, None otherwise
        """
        if key not in self.cache:
            return None

        # Check TTL
        creation_time = self.timestamps.get(key, 0)
        if time.time() - creation_time > self.ttl:
            # Expired, remove and return None
            del self.cache[key]
            del self.timestamps[key]
            if key in self.access_times:
                del self.access_times[key]
            logger.debug(f"Cache expired for key: {key[:50]}...")
            return None

        # Update access time for LRU
        self.access_times[key] = time.time()

        return self.cache[key].copy()

    def set(self, key: str, value: np.ndarray) -> None:
        """
        Store embedding in cache.

        If cache is full, removes the least recently used item.

        Args:
            key (str): Cache key (usually the input text)
            value (np.ndarray): Embedding vector to cache
        """
        # Evict LRU if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Find least recently used
            lru_key = min(
                self.access_times.keys(),
                key=lambda k: self.access_times.get(k, 0)
            )

            del self.cache[lru_key]
            del self.timestamps[lru_key]
            del self.access_times[lru_key]

            logger.debug(f"Evicted LRU cache entry: {lru_key[:50]}...")

        # Store embedding
        self.cache[key] = value.copy()
        self.timestamps[key] = time.time()
        self.access_times[key] = time.time()

    def clear_expired(self) -> int:
        """
        Remove all expired entries based on TTL.

        Returns:
            int: Number of entries removed
        """
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]

        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]
            if key in self.access_times:
                del self.access_times[key]

        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.timestamps.clear()
        self.access_times.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict: Cache stats including size and hit info
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
        }
