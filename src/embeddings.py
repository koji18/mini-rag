"""
Mini-RAG システムの埋め込みモジュール。

Sentence Transformers を使用したテキスト-ベクトル変換を処理します。
単一/バッチ埋め込み用の EmbeddingManager と キャッシング用の EmbeddingCache を提供します。
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
    Sentence Transformers を使用したテキスト埋め込みの管理。

    テキスト文字列を all-MiniLM-L6-v2 モデルを使用して
    384 次元のベクトルに変換します。単一テキスト、バッチ処理、
    コサイン類似度計算をサポートしています。

    属性:
        model (SentenceTransformer): 読み込まれた埋め込みモデル
        cache (EmbeddingCache): 埋め込みキャッシュインスタンス
    """

    def __init__(self, use_cache: bool = True):
        """
        EmbeddingManager をモデルとキャッシュで初期化します。

        引数:
            use_cache (bool): キャッシュを有効にするかどうか。デフォルト: True
        """
        self.model: Optional[SentenceTransformer] = None
        self.use_cache = use_cache
        self.cache = EmbeddingCache() if use_cache else None

        logger.info(f"EmbeddingManager initialized (model: {EMBEDDING_MODEL})")

    def _load_model(self) -> None:
        """初回使用時に埋め込みモデルを遅延ロードします。"""
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
        単一のテキスト文字列を埋め込みます。

        テキスト文字列を 384 次元のベクトルに変換します。
        キャッシュが有効な場合は使用されます。

        引数:
            text (str): 埋め込むテキスト

        戻り値:
            np.ndarray: 形状 (384,)、データ型 float32 の埋め込みベクトル

        例外:
            TypeError: text が文字列でない場合
            ValueError: text が空の場合
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
        複数のテキストをバッチで埋め込みます。

        複数のテキストを効率的に埋め込みます。各テキストについて
        まずキャッシュを確認し、その後残りのテキストをバッチで埋め込みます。

        引数:
            texts (List[str]): 埋め込むテキスト文字列のリスト

        戻り値:
            np.ndarray: 形状 (n, 384)、データ型 float32 の埋め込み配列
                       (n は入力テキストの数)

        例外:
            TypeError: texts が文字列のリストでない場合
            ValueError: texts が空の場合
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
        2 つのベクトル間のコサイン類似度を計算します。

        正規化されたベクトルの場合、内積と同等です。

        引数:
            vec1 (np.ndarray): 最初のベクトル (形状: (d,))
            vec2 (np.ndarray): 2 番目のベクトル (形状: (d,))

        戻り値:
            float: -1.0 ～ 1.0 の範囲のコサイン類似度
                  (正規化されたベクトルの場合は通常 0.0 ～ 1.0)

        例外:
            TypeError: 入力が numpy 配列でない場合
            ValueError: ベクトルの形状が互換性がない場合
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
        埋め込みベクトルのプロパティを検証します。

        引数:
            embedding (np.ndarray): 検証する埋め込みベクトル

        例外:
            ValueError: 埋め込みが無効な場合
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
    LRU削除と TTL サポート付きの埋め込みキャッシュ。

    埋め込みをメモリに保存し、最大サイズ制限があります。
    キャッシュが満杯の場合、最も最近使用されていない
    (Least Recently Used) 削除ポリシーを使用します。
    自動削除のための Time-To-Live (TTL) をサポートしています。

    属性:
        cache (Dict): メモリ内キャッシュ {text: embedding}
        timestamps (Dict): TTL 用の作成タイムスタンプ {text: timestamp}
        max_size (int): 最大キャッシュサイズ
        ttl (int): Time-to-live (秒単位)
    """

    def __init__(self,
                 max_size: int = EMBEDDING_CACHE_SIZE,
                 ttl: int = EMBEDDING_CACHE_TTL):
        """
        埋め込みキャッシュを初期化します。

        引数:
            max_size (int): キャッシュ可能な埋め込みの最大数。デフォルト: 10,000
            ttl (int): TTL (秒単位)。デフォルト: 86,400 (24時間)
        """
        self.cache: Dict[str, np.ndarray] = {}
        self.timestamps: Dict[str, float] = {}
        self.access_times: Dict[str, float] = {}

        self.max_size = max_size
        self.ttl = ttl

        logger.info(f"EmbeddingCache initialized (size: {max_size}, TTL: {ttl}s)")

    def get(self, key: str) -> Optional[np.ndarray]:
        """
        キャッシュから埋め込みを取得します。

        引数:
            key (str): キャッシュキー (通常は入力テキスト)

        戻り値:
            np.ndarray: 有効な場合はキャッシュされた埋め込み、そうでない場合は None
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
        埋め込みをキャッシュに保存します。

        キャッシュが満杯の場合、最も最近使用されていないアイテムを削除します。

        引数:
            key (str): キャッシュキー (通常は入力テキスト)
            value (np.ndarray): キャッシュする埋め込みベクトル
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
        TTL に基づいてすべての期限切れエントリを削除します。

        戻り値:
            int: 削除されたエントリの数
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
        """すべてのキャッシュエントリをクリアします。"""
        self.cache.clear()
        self.timestamps.clear()
        self.access_times.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, int]:
        """
        キャッシュ統計を取得します。

        戻り値:
            Dict: サイズなどのキャッシュ統計情報
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
        }
