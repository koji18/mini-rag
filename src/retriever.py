"""
Mini-RAG システムの検索モジュール。

埋め込みベクトルを使用した類似度検索機能を提供します。
RAGIndex と Retriever クラスを提供します。
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.config import INDEX_DIR, SIMILARITY_THRESHOLD, TOP_K
from src.embeddings import EmbeddingManager

# Setup logging
logger = logging.getLogger(__name__)


class RAGIndex:
    """
    チャンクと埋め込みを保持し、類似度検索を行うインデックスクラス。

    ファイルベースのインデックスで、チャンクと対応する埋め込みベクトルを保存し、
    クエリ埋め込みに基づいて類似チャンクを検索します。

    属性:
        chunks (List[str]): テキストチャンクのリスト
        embeddings (np.ndarray): 埋め込みベクトルの配列 (形状: (n, dim))
    """

    def __init__(self):
        """RAGIndex を初期化します。"""
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        logger.info("RAGIndex initialized")

    def add_chunks(self, chunks: List[str], embeddings: np.ndarray) -> None:
        """
        チャンクと埋め込みをインデックスに追加します。

        引数:
            chunks (List[str]): テキストチャンクのリスト
            embeddings (np.ndarray): 埋め込みベクトル (形状: (n, dim))

        例外:
            ValueError: chunks が空、または chunks と embeddings の数が一致しない場合
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({embeddings.shape[0]})"
            )

        self.chunks = chunks
        self.embeddings = embeddings
        logger.info(f"Added {len(chunks)} chunks to index")

    def retrieve(
        self, query_embedding: np.ndarray, top_k: int = TOP_K
    ) -> List[Dict[str, Any]]:
        """
        クエリ埋め込みに類似したチャンクを検索します。

        引数:
            query_embedding (np.ndarray): クエリの埋め込みベクトル (形状: (dim,))
            top_k (int): 返す結果の最大数。デフォルト: 3

        戻り値:
            List[Dict]: 検索結果のリスト
                       各結果は {'chunk': str, 'score': float, 'index': int} の形式

        例外:
            ValueError: インデックスが空の場合
        """
        if len(self.chunks) == 0:
            raise ValueError("Index is empty. Add chunks first.")

        # すべてのチャンクとの類似度を計算
        similarities = self._calculate_similarities(query_embedding)

        # 類似度でソート（降順）
        sorted_indices = np.argsort(similarities)[::-1]

        # top_k 個の結果を取得
        results = []
        for idx in sorted_indices[:top_k]:
            results.append(
                {"chunk": self.chunks[idx], "score": float(similarities[idx]), "index": int(idx)}
            )

        logger.debug(f"Retrieved {len(results)} results")
        return results

    def _calculate_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        クエリ埋め込みとすべてのチャンク埋め込み間のコサイン類似度を計算します。

        引数:
            query_embedding (np.ndarray): クエリの埋め込みベクトル

        戻り値:
            np.ndarray: 類似度の配列
        """
        # コサイン類似度 = 正規化されたベクトルの内積
        # query_embedding と embeddings はすでに正規化されていると仮定
        similarities = np.dot(self.embeddings, query_embedding)
        return similarities

    def save(self, filepath: str) -> None:
        """
        インデックスをファイルに保存します。

        引数:
            filepath (str): 保存先ファイルパス

        例外:
            OSError: ファイル保存エラー
        """
        data = {"chunks": self.chunks, "embeddings": self.embeddings}

        try:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Index saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def load(self, filepath: str) -> None:
        """
        ファイルからインデックスを読み込みます。

        引数:
            filepath (str): 読み込むファイルパス

        例外:
            FileNotFoundError: ファイルが存在しない場合
            OSError: ファイル読み込みエラー
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            self.chunks = data["chunks"]
            self.embeddings = data["embeddings"]
            logger.info(f"Index loaded from {filepath} ({len(self.chunks)} chunks)")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise


class Retriever:
    """
    テキストクエリから類似チャンクを検索するクラス。

    EmbeddingManager と RAGIndex を組み合わせて、
    テキストクエリを埋め込みベクトルに変換し、類似チャンクを検索します。

    属性:
        embedding_manager (EmbeddingManager): 埋め込み管理
        index (RAGIndex): 検索インデックス
        similarity_threshold (float): 類似度の閾値
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager = None,
        index: RAGIndex = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        """
        Retriever を初期化します。

        引数:
            embedding_manager (EmbeddingManager): 埋め込み管理インスタンス
            index (RAGIndex): インデックスインスタンス
            similarity_threshold (float): 類似度閾値。デフォルト: 0.3
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.index = index or RAGIndex()
        self.similarity_threshold = similarity_threshold

        logger.info(
            f"Retriever initialized (threshold: {similarity_threshold})"
        )

    def retrieve_similar_chunks(
        self, query: str, top_k: int = TOP_K
    ) -> List[Dict[str, Any]]:
        """
        クエリテキストに類似したチャンクを検索します。

        引数:
            query (str): 検索クエリ
            top_k (int): 返す結果の最大数。デフォルト: 3

        戻り値:
            List[Dict]: 検索結果のリスト
                       各結果は {'chunk': str, 'score': float} の形式

        例外:
            ValueError: query が空の場合
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # クエリを埋め込みに変換
        query_embedding = self.embedding_manager.embed_text(query)

        # 類似チャンクを検索
        results = self.index.retrieve(query_embedding, top_k=top_k)

        # 閾値でフィルタリング
        filtered_results = [
            {"chunk": r["chunk"], "score": r["score"]}
            for r in results
            if r["score"] >= self.similarity_threshold
        ]

        logger.info(
            f"Retrieved {len(filtered_results)}/{len(results)} chunks "
            f"above threshold {self.similarity_threshold}"
        )

        return filtered_results
