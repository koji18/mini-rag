"""
Mini-RAG システムのメインロジックモジュール。

RAG (Retrieval-Augmented Generation) パイプラインを実装します。
クエリから回答生成までの全フローを管理します。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from src.config import INDEX_DIR, LLM_TYPE, TOP_K
from src.embeddings import EmbeddingManager
from src.retriever import RAGIndex, Retriever

# Setup logging
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG (Retrieval-Augmented Generation) パイプラインクラス。

    クエリから回答生成までの全フローを管理します：
    1. クエリをembeddingに変換
    2. 類似チャンクを検索
    3. テンプレートベースまたはLLMで回答生成

    属性:
        embedding_manager (EmbeddingManager): 埋め込み管理
        retriever (Retriever): チャンク検索
        llm_type (str): 回答生成方式 ('template' or 'openai')
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager = None,
        retriever: Retriever = None,
        llm_type: str = LLM_TYPE,
    ):
        """
        RAGPipeline を初期化します。

        引数:
            embedding_manager (EmbeddingManager): 埋め込み管理インスタンス
            retriever (Retriever): 検索インスタンス
            llm_type (str): 回答生成方式。デフォルト: 'template'
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.retriever = retriever or Retriever(embedding_manager=self.embedding_manager)
        self.llm_type = llm_type

        logger.info(f"RAGPipeline initialized (llm_type: {llm_type})")

    def initialize(self, index_path: str = None) -> None:
        """
        インデックスを読み込んでパイプラインを初期化します。

        引数:
            index_path (str): インデックスファイルのパス。
                            None の場合はデフォルトパスを使用

        例外:
            FileNotFoundError: インデックスファイルが存在しない場合
        """
        if index_path is None:
            index_path = str(Path(INDEX_DIR) / "rag_index.pkl")

        # インデックスを読み込み
        self.retriever.index.load(index_path)
        logger.info(f"Pipeline initialized with index: {index_path}")

    def answer_query(self, query: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """
        クエリに対する回答を生成します。

        引数:
            query (str): ユーザーのクエリ
            top_k (int): 検索するチャンクの最大数。デフォルト: 3

        戻り値:
            Dict[str, Any]: 回答辞書
                - answer (str): 生成された回答
                - context (List[str]): 使用されたコンテキストチャンク
                - confidence (float): 信頼度スコア

        例外:
            ValueError: query が空の場合
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Processing query: {query[:50]}...")

        # 1. 類似チャンクを検索
        results = self.retriever.retrieve_similar_chunks(query, top_k=top_k)

        # 2. コンテキストチャンクを抽出
        context_chunks = [r["chunk"] for r in results]

        # 3. 信頼度スコアを計算（平均類似度）
        if results:
            confidence = sum(r["score"] for r in results) / len(results)
        else:
            confidence = 0.0

        # 4. 回答を生成
        if self.llm_type == "template":
            answer = self._generate_template_answer(query, context_chunks, confidence)
        elif self.llm_type == "openai":
            answer = self._generate_openai_answer(query, context_chunks)
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")

        logger.info(f"Answer generated (confidence: {confidence:.3f})")

        return {"answer": answer, "context": context_chunks, "confidence": confidence}

    def _generate_template_answer(
        self, query: str, context_chunks: List[str], confidence: float
    ) -> str:
        """
        テンプレートベースで回答を生成します。

        引数:
            query (str): ユーザーのクエリ
            context_chunks (List[str]): コンテキストチャンク
            confidence (float): 信頼度スコア

        戻り値:
            str: 生成された回答
        """
        if not context_chunks:
            return "申し訳ございません。関連する情報が見つかりませんでした。"

        # シンプルなテンプレートベースの回答生成
        if confidence > 0.7:
            intro = "以下の情報が見つかりました："
        elif confidence > 0.4:
            intro = "関連する情報として以下が見つかりました："
        else:
            intro = "参考情報として以下が見つかりましたが、関連性は低い可能性があります："

        # コンテキストチャンクを結合
        context_text = "\n\n".join(
            [f"- {chunk[:200]}..." if len(chunk) > 200 else f"- {chunk}" for chunk in context_chunks]
        )

        answer = f"{intro}\n\n{context_text}"

        return answer

    def _generate_openai_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        OpenAI API を使用して回答を生成します。

        引数:
            query (str): ユーザーのクエリ
            context_chunks (List[str]): コンテキストチャンク

        戻り値:
            str: 生成された回答

        注意:
            Phase 1 ではプレースホルダーのみ。Phase 3 で実装予定
        """
        logger.warning("OpenAI integration not implemented yet (Phase 3)")
        return self._generate_template_answer(query, context_chunks, 0.5)
