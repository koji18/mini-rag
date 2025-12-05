"""
Mini-RAG システムのドキュメント取り込みモジュール。

ドキュメントの読み込み、チャンク分割、インデックス作成を処理します。
DocumentLoader、DocumentChunker、DocumentIndexer クラスを提供します。
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, INDEX_DIR, MIN_CHUNK_SIZE

# Setup logging
logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    ドキュメントファイルを読み込むクラス。

    指定されたディレクトリから .txt, .md, .rst ファイルを読み込み、
    ドキュメントリストとして返します。

    属性:
        supported_extensions (tuple): サポートされるファイル拡張子
    """

    def __init__(self):
        """DocumentLoader を初期化します。"""
        self.supported_extensions = (".txt", ".md", ".rst")
        logger.info(f"DocumentLoader initialized (formats: {self.supported_extensions})")

    def load_documents(self, docs_dir: str) -> List[Dict[str, Any]]:
        """
        ディレクトリからドキュメントを読み込みます。

        引数:
            docs_dir (str): ドキュメントディレクトリのパス

        戻り値:
            List[Dict]: ドキュメントのリスト
                       各ドキュメントは {'path': str, 'content': str} の形式

        例外:
            ValueError: docs_dir が存在しないか、ディレクトリでない場合
            OSError: ファイル読み込みエラー
        """
        docs_path = Path(docs_dir)

        # 入力検証
        if not docs_path.exists():
            raise ValueError(f"Directory does not exist: {docs_dir}")

        if not docs_path.is_dir():
            raise ValueError(f"Path is not a directory: {docs_dir}")

        documents = []

        # サポートされているファイルを再帰的に検索
        for ext in self.supported_extensions:
            for file_path in docs_path.rglob(f"*{ext}"):
                try:
                    content = self._load_file(file_path)
                    if content.strip():
                        documents.append(
                            {"path": str(file_path.relative_to(docs_path)), "content": content}
                        )
                        logger.debug(f"Loaded document: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue

        logger.info(f"Loaded {len(documents)} documents from {docs_dir}")
        return documents

    def _load_file(self, file_path: Path) -> str:
        """
        ファイルを読み込んでテキストを返します。

        引数:
            file_path (Path): ファイルパス

        戻り値:
            str: ファイルの内容

        例外:
            OSError: ファイル読み込みエラー
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decode failed for {file_path}, trying other encodings")
            for encoding in ["latin-1", "cp1252", "shift-jis"]:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        return f.read()
                except (UnicodeDecodeError, LookupError):
                    continue
            raise OSError(f"Could not decode file: {file_path}")


class DocumentChunker:
    """
    ドキュメントをチャンクに分割するクラス。

    hierarchical 戦略を使用して、段落と文を考慮しながら
    テキストを適切なサイズのチャンクに分割します。

    属性:
        chunk_size (int): チャンクの最大サイズ
        overlap (int): チャンク間のオーバーラップサイズ
        min_chunk_size (int): チャンクの最小サイズ
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE,
    ):
        """
        DocumentChunker を初期化します。

        引数:
            chunk_size (int): チャンクの最大サイズ。デフォルト: 512
            overlap (int): オーバーラップサイズ。デフォルト: 50
            min_chunk_size (int): 最小チャンクサイズ。デフォルト: 50
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

        logger.info(
            f"DocumentChunker initialized (size: {chunk_size}, "
            f"overlap: {overlap}, min: {min_chunk_size})"
        )

    def chunk_documents(self, texts: List[str]) -> List[str]:
        """
        複数のテキストをチャンクに分割します。

        引数:
            texts (List[str]): 分割するテキストのリスト

        戻り値:
            List[str]: チャンクのリスト

        例外:
            TypeError: texts がリストでない場合
            ValueError: texts が空の場合
        """
        if not isinstance(texts, list):
            raise TypeError(f"Expected list, got {type(texts).__name__}")

        if not texts:
            raise ValueError("Input list cannot be empty")

        all_chunks = []

        for text in texts:
            if not isinstance(text, str):
                logger.warning(f"Skipping non-string item: {type(text)}")
                continue

            chunks = self.chunk_text(text)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(texts)} documents")
        return all_chunks

    def chunk_text(self, text: str) -> List[str]:
        """
        単一のテキストをチャンクに分割します。

        hierarchical 戦略を使用:
        1. まず段落で分割
        2. 段落が大きすぎる場合は文で分割
        3. 文が大きすぎる場合は文字数で分割

        引数:
            text (str): 分割するテキスト

        戻り値:
            List[str]: チャンクのリスト
        """
        if not text or not text.strip():
            return []

        paragraphs = self._split_into_paragraphs(text)
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                sentences = self._split_into_sentences(paragraph)
                for sentence in sentences:
                    if len(sentence) > self.chunk_size:
                        forced_chunks = self._force_split(sentence)
                        chunks.extend(forced_chunks)
                    elif len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = self._get_overlap(current_chunk) + sentence + " "
            else:
                if len(current_chunk) + len(paragraph) <= self.chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = self._get_overlap(current_chunk) + paragraph + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        chunks = [c for c in chunks if len(c) >= self.min_chunk_size]

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        テキストを段落に分割します。

        引数:
            text (str): 分割するテキスト

        戻り値:
            List[str]: 段落のリスト
        """
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        テキストを文に分割します。

        引数:
            text (str): 分割するテキスト

        戻り値:
            List[str]: 文のリスト
        """
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _force_split(self, text: str) -> List[str]:
        """
        大きすぎるテキストを強制的に分割します。

        引数:
            text (str): 分割するテキスト

        戻り値:
            List[str]: 分割されたチャンクのリスト
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - self.overlap

        return chunks

    def _get_overlap(self, text: str) -> str:
        """
        テキストの末尾からオーバーラップ部分を取得します。

        引数:
            text (str): 元のテキスト

        戻り値:
            str: オーバーラップ部分
        """
        if len(text) <= self.overlap:
            return text

        overlap_text = text[-self.overlap :]

        words = overlap_text.split()
        if len(words) > 1:
            return " ".join(words[1:]) + " "

        return overlap_text


class DocumentIndexer:
    """
    チャンクと埋め込みからインデックスを作成するクラス。

    RAGIndex を作成し、永続化を管理します。

    属性:
        index_dir (Path): インデックスを保存するディレクトリ
    """

    def __init__(self, index_dir: str = INDEX_DIR):
        """
        DocumentIndexer を初期化します。

        引数:
            index_dir (str): インデックスディレクトリのパス。デフォルト: config.INDEX_DIR
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DocumentIndexer initialized (index_dir: {index_dir})")

    def create_index(self, chunks: List[str], embeddings: Any) -> Dict[str, Any]:
        """
        チャンクと埋め込みからインデックスを作成します。

        引数:
            chunks (List[str]): テキストチャンクのリスト
            embeddings (Any): チャンクの埋め込み (numpy array または list)

        戻り値:
            Dict[str, Any]: インデックスオブジェクト
                           {'chunks': List[str], 'embeddings': Any}

        例外:
            ValueError: chunks が空、または chunks と embeddings の数が一致しない場合
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")

        if isinstance(embeddings, np.ndarray):
            if len(chunks) != embeddings.shape[0]:
                raise ValueError(
                    f"Number of chunks ({len(chunks)}) must match "
                    f"number of embeddings ({embeddings.shape[0]})"
                )
        elif isinstance(embeddings, list):
            if len(chunks) != len(embeddings):
                raise ValueError(
                    f"Number of chunks ({len(chunks)}) must match "
                    f"number of embeddings ({len(embeddings)})"
                )

        index = {
            "chunks": chunks,
            "embeddings": embeddings,
            "metadata": {
                "num_chunks": len(chunks),
                "chunk_size": CHUNK_SIZE,
                "overlap": CHUNK_OVERLAP,
            },
        }

        logger.info(f"Created index with {len(chunks)} chunks")
        return index

    def save_index(self, index: Dict[str, Any], filename: str = "rag_index.pkl") -> Path:
        """
        インデックスをファイルに保存します。

        引数:
            index (Dict): インデックスオブジェクト
            filename (str): 保存するファイル名。デフォルト: "rag_index.pkl"

        戻り値:
            Path: 保存されたファイルのパス

        例外:
            OSError: ファイル保存エラー
        """
        filepath = self.index_dir / filename

        try:
            with open(filepath, "wb") as f:
                pickle.dump(index, f)
            logger.info(f"Index saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def load_index(self, filename: str = "rag_index.pkl") -> Dict[str, Any]:
        """
        ファイルからインデックスを読み込みます。

        引数:
            filename (str): 読み込むファイル名。デフォルト: "rag_index.pkl"

        戻り値:
            Dict[str, Any]: インデックスオブジェクト

        例外:
            FileNotFoundError: ファイルが存在しない場合
            OSError: ファイル読み込みエラー
        """
        filepath = self.index_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")

        try:
            with open(filepath, "rb") as f:
                index = pickle.load(f)
            logger.info(f"Index loaded from {filepath}")
            return index
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
