"""
Mini-RAG システムのコマンドラインインターフェース。

RAGシステムの操作を提供するCLIツールです。
ドキュメントのインジェスト、クエリ実行、設定表示などが可能です。
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    INDEX_DIR,
    LLM_TYPE,
    SIMILARITY_THRESHOLD,
    TOP_K,
    get_config_summary,
)
from src.embeddings import EmbeddingManager
from src.ingest import DocumentChunker, DocumentIndexer, DocumentLoader
from src.rag import RAGPipeline
from src.retriever import RAGIndex, Retriever

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CLI:
    """
    Mini-RAG システムの CLI クラス。

    コマンドラインからRAGシステムを操作するためのインターフェースを提供します。

    サブコマンド:
        ingest: ドキュメントをインジェスト
        query: クエリを実行して回答を取得
        search: 類似チャンクを検索
        index: インデックス操作
        config: 設定表示
    """

    def __init__(self):
        """CLI を初期化します。"""
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """
        argparse パーサーを作成します。

        戻り値:
            argparse.ArgumentParser: コマンドラインパーサー
        """
        parser = argparse.ArgumentParser(
            prog="rag",
            description="Mini-RAG: Retrieval-Augmented Generation システム",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        subparsers = parser.add_subparsers(
            dest="command", help="実行するコマンド", required=True
        )

        # ingest コマンド
        ingest_parser = subparsers.add_parser(
            "ingest",
            help="ドキュメントをインジェストしてインデックスを作成",
        )
        ingest_parser.add_argument(
            "docs_dir",
            type=str,
            help="ドキュメントディレクトリのパス",
        )
        ingest_parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="インデックスの出力先（デフォルト: data/index/rag_index.pkl）",
        )

        # query コマンド
        query_parser = subparsers.add_parser(
            "query",
            help="クエリを実行して回答を取得",
        )
        query_parser.add_argument(
            "question",
            type=str,
            help="質問文",
        )
        query_parser.add_argument(
            "--index",
            type=str,
            default=None,
            help="使用するインデックスファイルのパス",
        )
        query_parser.add_argument(
            "--top-k",
            type=int,
            default=TOP_K,
            help=f"検索するチャンクの最大数（デフォルト: {TOP_K}）",
        )

        # search コマンド
        search_parser = subparsers.add_parser(
            "search",
            help="類似チャンクを検索（回答生成なし）",
        )
        search_parser.add_argument(
            "query",
            type=str,
            help="検索クエリ",
        )
        search_parser.add_argument(
            "--index",
            type=str,
            default=None,
            help="使用するインデックスファイルのパス",
        )
        search_parser.add_argument(
            "--top-k",
            type=int,
            default=TOP_K,
            help=f"検索するチャンクの最大数（デフォルト: {TOP_K}）",
        )

        # index コマンド
        index_parser = subparsers.add_parser(
            "index",
            help="インデックス操作",
        )
        index_subparsers = index_parser.add_subparsers(
            dest="index_command",
            help="インデックス操作",
            required=True,
        )

        # index rebuild サブコマンド
        rebuild_parser = index_subparsers.add_parser(
            "rebuild",
            help="インデックスを再構築",
        )
        rebuild_parser.add_argument(
            "docs_dir",
            type=str,
            help="ドキュメントディレクトリのパス",
        )
        rebuild_parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="インデックスの出力先",
        )

        # config コマンド
        config_parser = subparsers.add_parser(
            "config",
            help="設定操作",
        )
        config_subparsers = config_parser.add_subparsers(
            dest="config_command",
            help="設定操作",
            required=True,
        )

        # config show サブコマンド
        config_subparsers.add_parser(
            "show",
            help="現在の設定を表示",
        )

        return parser

    def run(self, args: Optional[list] = None) -> int:
        """
        CLI を実行します。

        引数:
            args (Optional[list]): コマンドライン引数（テスト用）

        戻り値:
            int: 終了コード（0: 成功、1: エラー）
        """
        parsed_args = self.parser.parse_args(args)

        try:
            if parsed_args.command == "ingest":
                return self._handle_ingest(parsed_args)
            elif parsed_args.command == "query":
                return self._handle_query(parsed_args)
            elif parsed_args.command == "search":
                return self._handle_search(parsed_args)
            elif parsed_args.command == "index":
                return self._handle_index(parsed_args)
            elif parsed_args.command == "config":
                return self._handle_config(parsed_args)
            else:
                logger.error(f"Unknown command: {parsed_args.command}")
                return 1

        except KeyboardInterrupt:
            logger.info("\n処理を中断しました")
            return 1
        except Exception as e:
            logger.error(f"エラーが発生しました: {e}", exc_info=True)
            return 1

    def _handle_ingest(self, args) -> int:
        """
        ingest コマンドを処理します。

        引数:
            args: パース済み引数

        戻り値:
            int: 終了コード
        """
        logger.info("=" * 60)
        logger.info("ドキュメントインジェスト開始")
        logger.info("=" * 60)

        # 1. ドキュメント読み込み
        logger.info(f"ドキュメントディレクトリ: {args.docs_dir}")
        loader = DocumentLoader()

        try:
            documents = loader.load_documents(args.docs_dir)
            logger.info(f"✅ {len(documents)} 件のドキュメントを読み込みました")
        except Exception as e:
            logger.error(f"ドキュメント読み込みエラー: {e}")
            return 1

        if not documents:
            logger.warning("ドキュメントが見つかりませんでした")
            return 1

        # 2. チャンク分割
        logger.info("チャンク分割中...")
        chunker = DocumentChunker()
        texts = [doc["content"] for doc in documents]
        chunks = chunker.chunk_documents(texts)
        logger.info(f"✅ {len(chunks)} 個のチャンクを作成しました")

        if not chunks:
            logger.warning("チャンクが作成されませんでした")
            return 1

        # 3. 埋め込み生成
        logger.info("埋め込み生成中...")
        embedding_manager = EmbeddingManager()
        embeddings = embedding_manager.embed_batch(chunks)
        logger.info(f"✅ 埋め込みを生成しました (shape: {embeddings.shape})")

        # 4. インデックス作成
        logger.info("インデックス作成中...")
        indexer = DocumentIndexer()
        index = indexer.create_index(chunks, embeddings)

        # 5. インデックス保存
        output_path = args.output or "rag_index.pkl"
        saved_path = indexer.save_index(index, output_path)
        logger.info(f"✅ インデックスを保存しました: {saved_path}")

        logger.info("=" * 60)
        logger.info("インジェスト完了")
        logger.info("=" * 60)

        return 0

    def _handle_query(self, args) -> int:
        """
        query コマンドを処理します。

        引数:
            args: パース済み引数

        戻り値:
            int: 終了コード
        """
        logger.info("=" * 60)
        logger.info(f"クエリ: {args.question}")
        logger.info("=" * 60)

        # RAGパイプライン初期化
        try:
            pipeline = RAGPipeline()
            pipeline.initialize(args.index)
            logger.info("✅ パイプライン初期化完了")
        except Exception as e:
            logger.error(f"パイプライン初期化エラー: {e}")
            return 1

        # クエリ実行
        try:
            result = pipeline.answer_query(args.question, top_k=args.top_k)
        except Exception as e:
            logger.error(f"クエリ実行エラー: {e}")
            return 1

        # 結果表示
        print("\n" + "=" * 60)
        print("【回答】")
        print("=" * 60)
        print(result["answer"])
        print("\n" + "=" * 60)
        print(f"【信頼度】 {result['confidence']:.3f}")
        print("=" * 60)
        print(f"【使用されたコンテキスト】 {len(result['context'])} 件")
        for i, context in enumerate(result["context"], 1):
            print(f"\n{i}. {context[:200]}...")
        print("=" * 60)

        return 0

    def _handle_search(self, args) -> int:
        """
        search コマンドを処理します。

        引数:
            args: パース済み引数

        戻り値:
            int: 終了コード
        """
        logger.info("=" * 60)
        logger.info(f"検索クエリ: {args.query}")
        logger.info("=" * 60)

        # Retriever初期化
        try:
            embedding_manager = EmbeddingManager()
            index = RAGIndex()

            if args.index:
                index.load(args.index)
            else:
                default_index = Path(INDEX_DIR) / "rag_index.pkl"
                index.load(str(default_index))

            retriever = Retriever(
                embedding_manager=embedding_manager,
                index=index,
            )
            logger.info("✅ Retriever 初期化完了")
        except Exception as e:
            logger.error(f"Retriever 初期化エラー: {e}")
            return 1

        # 検索実行
        try:
            results = retriever.retrieve_similar_chunks(args.query, top_k=args.top_k)
        except Exception as e:
            logger.error(f"検索エラー: {e}")
            return 1

        # 結果表示
        print("\n" + "=" * 60)
        print(f"【検索結果】 {len(results)} 件")
        print("=" * 60)

        if not results:
            print("関連するチャンクが見つかりませんでした")
        else:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. [スコア: {result['score']:.3f}]")
                print(f"   {result['chunk'][:200]}...")

        print("=" * 60)

        return 0

    def _handle_index(self, args) -> int:
        """
        index コマンドを処理します。

        引数:
            args: パース済み引数

        戻り値:
            int: 終了コード
        """
        if args.index_command == "rebuild":
            logger.info("インデックス再構築を開始します...")
            # ingest と同じ処理
            return self._handle_ingest(args)
        else:
            logger.error(f"Unknown index command: {args.index_command}")
            return 1

    def _handle_config(self, args) -> int:
        """
        config コマンドを処理します。

        引数:
            args: パース済み引数

        戻り値:
            int: 終了コード
        """
        if args.config_command == "show":
            print("\n" + "=" * 60)
            print("Mini-RAG システム設定")
            print("=" * 60)
            print(f"\n【埋め込み設定】")
            print(f"  モデル: {EMBEDDING_MODEL}")
            print(f"  次元数: {EMBEDDING_DIMENSION}")
            print(f"\n【チャンク設定】")
            print(f"  チャンクサイズ: {CHUNK_SIZE}")
            print(f"  オーバーラップ: {CHUNK_OVERLAP}")
            print(f"\n【検索設定】")
            print(f"  類似度閾値: {SIMILARITY_THRESHOLD}")
            print(f"  Top-K: {TOP_K}")
            print(f"\n【LLM設定】")
            print(f"  LLMタイプ: {LLM_TYPE}")
            print(f"\n【インデックス設定】")
            print(f"  インデックスディレクトリ: {INDEX_DIR}")
            print("=" * 60)

            return 0
        else:
            logger.error(f"Unknown config command: {args.config_command}")
            return 1


def main():
    """CLI のメインエントリーポイント。"""
    cli = CLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
