# Mini-RAG 実装ロードマップ

実装方針に基づいた段階的な実装計画です。

## 📋 実装方針のまとめ

| 項目 | 決定 |
|------|------|
| **埋め込みモデル** | Sentence Transformers (all-MiniLM-L6-v2) |
| **チャンク分割** | パラグラフ + 文ハイブリッド |
| **インデックス** | ファイルベース (pickle + JSON) |
| **検索・ランキング** | コサイン類似度 |
| **LLM統合** | Phase 1: Template / Phase 2: OpenAI GPT |
| **エラーハンドリング** | デフォルト応答 (graceful) |
| **パフォーマンス** | キャッシング |
| **ファイル形式** | 複数形式 (txt, PDF, 画像等) |

---

## 🚀 実装フェーズ

### **Phase 1: MVP（基本機能）**
**目標**: テンプレートベースのRAGで動作確認

#### 1.1 `embeddings.py` - 埋め込み機能の実装

```python
# 実装内容:
- EmbeddingManager クラス
  - embed_text(text: str) -> np.ndarray
  - embed_batch(texts: List[str]) -> np.ndarray
  - cosine_similarity(vec1, vec2) -> float
- キャッシング機構
  - EmbeddingCache クラス
```

**テスト**: `test_modules.py::TestEmbeddingsModule` で検証

---

#### 1.2 `ingest.py` - 文書処理の実装

```python
# 実装内容:
- DocumentLoader クラス
  - load_documents(docs_dir) -> List[Document]
  - サポート形式: .txt, .md, .rst (テキストのみ)

- DocumentChunker クラス
  - chunk_documents(texts: List[str]) -> List[str]
  - 戦略: hierarchical (paragraph + sentence)
  - CHUNK_SIZE = 512, OVERLAP = 50

- DocumentIndexer クラス
  - create_index(chunks, embeddings) -> RAGIndex
```

**テスト**: `test_modules.py::TestIngestModule` で検証

---

#### 1.3 `retriever.py` - 検索機能の実装

```python
# 実装内容:
- RAGIndex クラス (ファイルベース)
  - add_chunks(chunks, embeddings)
  - retrieve(query_embedding, top_k=3) -> List[Result]
  - save(filepath)
  - load(filepath)

- Retriever クラス
  - retrieve_similar_chunks(query: str) -> List[str]
  - メトリック: cosine_similarity
```

**テスト**: `test_modules.py::TestRetrieverModule` で検証

---

#### 1.4 `rag.py` - RAGメインロジックの実装

```python
# 実装内容:
- RAGPipeline クラス
  - initialize(index_path, config)
  - answer_query(query: str) -> Dict[str, Any]
  - 内部フロー:
    1. query → embedding
    2. embedding → retrieve chunks
    3. chunks → template-based answer

- Template-based generation:
  - 関連チャンクを単純に組み合わせて回答生成
```

**テスト**: `test_modules.py::TestRAGModule` で検証

---

#### 1.5 `cli.py` - CLIの実装

```python
# 実装内容:
- Command-line interface
  - Commands:
    - rag ingest <docs_dir>
    - rag query <question>
    - rag search <query>
    - rag index rebuild
    - rag config show
```

**テスト**: `test_modules.py::TestCLIModule` で検証

---

### **Phase 2: ファイル形式対応**
**目標**: PDF, 画像等の複数ファイル形式に対応

#### 2.1 `ingest.py` 拡張 - PDF処理

```python
# 追加機能:
- PDFProcessor クラス
  - extract_text_from_pdf(pdf_path) -> str
  - extract_images_from_pdf(pdf_path) -> List[Image]

# 依存: PyPDF2, pdf2image

# テスト対象: test_modules.py に追加テスト
```

---

#### 2.2 `ingest.py` 拡張 - 画像OCR処理

```python
# 追加機能:
- ImageProcessor クラス
  - ocr_image(image_path) -> str
  - supported_formats: .png, .jpg, .jpeg, .gif

# 依存: pytesseract, Pillow, tesseract (system)

# インストール (Ubuntu/Debian):
# sudo apt-get install tesseract-ocr
# pip install pytesseract Pillow

# テスト対象: test_edge_cases.py に OCR テスト追加
```

---

#### 2.3 `ingest.py` 拡張 - その他ファイル形式

```python
# 追加機能:
- CSVProcessor: CSV → テキスト変換
- JSONProcessor: JSON → テキスト抽出
- DocxProcessor: Word → テキスト抽出
- PptxProcessor: PowerPoint → テキスト抽出

# 依存: python-docx, python-pptx, pandas

# テスト対象: 統合テスト test_integration.py に追加
```

---

### **Phase 3: LLM統合**
**目標**: OpenAI GPTによる高品質な回答生成

#### 3.1 `rag.py` 拡張 - OpenAI統合

```python
# 変更内容:
- LLMPipeline クラス
  - generate_with_openai(query, context) -> str
  - プロンプト構築
  - API呼び出し + エラーハンドリング
  - リトライロジック

# 設定:
# config.py で LLM_TYPE を "template" → "openai" に変更
# OPENAI_API_KEY 環境変数を設定

# 使用方法:
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from src.config import LLM_TYPE
# LLM_TYPE を "openai" に変更してから実行
```

**テスト**: `test_integration.py` に統合テスト追加

---

#### 3.2 エラーハンドリング - リトライロジック

```python
# 実装内容:
- Retry mechanism with exponential backoff
- RateLimitError, APIError のハンドリング
- フォールバック: template-based generation へ

# テスト対象: test_edge_cases.py に追加
```

---

### **Phase 4: パフォーマンス最適化**
**目標**: 大規模データセット対応

#### 4.1 キャッシング実装

```python
# 実装内容:
- EmbeddingCache (既に Phase 1 で実装)
- RetrievalCache
  - LRU キャッシュ
  - TTL (Time-to-Live) サポート

# テスト対象: test_modules.py に パフォーマンステスト追加
```

---

#### 4.2 バッチ処理 + 並行処理

```python
# 実装内容:
- Batch embedding generation
- Parallel document loading
- ThreadPoolExecutor / ProcessPoolExecutor 活用

# テスト対象: test_integration.py::TestPerformanceIntegration
```

---

#### 4.3 インデックス最適化（オプション）

```python
# 選択肢:
- FAISS への移行（< 1,000,000 チャンク）
- Annoy への移行（ローカルストレージ最適化）

# 後のマイルストーン
```

---

## 📊 実装進捗テンプレート

### Phase 1 進捗チェックリスト

- [ ] **1.1 embeddings.py 実装**
  - [ ] EmbeddingManager クラス作成
  - [ ] embed_text() メソッド実装
  - [ ] cosine_similarity() 実装
  - [ ] EmbeddingCache 実装
  - [ ] ユニットテスト通過

- [ ] **1.2 ingest.py 実装**
  - [ ] DocumentLoader クラス作成
  - [ ] .txt, .md, .rst 読み込み対応
  - [ ] DocumentChunker 実装 (hierarchical)
  - [ ] DocumentIndexer 実装
  - [ ] ユニットテスト通過

- [ ] **1.3 retriever.py 実装**
  - [ ] RAGIndex クラス作成 (ファイルベース)
  - [ ] retrieve() メソッド実装
  - [ ] save/load 機能
  - [ ] ユニットテスト通過

- [ ] **1.4 rag.py 実装**
  - [ ] RAGPipeline クラス作成
  - [ ] answer_query() メソッド実装
  - [ ] Template-based generation
  - [ ] エラーハンドリング
  - [ ] ユニットテスト通過

- [ ] **1.5 cli.py 実装**
  - [ ] コマンドパーサー実装
  - [ ] ingest コマンド
  - [ ] query コマンド
  - [ ] CLI統合テスト通過

---

## 🧪 テスト戦略

### Phase 1 テスト

```bash
# ユニットテスト実行
pytest tests/test_modules.py -v

# 統合テスト実行
pytest tests/test_integration.py -v -m integration

# エッジケーステスト
pytest tests/test_edge_cases.py -v

# カバレッジレポート
pytest --cov=src --cov-report=html
```

### 段階的テスト

1. **1.1完了後**: `test_modules.py::TestEmbeddingsModule` 通過
2. **1.2完了後**: `test_modules.py::TestIngestModule` 通過
3. **1.3完了後**: `test_modules.py::TestRetrieverModule` 通過
4. **1.4完了後**: `test_modules.py::TestRAGModule` 通過
5. **1.5完了後**: 全テスト通過

---

## 🔧 各モジュールの API 仕様

### embeddings.py

```python
from src.embeddings import EmbeddingManager, cosine_similarity

# 初期化
manager = EmbeddingManager()

# テキスト埋め込み
embedding = manager.embed_text("Pythonとは？")
# Returns: np.ndarray of shape (384,)

# バッチ埋め込み
embeddings = manager.embed_batch(["テキスト1", "テキスト2"])
# Returns: np.ndarray of shape (2, 384)

# 類似度計算
sim = cosine_similarity(emb1, emb2)
# Returns: float in [-1, 1]
```

---

### ingest.py

```python
from src.ingest import DocumentLoader, DocumentChunker, DocumentIndexer

# 1. 文書読み込み
loader = DocumentLoader()
documents = loader.load_documents("./data/docs")
# Returns: List[Document] where Document = {"path", "content"}

# 2. チャンク分割
chunker = DocumentChunker()
chunks = chunker.chunk_documents(documents)
# Returns: List[str]

# 3. インデックス作成
indexer = DocumentIndexer()
index = indexer.create_index(chunks, embeddings)
# Returns: RAGIndex object
```

---

### retriever.py

```python
from src.retriever import Retriever

# 初期化
retriever = Retriever(index_path="./data/index/rag_index.pkl")

# クエリ検索
results = retriever.retrieve_similar_chunks("Pythonとは？", top_k=3)
# Returns: List[{"chunk": str, "score": float, ...}]
```

---

### rag.py

```python
from src.rag import RAGPipeline

# 初期化
rag = RAGPipeline()

# クエリ処理
response = rag.answer_query("Pythonとは？")
# Returns: {
#   "answer": str,
#   "context": List[str],
#   "sources": List[str],
#   "confidence": float
# }
```

---

### cli.py

```bash
# コマンドラインから実行

# 1. 文書インジェスト
python -m src.cli ingest ./data/docs

# 2. クエリ実行
python -m src.cli query "Pythonとは？"

# 3. インデックス再構築
python -m src.cli index rebuild

# 4. 設定表示
python -m src.cli config show
```

---

## 📚 参考リソース

### 埋め込みモデル
- Sentence Transformers: https://www.sbert.net/
- all-MiniLM-L6-v2: Lightweight & Fast

### 文書処理
- PyPDF2: PDF処理
- pytesseract: OCR
- python-docx: Word処理
- python-pptx: PowerPoint処理

### LLM統合
- OpenAI API: https://platform.openai.com/docs/api-reference
- gpt-4o-mini: Cost-effective model

---

## ⚠️ 注意事項

### Phase 1 限定事項
- LLM統合はテンプレートベースのみ
- ファイル形式は .txt, .md, .rst のみ
- 埋め込みはCPUで実行

### パフォーマンス考慮
- CHUNK_SIZE = 512 で最適化済み
- キャッシングで高速化
- 10,000+ チャンクの場合は FAISS への移行を検討

### エラーハンドリング
- すべてのエラーはデフォルト応答で返却
- ログファイルに詳細出力
- ユーザーには分かりやすいメッセージを表示

---

## 🎯 完了時のマイルストーン

### Phase 1 完了時
✅ RAGシステムが動作
✅ 基本的なクエリ応答が可能
✅ ユニットテスト 100% 通過

### Phase 2 完了時
✅ 複数ファイル形式対応
✅ PDF/画像処理完全サポート

### Phase 3 完了時
✅ 高品質なLLM統合
✅ OpenAI GPT による自然な回答

### Phase 4 完了時
✅ 大規模データセット対応
✅ 高速検索機能

---

## 📝 次のステップ

1. **requirements.txt** をインストール
   ```bash
   pip install -r requirements.txt
   ```

2. **Phase 1** を実装
   - 各モジュールを順番に実装
   - テストを実行しながら進める

3. **設定の確認**
   ```bash
   python src/config.py
   ```

---

最終更新: 2024年12月
