# GitHub Issues リスト

IMPLEMENTATION_ROADMAP.md に基づいた GitHub issues の一覧と作成手順です。

---

## 🚀 Phase 1: MVP（基本機能）

### Issue 1.1: embeddings.py - 埋め込み機能の実装

```markdown
Title: [Phase 1] embeddings.py - 埋め込み機能の実装

Body:
## 実装内容

Sentence Transformers を使用した埋め込み機能の実装

### 実装モジュール
- EmbeddingManager クラス
  - embed_text(text: str) -> np.ndarray
  - embed_batch(texts: List[str]) -> np.ndarray
  - cosine_similarity(vec1, vec2) -> float
- EmbeddingCache クラス

### テスト対象
- tests/test_modules.py::TestEmbeddingsModule

### チェックリスト
- [ ] EmbeddingManager クラス実装
- [ ] embed_text() メソッド実装
- [ ] embed_batch() メソッド実装
- [ ] cosine_similarity() 実装
- [ ] EmbeddingCache 実装
- [ ] ユニットテスト 100% pass
- [ ] docstring 追加

### 依存
- sentence-transformers >= 2.2.0
- numpy >= 1.21.0

### 詳細は
[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md#11-embeddingspy---埋め込み機能の実装)
```

---

### Issue 1.2: ingest.py - 文書処理の実装

```markdown
Title: [Phase 1] ingest.py - 文書処理の実装

Body:
## 実装内容

DocumentLoader, DocumentChunker, DocumentIndexer の実装

### 実装モジュール
- DocumentLoader クラス
  - load_documents(docs_dir) -> List[Document]
  - サポート形式: .txt, .md, .rst

- DocumentChunker クラス
  - chunk_documents(texts: List[str]) -> List[str]
  - 戦略: hierarchical (paragraph + sentence)
  - CHUNK_SIZE = 512, OVERLAP = 50

- DocumentIndexer クラス
  - create_index(chunks, embeddings) -> RAGIndex

### テスト対象
- tests/test_modules.py::TestIngestModule

### チェックリスト
- [ ] DocumentLoader クラス実装
- [ ] DocumentChunker クラス実装
- [ ] DocumentIndexer クラス実装
- [ ] ユニットテスト 100% pass
- [ ] docstring 追加
- [ ] エラーハンドリング実装

### 詳細は
[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md#12-ingestpy---文書処理の実装)
```

---

### Issue 1.3: retriever.py - 検索機能の実装

```markdown
Title: [Phase 1] retriever.py - 検索機能の実装

Body:
## 実装内容

ファイルベースのインデックスと検索機能の実装

### 実装モジュール
- RAGIndex クラス (ファイルベース)
  - add_chunks(chunks, embeddings)
  - retrieve(query_embedding, top_k=3) -> List[Result]
  - save(filepath)
  - load(filepath)

- Retriever クラス
  - retrieve_similar_chunks(query: str) -> List[str]
  - メトリック: cosine_similarity

### テスト対象
- tests/test_modules.py::TestRetrieverModule

### チェックリスト
- [ ] RAGIndex クラス実装
- [ ] Retriever クラス実装
- [ ] ファイル I/O 実装
- [ ] コサイン類似度計算実装
- [ ] ユニットテスト 100% pass
- [ ] docstring 追加

### 依存
- Issue 1.1 完了（EmbeddingManager）

### 詳細は
[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md#13-retrieverpy---検索機能の実装)
```

---

### Issue 1.4: rag.py - RAGメインロジック実装

```markdown
Title: [Phase 1] rag.py - RAGメインロジック実装

Body:
## 実装内容

RAGPipeline とテンプレートベース回答生成の実装

### 実装モジュール
- RAGPipeline クラス
  - initialize(index_path, config)
  - answer_query(query: str) -> Dict[str, Any]

- Template-based generation:
  - 関連チャンクを組み合わせて回答生成

### テスト対象
- tests/test_modules.py::TestRAGModule

### チェックリスト
- [ ] RAGPipeline クラス実装
- [ ] initialize() メソッド実装
- [ ] answer_query() メソッド実装
- [ ] テンプレートベース生成実装
- [ ] エラーハンドリング実装
- [ ] ユニットテスト 100% pass
- [ ] docstring 追加

### 依存
- Issue 1.1 完了（Embeddings）
- Issue 1.3 完了（Retriever）

### 詳細は
[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md#14-ragpy---ragメインロジックの実装)
```

---

### Issue 1.5: cli.py - CLI実装

```markdown
Title: [Phase 1] cli.py - CLIインターフェース実装

Body:
## 実装内容

コマンドラインインターフェースの実装

### 実装コマンド
- rag ingest <docs_dir>
- rag query <question>
- rag search <query>
- rag index rebuild
- rag config show

### テスト対象
- tests/test_modules.py::TestCLIModule

### チェックリスト
- [ ] コマンドパーサー実装
- [ ] ingest コマンド実装
- [ ] query コマンド実装
- [ ] search コマンド実装
- [ ] index rebuild コマンド実装
- [ ] config show コマンド実装
- [ ] ユニットテスト 100% pass
- [ ] ヘルプメッセージ追加

### 依存
- Issue 1.1 〜 1.4 完了

### 詳細は
[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md#15-clipy---cliの実装)
```

---

## 📂 Phase 2: ファイル形式対応

### Issue 2.1: ingest.py 拡張 - PDF処理

```markdown
Title: [Phase 2] ingest.py 拡張 - PDF処理

Body:
## 実装内容

PDF ファイルのテキスト抽出機能を追加

### 実装内容
- PDFProcessor クラス
  - extract_text_from_pdf(pdf_path) -> str
  - extract_images_from_pdf(pdf_path) -> List[Image]

### テスト対象
- tests/test_edge_cases.py に PDF テスト追加

### チェックリスト
- [ ] PDFProcessor クラス実装
- [ ] テキスト抽出実装
- [ ] 画像抽出実装
- [ ] エラーハンドリング
- [ ] テスト追加
- [ ] docstring 追加

### 依存
- PyPDF2 >= 3.0.0
- pdf2image
- Issue 1.2 完了

### 詳細は
[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md#21-ingestpy-拡張---pdf処理)
```

---

### Issue 2.2: ingest.py 拡張 - 画像OCR処理

```markdown
Title: [Phase 2] ingest.py 拡張 - 画像OCR処理

Body:
## 実装内容

画像ファイルのOCR処理機能を追加

### 実装内容
- ImageProcessor クラス
  - ocr_image(image_path) -> str
  - supported_formats: .png, .jpg, .jpeg, .gif

### テスト対象
- tests/test_edge_cases.py に OCR テスト追加

### チェックリスト
- [ ] ImageProcessor クラス実装
- [ ] OCR 処理実装
- [ ] 複数形式サポート
- [ ] エラーハンドリング
- [ ] テスト追加
- [ ] docstring 追加

### システム依存
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

### Python依存
- pytesseract >= 0.3.10
- Pillow >= 9.0.0

### 詳細は
[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md#22-ingestpy-拡張---画像ocr処理)
```

---

### Issue 2.3: ingest.py 拡張 - その他ファイル形式

```markdown
Title: [Phase 2] ingest.py 拡張 - その他ファイル形式

Body:
## 実装内容

CSV, JSON, Word, PowerPoint 形式対応

### 実装内容
- CSVProcessor: CSV → テキスト変換
- JSONProcessor: JSON → テキスト抽出
- DocxProcessor: Word → テキスト抽出
- PptxProcessor: PowerPoint → テキスト抽出

### テスト対象
- tests/test_integration.py に統合テスト追加

### チェックリスト
- [ ] CSVProcessor 実装
- [ ] JSONProcessor 実装
- [ ] DocxProcessor 実装
- [ ] PptxProcessor 実装
- [ ] テスト追加
- [ ] docstring 追加

### 依存
- python-docx
- python-pptx
- pandas >= 1.3.0

### 詳細は
[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md#23-ingestpy-拡張---その他ファイル形式)
```

---

## 🧠 Phase 3: LLM統合

### Issue 3.1: rag.py 拡張 - OpenAI統合

```markdown
Title: [Phase 3] rag.py 拡張 - OpenAI統合

Body:
## 実装内容

OpenAI GPT-4o-mini による高品質な回答生成

### 実装内容
- LLMPipeline クラス
  - generate_with_openai(query, context) -> str
  - プロンプト構築
  - API呼び出し + エラーハンドリング
  - リトライロジック

### テスト対象
- tests/test_integration.py に統合テスト追加

### チェックリスト
- [ ] LLMPipeline クラス実装
- [ ] プロンプト構築実装
- [ ] API呼び出し実装
- [ ] リトライロジック実装
- [ ] エラーハンドリング
- [ ] テスト追加
- [ ] docstring 追加

### 設定
- config.py で LLM_TYPE を "openai" に変更
- OPENAI_API_KEY 環境変数を設定

### 依存
- openai >= 1.0.0

### 詳細は
[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md#31-ragpy-拡張---openai統合)
```

---

### Issue 3.2: エラーハンドリング - リトライロジック

```markdown
Title: [Phase 3] エラーハンドリング - リトライロジック

Body:
## 実装内容

OpenAI API エラーの堅牢なハンドリング

### 実装内容
- Retry mechanism with exponential backoff
- RateLimitError, APIError のハンドリング
- フォールバック: template-based generation へ

### テスト対象
- tests/test_edge_cases.py に追加

### チェックリスト
- [ ] リトライ機構実装
- [ ] Exponential backoff 実装
- [ ] エラーハンドリング実装
- [ ] フォールバック実装
- [ ] テスト追加

### 詳細は
[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md#32-エラーハンドリング---リトライロジック)
```

---

## ⚡ Phase 4: パフォーマンス最適化

### Issue 4.1: キャッシング実装

```markdown
Title: [Phase 4] キャッシング実装

Body:
## 実装内容

高度なキャッシング機構の実装

### 実装内容
- EmbeddingCache (既に Phase 1 で実装)
- RetrievalCache
  - LRU キャッシュ
  - TTL (Time-to-Live) サポート

### テスト対象
- tests/test_modules.py にパフォーマンステスト追加

### チェックリスト
- [ ] RetrievalCache クラス実装
- [ ] LRU キャッシュ実装
- [ ] TTL サポート実装
- [ ] キャッシュ管理実装
- [ ] テスト追加
- [ ] docstring 追加

### 詳細は
[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md#41-キャッシング実装)
```

---

### Issue 4.2: バッチ処理 + 並行処理

```markdown
Title: [Phase 4] バッチ処理 + 並行処理

Body:
## 実装内容

大規模データセット対応の並行処理実装

### 実装内容
- Batch embedding generation
- Parallel document loading
- ThreadPoolExecutor / ProcessPoolExecutor 活用

### テスト対象
- tests/test_integration.py::TestPerformanceIntegration

### チェックリスト
- [ ] バッチ埋め込み実装
- [ ] 並行ドキュメント読み込み実装
- [ ] ThreadPoolExecutor 活用
- [ ] パフォーマンステスト追加
- [ ] docstring 追加

### 詳細は
[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md#42-バッチ処理--並行処理)
```

---

### Issue 4.3: インデックス最適化（オプション）

```markdown
Title: [Phase 4] インデックス最適化（オプション）

Body:
## 実装内容

大規模インデックス向けの最適化

### 選択肢
- FAISS への移行（< 1,000,000 チャンク）
- Annoy への移行（ローカルストレージ最適化）

### テスト対象
- パフォーマンスベンチマーク

### チェックリスト
- [ ] FAISS または Annoy 選定
- [ ] インデックス移行実装
- [ ] パフォーマンス検証
- [ ] ドキュメント更新

### 詳細は
[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md#43-インデックス最適化オプション)
```

---

## 📌 Issue 作成手順

### GitHub Web UI で手動作成

1. リポジトリの **Issues** タブを開く
2. **New issue** をクリック
3. 上記の **Title** と **Body** をコピペ
4. **Labels** に `Phase X`, `implementation` を追加
5. **Submit new issue** をクリック

### GitHub CLI で自動作成

```bash
# 例: Issue 1.1 を作成
gh issue create \
  --title "[Phase 1] embeddings.py - 埋め込み機能の実装" \
  --body "Sentence Transformers を使用した埋め込み機能の実装..." \
  --label "Phase 1" \
  --label "implementation"
```

---

## 📊 総計

- **Phase 1**: 5 issues
- **Phase 2**: 3 issues
- **Phase 3**: 2 issues
- **Phase 4**: 3 issues

**合計: 13 issues**

---

詳細は [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) を参照。
