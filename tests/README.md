# Mini-RAG テストスイート

包括的なテストスイートの使用方法と説明

## 📊 概要

Mini-RAG プロジェクトのテストスイートは、以下の4つのテストファイルで構成されています:

| ファイル | 行数 | 説明 | テスト数 |
|---------|------|------|--------|
| **test_basic.py** | 598 | 基本機能テスト | 50+ |
| **test_modules.py** | 452 | モジュール別テスト | 40+ |
| **test_integration.py** | 445 | 統合テスト | 45+ |
| **test_edge_cases.py** | 604 | エッジケーステスト | 60+ |
| **conftest.py** | 406 | 共有フィクスチャ | 40+ |

**合計**: 2,500行以上のテストコード

---

## 🚀 クイックスタート

### 依存関係のインストール

```bash
pip install -r requirements.txt
pip install pytest pytest-cov pytest-xdist
```

### すべてのテストを実行

```bash
# 基本実行
pytest tests/ -v

# カバレッジレポート付き
pytest tests/ -v --cov=src --cov-report=html

# 並列実行（高速）
pytest tests/ -v -n auto
```

### 特定のテストを実行

```bash
# 特定のテストクラス
pytest tests/test_basic.py::TestConfig -v

# 特定のテストメソッド
pytest tests/test_basic.py::TestConfig::test_config_exists -v

# マーカーで実行
pytest tests/ -m unit -v
pytest tests/ -m integration -v
pytest tests/ -m performance -v
```

---

## 📁 テストファイル構成

### test_basic.py - 基本機能テスト

基本的な動作確認とコンポーネントの単純な相互作用をテスト

```python
# テストクラス:
- TestConfig                  # 設定値の妥当性
- TestEmbeddings              # 埋め込み機能
- TestDocumentProcessing      # 文書処理
- TestRetriever               # 検索機能
- TestRAGPipeline             # RAGパイプライン
- TestIntegration             # 統合動作
- TestErrorHandling           # エラーハンドリング
- TestPerformance             # パフォーマンス
- TestDataValidation          # データ検証
- TestConfigurationIntegration # 設定の統合
```

**実行時間**: 〜10秒

### test_modules.py - モジュール別テスト

各モジュールの個別機能とAPIをテスト

```python
# テストクラス:
- TestConfigModule            # config.py の検証
- TestEmbeddingsModule        # embeddings.py モック
- TestIngestModule            # ingest.py モック
- TestRetrieverModule         # retriever.py モック
- TestRAGModule               # rag.py モック
- TestCLIModule               # cli.py モック
- TestModuleIntegration       # モジュール間の相互作用
- TestModuleErrorHandling     # モジュールエラー処理
```

**実行時間**: 〜8秒

### test_integration.py - 統合テスト

エンドツーエンドのワークフローと複合機能をテスト

```python
# テストクラス:
- TestDocumentPipeline        # ドキュメント処理パイプライン
- TestRAGPipeline             # RAGパイプライン全体
- TestRetrievalQuality        # 検索品質
- TestAnswerGeneration        # 回答生成
- TestEndToEndWorkflow        # エンドツーエンド
- TestPerformanceIntegration  # パフォーマンス統合
- TestRobustness              # 堅牢性
- TestConfigurationIntegration # 設定統合
- TestDataConsistency         # データ一貫性
```

**実行時間**: 〜15秒

### test_edge_cases.py - エッジケーステスト

異常入力、境界条件、エラーシナリオをテスト

```python
# テストクラス:
- TestInputValidation         # 入力検証
- TestDocumentProcessingEdgeCases  # 文書処理エッジケース
- TestChunkingEdgeCases       # チャンク分割エッジケース
- TestSimilarityEdgeCases     # 類似度計算エッジケース
- TestRetrievalEdgeCases      # 検索エッジケース
- TestAnswerGenerationEdgeCases # 回答生成エッジケース
- TestPerformanceEdgeCases    # パフォーマンスエッジケース
- TestFileSystemEdgeCases     # ファイルシステムエッジケース
- TestBoundaryConditions      # 境界条件
- TestRecoveryAndResilience   # 復旧と耐性
- TestConcurrencyEdgeCases    # 並行処理エッジケース
```

**実行時間**: 〜12秒

### conftest.py - 共有フィクスチャ

すべてのテストで使用される共有フィクスチャ

```python
# フィクスチャ（40+個）:

# セッションレベル
- test_data_dir              # テンポラリデータディレクトリ
- sample_documents           # サンプル文書
- sample_queries             # サンプルクエリ
- expected_configs           # 期待設定値

# モジュールレベル
- temp_index_dir             # インデックス用ディレクトリ
- temp_cache_dir             # キャッシュ用ディレクトリ
- docs_dir                   # ドキュメント用ディレクトリ

# 関数レベル
- sample_text                # サンプルテキスト
- sample_texts               # 複数のサンプルテキスト
- sample_chunks              # サンプルチャンク
- sample_embeddings          # サンプル埋め込み
- related_text               # 関連テキスト
- unrelated_text             # 無関連テキスト
- config_dict                # 設定辞書

# バリデータ
- validate_embedding         # 埋め込み検証関数
- validate_chunks            # チャンク検証関数
- validate_query_result      # クエリ結果検証関数
- validate_retrieval_result  # 検索結果検証関数

# モック
- mock_embedding_manager     # モック埋め込みマネージャ
- mock_retriever             # モック検索器
- mock_rag_pipeline          # モックRAGパイプライン

# ジェネレータ
- generate_random_embedding  # ランダム埋め込み生成
- generate_related_chunks    # 関連チャンク生成
```

---

## 🎯 マーカーベースのテスト実行

テストは以下のマーカーで分類されています:

### マーカー一覧

```bash
# ユニットテスト（高速、隔離）
pytest tests/ -m unit -v

# 統合テスト（モジュール間の相互作用）
pytest tests/ -m integration -v

# パフォーマンステスト
pytest tests/ -m performance -v

# 遅いテスト
pytest tests/ -m slow -v

# 埋め込みモデルが必要
pytest tests/ -m requires_model -v

# エッジケーステスト
pytest tests/ -m edge_case -v
```

### 複数マーカーの組み合わせ

```bash
# Unit AND not slow
pytest tests/ -m "unit and not slow" -v

# Integration OR performance
pytest tests/ -m "integration or performance" -v

# Edge cases excluding slow
pytest tests/ -m "edge_case and not slow" -v
```

---

## 📊 テスト実行パターン

### パターン1: 全テスト実行（フルスイート）

```bash
pytest tests/ -v --tb=short
```

**所要時間**: 〜45秒
**対象**: すべてのテスト（200+）
**用途**: CI/CD、完全検証

### パターン2: 高速テスト（ユニット + 基本）

```bash
pytest tests/ -m "unit or basic" -v
```

**所要時間**: 〜20秒
**対象**: ユニットテスト中心
**用途**: 開発中の頻繁な実行

### パターン3: 統合テストのみ

```bash
pytest tests/ -m integration -v
```

**所要時間**: 〜15秒
**対象**: エンドツーエンド機能
**用途**: 機能統合の検証

### パターン4: パフォーマンステスト

```bash
pytest tests/ -m performance -v
```

**所要時間**: 〜10秒
**対象**: パフォーマンス関連
**用途**: 速度最適化確認

### パターン5: エッジケース検証

```bash
pytest tests/ -m edge_case -v
```

**所要時間**: 〜12秒
**対象**: 異常系テスト
**用途**: 堅牢性確認

---

## 📈 カバレッジレポート

### カバレッジ生成

```bash
# HTMLレポート生成
pytest tests/ --cov=src --cov-report=html

# ターミナル出力
pytest tests/ --cov=src --cov-report=term

# XML出力（CI用）
pytest tests/ --cov=src --cov-report=xml
```

### レポート表示

```bash
# HTMLレポートを開く
open htmlcov/index.html  # macOS
# または
firefox htmlcov/index.html  # Linux
```

### 目標カバレッジ

- **Phase 1 実装時**: 70% 以上
- **Phase 2 実装時**: 80% 以上
- **本番環境**: 85% 以上

---

## 🐛 トラブルシューティング

### テスト実行エラー

#### エラー: ModuleNotFoundError

```bash
# 解決策: src を Python path に追加
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/ -v
```

#### エラー: import errors in conftest.py

```bash
# 解決策: 依存関係を確認
pip install -r requirements.txt
pip install pytest pytest-cov
```

#### エラー: Fixture 'xxx' not found

```bash
# 解決策: conftest.py が tests/ ディレクトリに存在するか確認
ls -la tests/conftest.py

# キャッシュをクリア
pytest --cache-clear tests/ -v
```

### テスト失敗デバッグ

#### 詳細ログを表示

```bash
pytest tests/test_basic.py::TestConfig::test_config_exists -v -s
```

#### パイディープスタック表示

```bash
pytest tests/ -v --tb=long
```

#### 最初の失敗で停止

```bash
pytest tests/ -x
```

#### 前回失敗したテストを実行

```bash
pytest tests/ --lf
```

---

## 🔄 Phase別テスト実行ガイド

### Phase 1（実装中）

すべてのモジュールが実装されるまで、テストはスキップロジックで動作します:

```bash
# Phase 1 テストのみ実行
pytest tests/test_basic.py tests/test_modules.py -v

# 進捗確認
pytest tests/ --tb=line -v | grep PASSED
```

### Phase 2（ファイル形式拡張）

新しいテストを追加してから実装:

```bash
# Phase 2 エッジケーステスト実行
pytest tests/test_edge_cases.py -m "edge_case" -v
```

### Phase 3（LLM統合）

統合テストで OpenAI API 相互作用を検証:

```bash
# LLM 統合テスト
pytest tests/test_integration.py::TestRAGPipeline -v
```

### Phase 4（パフォーマンス最適化）

パフォーマンステストで改善を測定:

```bash
pytest tests/ -m performance -v --durations=10
```

---

## 📋 チェックリスト

実装進捗の追跡:

### Phase 1 テスト要件

- [ ] embeddings.py テスト通過
- [ ] ingest.py テスト通過
- [ ] retriever.py テスト通過
- [ ] rag.py テスト通過
- [ ] cli.py テスト通過
- [ ] カバレッジ 70% 以上

### 品質保証

- [ ] 全ユニットテスト通過
- [ ] 全統合テスト通過
- [ ] 全エッジケーステスト通過
- [ ] パフォーマンス要件達成
- [ ] カバレッジ 80% 以上

---

## 🔧 テストカスタマイズ

### カスタムマーカー追加

`pytest.ini` に追加:

```ini
[pytest]
markers =
    custom_marker: description
```

テストで使用:

```python
@pytest.mark.custom_marker
def test_example():
    pass
```

### フィクスチャのカスタマイズ

`conftest.py` を編集:

```python
@pytest.fixture
def custom_fixture():
    # セットアップ
    yield value
    # クリーンアップ
```

---

## 📞 よくある質問

**Q: テストが遅い場合は？**
> A: 並列実行を使用:
> ```bash
> pip install pytest-xdist
> pytest tests/ -n auto
> ```

**Q: 特定のテストだけ実行したい**
> A: テストメソッド名で指定:
> ```bash
> pytest tests/test_basic.py::TestConfig::test_embedding_dimension -v
> ```

**Q: フィクスチャの内容を確認したい**
> A: テストに print を追加:
> ```python
> def test_example(sample_documents):
>     print(sample_documents)
>     assert True
>
> # 実行時: pytest test_example -s
> ```

**Q: 新しいテストを追加するには？**
> A: 適切なテストファイルに追加:
> ```python
> class TestNewFeature:
>     def test_something(self):
>         assert True
> ```

---

## 🎯 リファレンス

### pytest コマンドオプション

| オプション | 説明 |
|----------|------|
| `-v` | 詳細出力 |
| `-s` | print 出力を表示 |
| `-x` | 最初の失敗で停止 |
| `--tb=short` | 短いトレースバック |
| `--tb=long` | 長いトレースバック |
| `-k "pattern"` | パターンマッチング実行 |
| `-m "marker"` | マーカーで実行 |
| `--co` | テスト収集のみ |
| `--durations=10` | 遅いテスト Top 10 表示 |

### ファイアチェック

デバッグ時の便利なテクニック:

```bash
# テストのみ収集（実行しない）
pytest tests/ --collect-only

# テストカウント
pytest tests/ --collect-only -q | wc -l

# 失敗したテストのみ実行
pytest tests/ --lf

# 修正済みテストを実行
pytest tests/ --ff

# ランダム順序で実行
pytest tests/ --random-order
```

---

**最終更新**: 2025年12月

Happy Testing! 🚀
