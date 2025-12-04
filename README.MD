# Mini-RAG: Retrieval-Augmented Generation System

シンプルな構成で動く、学習用のミニ RAG（Retrieval-Augmented Generation）です。

## 📋 概要

Mini-RAG は、ユーザーが提供したドキュメントから関連情報を検索し、その情報に基づいて質問に回答するシステムです。

**主な特徴:**
- ✅ **軽量**: Sentence Transformers を使用した埋め込み
- ✅ **シンプル**: 依存関係を最小化
- ✅ **拡張性**: Phase 1～4 で段階的に機能拡張可能
- ✅ **テスト対応**: 2,500行以上の包括的なテストコード
- ✅ **多形式対応**: テキスト、PDF、画像（OCR）など対応予定

---

## 🚀 クイックスタート

### 1. インストール

```bash
pip install -r requirements.txt
```

### 2. 設定確認

```bash
python src/config.py
```

### 3. プロジェクト初期化

```bash
python main.py
```

### 4. ドキュメント配置

```bash
cp your_document.txt data/docs/
```

### 5. 実装開始

```bash
pytest tests/ -v
cat IMPLEMENTATION_ROADMAP.md
```

---

## 📁 プロジェクト構成

```text
mini-rag/
├── main.py                        プロジェクト初期化スクリプト
├── requirements.txt               依存パッケージ
├── pytest.ini                     テスト設定
├── README.MD                      このファイル
│
├── IMPLEMENTATION_ROADMAP.md      実装計画（詳細）
│
├── src/                           実装モジュール
│   ├── __init__.py
│   ├── config.py                  実装方針を定数化（⭐重要）
│   ├── embeddings.py              埋め込み機能（Phase 1）
│   ├── ingest.py                  文書処理（Phase 1）
│   ├── retriever.py               検索機能（Phase 1）
│   ├── rag.py                     RAG パイプライン（Phase 1）
│   └── cli.py                     CLI インターフェース（Phase 1）
│
├── data/
│   ├── docs/                      ドキュメント配置用
│   ├── index/                     インデックス（自動生成）
│   └── cache/                     キャッシュ（自動生成）
│
└── tests/                         テストスイート
    ├── README.md                  テスト説明書
    ├── conftest.py                共有フィクスチャ
    ├── test_basic.py              基本テスト
    ├── test_modules.py            モジュール別テスト
    ├── test_integration.py        統合テスト
    └── test_edge_cases.py         エッジケーステスト
```

---

## 🎯 実装方針

| # | 項目 | 決定 |
|---|------|------|
| 1️⃣ | **埋め込みモデル** | Sentence Transformers (all-MiniLM-L6-v2) |
| 2️⃣ | **文書分割** | パラグラフ + 文ハイブリッド (512文字) |
| 3️⃣ | **インデックス** | ファイルベース (pickle + JSON) |
| 4️⃣ | **検索・ランキング** | コサイン類似度 (threshold: 0.3, top_k: 3) |
| 5️⃣ | **LLM統合** | Phase 1: テンプレート / Phase 2: OpenAI |
| 6️⃣ | **エラー処理** | デフォルト応答（graceful mode） |
| 7️⃣ | **パフォーマンス** | キャッシング（埋め込み & 検索結果） |
| 8️⃣ | **ファイル形式** | 複数形式対応（txt, PDF, 画像等） |

詳細は [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) を参照。

---

## 🔧 実装フェーズ

### Phase 1: MVP（基本機能）
- ✅ embeddings.py - Sentence Transformers による埋め込み
- ✅ ingest.py - テキスト読み込みとハイブリッド分割
- ✅ retriever.py - ファイルベースのインデックスと検索
- ✅ rag.py - テンプレートベースの回答生成
- ✅ cli.py - コマンドラインインターフェース

### Phase 2: ファイル形式拡張
- PDF テキスト抽出
- 画像 OCR 処理
- Office ファイル対応

### Phase 3: LLM 統合
- OpenAI GPT-4o-mini による回答生成
- エラーハンドリング

### Phase 4: パフォーマンス最適化
- 高度なキャッシング
- インデックス最適化

---

## 📚 ドキュメント

| ドキュメント | 内容 |
|-------------|------|
| **[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)** | フェーズ別実装計画、API 仕様 |
| **[tests/README.md](tests/README.md)** | テスト戦略と実行方法 |
| **[src/config.py](src/config.py)** | 実装方針の定数化 |

---

## 🧪 テスト

```bash
# すべてのテストを実行
pytest tests/ -v

# 特定のテストのみ
pytest tests/test_modules.py::TestEmbeddingsModule -v

# カバレッジ
pytest --cov=src --cov-report=html
```

詳細は [tests/README.md](tests/README.md) を参照。

---

## 💻 使用方法

### 1. セットアップ

```bash
pip install -r requirements.txt
python src/config.py
python main.py
```

### 2. ドキュメント配置

```bash
echo "あなたのテキスト" > data/docs/document.txt
```

### 3. 実装開始

```bash
cat IMPLEMENTATION_ROADMAP.md
pytest tests/ -v
```

---

## ⚠️ 前提条件

- Python 3.8+
- pip または conda
- OCR 機能を使う場合は tesseract をインストール

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

---

## 📞 FAQ

**Q: どこから始めるか？**
> `python main.py` 実行後、`IMPLEMENTATION_ROADMAP.md` を読んで Phase 1 実装を開始してください。

**Q: LLM をすぐに使いたい**
> Phase 1 完了後、`src/config.py` で `LLM_TYPE = "openai"` に変更してください。

**Q: テストはいつ実行する？**
> 各モジュール実装後すぐに実行してください。

---

## 🎓 学習内容

このプロジェクトで学べること:
- RAG アーキテクチャの基礎
- 埋め込みモデルによるテキスト表現
- テキスト処理と日本語対応
- ベクトル検索と類似度計算
- LLM 統合
- テスト駆動開発

---

## 📝 ライセンス

学習用プロジェクト

---

## 🎉 次のステップ

```bash
# 1. セットアップ
pip install -r requirements.txt
python src/config.py

# 2. 実装計画確認
cat IMPLEMENTATION_ROADMAP.md

# 3. Phase 1 実装開始
# embeddings.py → ingest.py → retriever.py → rag.py → cli.py

# 4. テスト実行
pytest tests/ -v
```

**Happy Coding! 🚀**