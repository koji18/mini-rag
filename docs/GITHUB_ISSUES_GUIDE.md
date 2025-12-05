# GitHub Issues 作成ガイド

IMPLEMENTATION_ROADMAP.md に基づいた GitHub issues を作成するガイド

---

## 📋 概要

このプロジェクトには **13個の GitHub issues** があります：

- **Phase 1 (MVP)**: 5 issues
- **Phase 2 (ファイル形式)**: 3 issues
- **Phase 3 (LLM統合)**: 2 issues
- **Phase 4 (パフォーマンス)**: 3 issues

---

## 🚀 Issues 作成方法

### 方法1: 自動作成（推奨）

GitHub CLI を使用して一括作成します：

```bash
# 実行前に GitHub リポジトリが remote に設定されていることを確認
git remote -v

# Issues を一括作成
bash scripts/create_issues.sh
```

**前提条件:**
```bash
# GitHub CLI をインストール
# macOS
brew install gh

# Ubuntu/Debian
sudo apt-get install gh

# 認証設定
gh auth login
```

### 方法2: 手動作成

GitHub Web UI で直接作成します：

1. **リポジトリの Issues タブを開く**
   - https://github.com/your-user/mini-rag/issues

2. **"New issue" をクリック**

3. **ISSUES.md の内容をコピペ**
   - [ISSUES.md](../ISSUES.md) から該当 issue の Title と Body をコピペ

4. **ラベルを追加**
   - `Phase X` (例: `Phase 1`, `Phase 2`)
   - `implementation`

5. **Submit をクリック**

---

## 📌 Phase 別 Issues

### Phase 1: MVP（基本機能）

| # | Issue | 実装内容 | テスト |
|---|-------|--------|------|
| 1.1 | embeddings.py | 埋め込み機能 | `test_modules.py::TestEmbeddingsModule` |
| 1.2 | ingest.py | 文書処理 | `test_modules.py::TestIngestModule` |
| 1.3 | retriever.py | 検索機能 | `test_modules.py::TestRetrieverModule` |
| 1.4 | rag.py | RAGパイプライン | `test_modules.py::TestRAGModule` |
| 1.5 | cli.py | CLIインターフェース | `test_modules.py::TestCLIModule` |

**実装順序:**
```
1.1 embeddings.py
  ↓
1.2 ingest.py
  ↓
1.3 retriever.py
  ↓
1.4 rag.py
  ↓
1.5 cli.py
```

### Phase 2: ファイル形式対応

| # | Issue | 実装内容 | テスト |
|---|-------|--------|------|
| 2.1 | PDF処理 | PDF テキスト抽出 | `test_edge_cases.py` |
| 2.2 | OCR処理 | 画像 OCR | `test_edge_cases.py` |
| 2.3 | その他形式 | CSV/JSON/Word/PowerPoint | `test_integration.py` |

### Phase 3: LLM統合

| # | Issue | 実装内容 | テスト |
|---|-------|--------|------|
| 3.1 | OpenAI統合 | GPT-4o-mini 連携 | `test_integration.py` |
| 3.2 | エラーハンドリング | リトライロジック | `test_edge_cases.py` |

### Phase 4: パフォーマンス最適化

| # | Issue | 実装内容 | テスト |
|---|-------|--------|------|
| 4.1 | キャッシング | LRU + TTL | `test_modules.py` |
| 4.2 | バッチ処理 | 並行処理実装 | `test_integration.py` |
| 4.3 | インデックス最適化 | FAISS/Annoy移行 | パフォーマンス測定 |

---

## ✅ Issue 完了の確認

各 issue が完了したかどうかを確認する方法：

```bash
# 特定の Phase 1 issue で言及されているテストを実行
pytest tests/test_modules.py::TestEmbeddingsModule -v

# 全テスト実行
pytest tests/ -v

# カバレッジ確認
pytest tests/ --cov=src --cov-report=html
```

### チェックリスト

各 issue には以下のチェックリストがあります：

- [ ] コード実装完了
- [ ] ユニットテスト追加・更新
- [ ] docstring 記載
- [ ] エラーハンドリング実装
- [ ] テスト 100% pass
- [ ] コードレビュー承認（チーム開発時）

---

## 🔄 Issue の進捗管理

### Issue の状態遷移

```
Open → In Progress → In Review → Closed
```

### ステータス確認コマンド

```bash
# すべての Open issues を表示
gh issue list

# 特定の Phase の issues を表示
gh issue list --label "Phase 1"

# Closed issues を表示
gh issue list --state closed
```

### Issue の更新

```bash
# Issue にコメント追加
gh issue comment <issue-number> -b "実装完了しました"

# Issue をクローズ
gh issue close <issue-number>

# Issue に PR をリンク
gh issue close <issue-number> --comment "Closes #<issue-number>" # PR から実行
```

---

## 🏷️ ラベルの説明

このプロジェクトで使用されるラベル：

| ラベル | 説明 |
|--------|------|
| `Phase 1` | MVP（基本機能）実装 |
| `Phase 2` | ファイル形式拡張 |
| `Phase 3` | LLM統合 |
| `Phase 4` | パフォーマンス最適化 |
| `implementation` | 実装タスク |
| `bug` | バグ報告 |
| `enhancement` | 改善提案 |
| `documentation` | ドキュメント |

---

## 📊 進捗ダッシュボード

GitHub Projects を使用して進捗を管理：

```bash
# GitHub Web UI で Project を作成
# 1. Projects タブ → New project
# 2. "Roadmap" テンプレートを選択
# 3. Phase 1-4 をカラムとして追加
# 4. Issues をドラッグ&ドロップで管理
```

---

## 🔗 関連リソース

- [IMPLEMENTATION_ROADMAP.md](../IMPLEMENTATION_ROADMAP.md) - 詳細な実装計画
- [ISSUES.md](../ISSUES.md) - Issue リストテンプレート
- [tests/README.md](../tests/README.md) - テスト戦略
- [src/config.py](../src/config.py) - 実装方針定数

---

## 💡 ベストプラクティス

### Issue 作成時

✅ **良い例:**
```markdown
Title: [Phase 1] embeddings.py - 埋め込み機能の実装

Body:
## 実装内容
明確に何を実装するかを記述

## テスト対象
どのテストで検証するか記述

## チェックリスト
- [ ] Item 1
- [ ] Item 2
```

❌ **悪い例:**
```
Title: 埋め込み機能
Body: 実装してください
```

### PR との連携

```bash
# Issue を PR で解決する場合
gh pr create --title "[Phase 1] Implement embeddings.py" \
  --body "Closes #1"  # Issue 番号を参照
```

PR がマージされると、自動的に関連 issue がクローズされます。

---

## 🆘 トラブルシューティング

### gh コマンドが見つからない

```bash
# インストール確認
which gh

# インストール（macOS）
brew install gh

# インストール（Ubuntu）
sudo apt-get install gh
```

### 認証エラー

```bash
# 再度認証ログイン
gh auth logout
gh auth login

# 認証状態確認
gh auth status
```

### Issue 作成スクリプトが失敗

```bash
# スクリプトの実行権限を確認
ls -la scripts/create_issues.sh

# 権限を付与
chmod +x scripts/create_issues.sh

# デバッグモードで実行
bash -x scripts/create_issues.sh
```

---

## 📝 まとめ

1. **Issues 自動作成**: `bash scripts/create_issues.sh`
2. **進捗確認**: `gh issue list`
3. **テスト実行**: `pytest tests/ -v`
4. **完了後**: PR を作成して issue をクローズ

---

Happy Coding! 🚀
