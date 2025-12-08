# 💬 チャットアプリケーション使い方

## 起動方法

```bash
# 1. チャットアプリケーションを起動
streamlit run app.py

# または仮想環境を使用
.venv/bin/streamlit run app.py
```

ブラウザで自動的に開きます（通常は http://localhost:8501）

## 機能

### 基本機能
- 💬 チャット形式での質問応答
- 📊 信頼度スコア表示
- 📝 参照したチャンクの表示
- 📜 チャット履歴の保持

### サイドバー設定
- インデックスファイルの選択
- 検索するチャンク数（Top-K）の調整

## 使い方

1. **質問を入力**
   - 画面下部の入力欄に質問を入力
   - 例：「1dayと2weekどっちがおすすめ？」

2. **回答を確認**
   - AIが自動的に関連情報を検索
   - テンプレートベースで回答を生成
   - 信頼度スコアで確度を確認

3. **詳細情報を確認**
   - 「詳細情報」をクリック
   - 参照したチャンクの内容を確認

## カスタマイズ

### インデックスを切り替える
サイドバーで別のインデックスファイルを指定できます：
- `contact_lens_index.pkl` - コンタクトレンズ専門
- `test_index.pkl` - テスト用
- その他、自分で作成したインデックス

### Top-K を調整
スライダーで検索するチャンク数を調整：
- 少ない（1-2）: より絞り込んだ回答
- 多い（5-10）: より広範囲の情報

## トラブルシューティング

### インデックスが見つからない
```
初期化エラー: Index file not found: ...
```
→ 先にインデックスを作成してください：
```bash
python -m src.cli ingest data/contact_lens_docs --output contact_lens_index.pkl
```

### ポートが使用中
```
Port 8501 is already in use
```
→ 別のポートで起動：
```bash
streamlit run app.py --server.port 8502
```

## デプロイ

### Streamlit Cloud にデプロイ
1. GitHubにpush
2. https://share.streamlit.io/ でデプロイ
3. 公開URLで誰でもアクセス可能に

### ローカルネットワークで共有
```bash
streamlit run app.py --server.address 0.0.0.0
```
→ 同じネットワーク内の他のデバイスからアクセス可能
