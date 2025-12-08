"""
Mini-RAG チャットアプリケーション (Streamlit版)

RAGシステムを使用した質問応答チャットボット
"""

import streamlit as st
from pathlib import Path

from src.rag import RAGPipeline
from src.config import INDEX_DIR

# ページ設定
st.set_page_config(
    page_title="コンタクトレンズ相談チャット",
    page_icon="👁️",
    layout="wide"
)

# タイトル
st.title("コンタクトレンズ相談チャット")
st.markdown("コンタクトレンズに関する質問にお答えします！")

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    
    # インデックス選択
    index_file = st.text_input(
        "インデックスファイル",
        value="contact_lens_index.pkl"
    )
    
    # Top-K設定
    top_k = st.slider("検索するチャンク数", 1, 10, 3)
    
    # 信頼度閾値表示
    st.info("現在の類似度閾値: 0.3")
    
    st.markdown("---")
    st.markdown("### 💡 使い方")
    st.markdown("""
    1. 質問を入力
    2. 送信ボタンをクリック
    3. RAGが関連情報を検索
    4. 回答が表示されます
    """)

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

# パイプライン初期化
@st.cache_resource
def init_pipeline(index_file):
    """RAGパイプラインを初期化"""
    try:
        pipeline = RAGPipeline()
        index_path = Path(INDEX_DIR) / index_file
        pipeline.initialize(str(index_path))
        return pipeline, None
    except Exception as e:
        return None, str(e)

# パイプライン初期化（初回のみ）
if st.session_state.pipeline is None:
    with st.spinner("RAGシステムを初期化中..."):
        pipeline, error = init_pipeline(index_file)
        if error:
            st.error(f"初期化エラー: {error}")
            st.stop()
        else:
            st.session_state.pipeline = pipeline
            st.success("✅ RAGシステム準備完了")

# チャット履歴表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # アシスタントの回答の場合、詳細情報も表示
        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("📊 詳細情報"):
                st.markdown(f"**信頼度**: {message['metadata']['confidence']:.3f}")
                st.markdown(f"**使用されたコンテキスト**: {message['metadata']['num_contexts']}件")

# チャット入力
if prompt := st.chat_input("コンタクトレンズについて質問してください..."):
    # ユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # アシスタントの応答を生成
    with st.chat_message("assistant"):
        with st.spinner("回答を生成中..."):
            try:
                # RAGで回答生成
                result = st.session_state.pipeline.answer_query(prompt, top_k=top_k)
                
                # 回答表示
                st.markdown(result["answer"])
                
                # 詳細情報
                with st.expander("📊 詳細情報"):
                    st.markdown(f"**信頼度**: {result['confidence']:.3f}")
                    st.markdown(f"**使用されたコンテキスト**: {len(result['context'])}件")
                    
                    # コンテキスト表示
                    st.markdown("**参照したチャンク:**")
                    for i, ctx in enumerate(result['context'], 1):
                        st.text_area(
                            f"チャンク {i}",
                            value=ctx,
                            height=100,
                            key=f"ctx_{len(st.session_state.messages)}_{i}"
                        )
                
                # アシスタントメッセージを保存
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": {
                        "confidence": result["confidence"],
                        "num_contexts": len(result['context'])
                    }
                })
                
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

# フッター
st.markdown("---")
st.markdown("**Mini-RAG** - Retrieval-Augmented Generation システム")
