import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_aws import ChatBedrock

# --- AWS認証情報とリージョンの設定 ---
# LangChain (boto3) は自動的に以下の環境変数を参照します。
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_SESSION_TOKEN (任意)
# - AWS_DEFAULT_REGION
#
# そのため、コード内でキーを直接設定する必要はありません。
# アプリ起動前に環境変数が設定されているかどうかのガイドを表示します。
if not all(k in os.environ for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]):
    st.error("AWSの認証情報またはリージョンが環境変数に設定されていません。")
    st.info("""
    アプリを実行する前に、ターミナルで以下の環境変数を設定してください。
    BedrockでClaude 3モデルが利用可能なリージョン（例: `us-east-1`）を指定する必要があります。

    **macOS / Linux:**
    ```bash
    export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
    export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
    export AWS_DEFAULT_REGION="us-east-1"
    # AWS STS tạm thờiの認証情報を使用している場合は以下も設定
    # export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"
    ```

    **Windows (コマンドプロンプト):**
    ```bash
    set AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
    set AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
    set AWS_DEFAULT_REGION="us-east-1"
    ```
    """)
    st.stop()


# --- ページ設定 ---
st.set_page_config(
    page_title="🤖 AWS Bedrock (Claude) チャットボット",
    page_icon="🤖",
    layout="centered"
)


# --- サイドバーの設定 ---
st.sidebar.title("⚙️ 設定")

st.sidebar.markdown("### モデル選択")
# Bedrockで利用可能なClaudeモデルのIDを指定
selected_model = st.sidebar.selectbox(
    "使用するモデルを選んでください",
    (
        "anthropic.claude-3-haiku-v1:0",
        "anthropic.claude-3-sonnet-v1:0",
        "anthropic.claude-3-opus-v1:0",
        "anthropic.claude-v2:1" # 参考: 旧世代モデル
    ),
    index=1 # デフォルトはSonnet
)

st.sidebar.markdown("### パラメータ設定")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                                help="値が低いほど決定的で、高いほど多様な応答になります。")
# BedrockのClaudeモデルは `max_tokens` というパラメータ名を使用
max_tokens = st.sidebar.slider("Max Tokens", min_value=256, max_value=20000, value=4096, step=128,
                               help="応答として生成されるトークンの最大数です。")

st.sidebar.markdown("---")

st.sidebar.markdown("### 操作")
if st.sidebar.button("会話履歴をリセット"):
    st.session_state.messages = [
        AIMessage(content="こんにちは！AWS BedrockのClaudeです。会話がリセットされました。何かお手伝いできますか？")
    ]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("このアプリはAWS Bedrock上のClaudeモデルを使用しています。")


# --- メイン画面 ---
st.title("🤖 AWS Bedrock (Claude) チャットボット")
st.caption(f"選択中モデル: `{selected_model}`")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="こんにちは！AWS BedrockのClaudeです。何かお手伝いできることはありますか？")
    ]

# 履歴の表示
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

# ユーザーの入力を待つ
if prompt := st.chat_input("メッセージを入力してください..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AIが考え中です..."):
            try:
                # Bedrockモデルを初期化
                llm = ChatBedrock(
                    # credentials_profile_name="your-profile-name", # 名前付きプロファイルを使いたい場合
                    model_id=selected_model,
                    model_kwargs={
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                )

                # LLMを呼び出し
                response = llm.invoke(st.session_state.messages)
                response_content = response.content

                # 応答を履歴に追加
                st.session_state.messages.append(AIMessage(content=response_content))

                st.markdown(response_content)

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                st.info("AWSの認証情報が正しいか、また選択したモデル（`{selected_model}`）へのアクセスがBedrockコンソールで有効になっているか確認してください。")