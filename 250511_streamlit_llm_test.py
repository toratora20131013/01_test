import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 定数設定 ---
# Geminiモデルの名前 (例: "gemini-pro", "gemini-1.0-pro", "gemini-1.5-pro-latest" など)
# ご利用可能なモデル名に合わせて変更してください。
GEMINI_MODEL_NAME = "gemini-2.0-flash"  # 例として最新のモデルを指定
TEMPERATURE_SETTING = 0.7  # 生成されるテキストのランダム性を調整 (0.0から1.0)

# --- LLMの初期化 ---
# 環境変数からAPIキーを取得
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# APIキーが設定されていない場合の処理
if not GEMINI_API_KEY:
    st.error("エラー: 環境変数 `GEMINI_API_KEY` が設定されていません。")
    st.stop()  # スクリプトの実行を停止

try:
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=TEMPERATURE_SETTING,
        convert_system_message_to_human=True # システムメッセージを人間からのメッセージとして扱う場合に設定
    )
except Exception as e:
    st.error(f"LLMの初期化中にエラーが発生しました: {e}")
    st.stop()

# --- Streamlit UI ---
st.title("💬 Gemini チャットボット")
st.caption(f"🚀 モデル: {GEMINI_MODEL_NAME}")

# プロンプト入力
prompt = st.text_area("メッセージを入力してください:", height=150)

# 実行ボタン
if st.button("送信"):
    if prompt:
        with st.spinner("AIが応答を生成中です..."):
            try:
                # LLMにプロンプトを送信し、応答を取得
                # invokeメソッドはHumanMessageなどのLangChainのメッセージオブジェクトを期待することがあるため、
                # シンプルな文字列で問題ないか、あるいはHumanMessageでラップするかは
                # langchain_google_genaiのバージョンや使い方によります。
                # ここではシンプルな文字列で試みます。
                response = llm.invoke(prompt)

                # 応答を表示
                st.subheader("🤖 AIの応答:")
                # responseオブジェクトがAIMessageオブジェクトの場合、.contentでテキストを取得
                if hasattr(response, 'content'):
                    st.markdown(response.content)
                else:
                    st.markdown(response) # 直接文字列が返ってくる場合

            except Exception as e:
                st.error(f"応答の生成中にエラーが発生しました: {e}")
    else:
        st.warning("メッセージを入力してください。")

# --- 注意事項 ---
st.sidebar.header("利用上の注意")
st.sidebar.info(
    "このチャットボットはGoogleのGeminiモデルを使用しています。\n"
    "APIキーが正しく設定されていることを確認してください。\n"
    "環境変数 `GEMINI_API_KEY` に有効なAPIキーを設定する必要があります。"
)
st.sidebar.markdown(
    "モデル名 (`GEMINI_MODEL_NAME`) や温度設定 (`TEMPERATURE_SETTING`) は、"
    "スクリプト内の定数を変更することで調整できます。"
)