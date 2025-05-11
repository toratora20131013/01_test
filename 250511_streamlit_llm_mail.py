import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
# from streamlit_copy_button import copy_button # ← この行を削除またはコメントアウト
from st_copy_to_clipboard import st_copy_to_clipboard # ← 新しいライブラリをインポート

# --- 定数設定 ---
GEMINI_MODEL_NAME = "gemini-2.0-flash"
TEMPERATURE_SETTING = 0.7

# --- LLMの初期化 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("エラー: 環境変数 `GEMINI_API_KEY` が設定されていません。")
    st.stop()

try:
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=TEMPERATURE_SETTING,
    )
except Exception as e:
    st.error(f"LLMの初期化中にエラーが発生しました: {e}")
    st.stop()

# --- LLMへの指示を生成する関数 (変更なし) ---
def create_reply_prompt_with_dynamic_address(original_email_text):
    prompt = f"""あなたは「〇〇株式会社の〇〇」です。以下の「元のメール」を受け取りました。

タスク：
1.  「元のメール」から、送信者の「会社名」と「氏名（敬称は不要、姓と名）」を特定してください。
2.  特定した情報に基づき、返信メールの冒頭に以下の形式で宛名を作成してください。
    `{{抽出した会社名}}　{{抽出した氏名}}様`
    もし会社名や氏名が特定できない場合は、それぞれ「XXXX株式会社」「XXXX様」としてください。
3.  作成した宛名の後に、以下の固定挨拶文を続けてください。
    ```
    いつもお世話になっております。
    〇〇株式会社の〇〇です。
    ```
4.  上記の挨拶文に続けて、「元のメール」に対する丁寧なビジネスメールの返信本文を作成してください。本文には、具体的な用件への返答や次のアクションなどを明確に記述してください。
5.  文末には、適切な結びの言葉とあなたの署名（「〇〇株式会社 〇〇」など）を自然に含めてください。

最終的な出力として、上記2〜5を結合した返信メール全体を生成してください。

元のメール:
---
{original_email_text}
---

返信メール全体:
"""
    return prompt

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📧 AIメール返信アシスタント (宛名抽出・コピー機能付き)")
st.caption(f"🚀 モデル: {GEMINI_MODEL_NAME}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("受信したメール")
    original_email = st.text_area(
        "ここに受信したメールの内容を貼り付けてください:",
        height=300,
        key="original_email_input"
    )

    if st.button("返信メールを作成", type="primary", key="generate_reply_button"):
        if original_email:
            with st.spinner("AIが宛名を抽出し、返信メールを作成中です..."):
                try:
                    reply_generation_prompt = create_reply_prompt_with_dynamic_address(original_email)
                    llm_response = llm.invoke(reply_generation_prompt)

                    full_reply_email = ""
                    if hasattr(llm_response, 'content'):
                        full_reply_email = llm_response.content.strip()
                    else:
                        full_reply_email = str(llm_response).strip()

                    st.session_state.reply_email_output = full_reply_email

                except Exception as e:
                    st.error(f"返信メールの生成中にエラーが発生しました: {e}")
                    st.session_state.reply_email_output = "エラーにより返信を生成できませんでした。"
        else:
            st.warning("返信を作成するには、元のメール内容を貼り付けてください。")
            st.session_state.reply_email_output = ""

with col2:
    st.subheader("生成された返信メール案")
    reply_output = st.session_state.get("reply_email_output", "")
    st.text_area(
        "AIが作成した返信メールの案です。特に宛名が正しく抽出されているか確認し、適宜修正してください:",
        value=reply_output,
        height=450,
        key="reply_email_display"
    )

    # 返信メールが空でない場合にコピーボタンを表示 (st_copy_to_clipboard を使用)
    if reply_output and reply_output != "エラーにより返信を生成できませんでした。":
        st_copy_to_clipboard(
            text=reply_output,                          # コピーするテキスト
            before_copy_label="返信メールをコピーする",     # コピー実行前のボタンのラベル
            after_copy_label="コピーしました！",          # コピー実行後のボタンのラベル (これがフィードバックになります)
            key="copy_reply_btn"                        # Streamlitウィジェットのキー
            # show_text=False (デフォルトのままなら指定不要)
        )
        # st.caption("上のボタンを押すと、上記の返信メール案がクリップボードにコピーされます。") # メッセージはコンポーネントが自動で出す場合が多い

# --- 注意事項 ---
st.sidebar.header("使い方と注意点")
st.sidebar.info(
    "1. 左側のテキストエリアに受信したメールを貼り付けます。\n"
    "2. 「返信メールを作成」ボタンを押します。\n"
    "3. 右側のテキストエリアにAIが作成した返信メールの案が表示されます。\n"
    "4. **AIによる会社名・氏名の抽出は完璧ではありません。必ず宛名と内容を確認し、必要に応じて修正・加筆してください。**\n"
    "5. 表示された返信メール案の下にある「返信メールをコピーする」ボタンを押すと、内容をクリップボードにコピーできます。\n"
    "6. 署名者（〇〇株式会社の〇〇です）は、現在スクリプト内で固定されています。変更する場合はプロンプトと固定挨拶文を編集してください。"
)
st.sidebar.markdown(
    "**固定の挨拶文（宛名の下に続く）:**\n"
    "```\n"
    "いつもお世話になっております。\n"
    "〇〇株式会社の〇〇です。\n"
    "```"
)