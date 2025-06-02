import streamlit as st
from langchain_aws import ChatBedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# --- 初期設定 ---
# AWS Bedrockの設定 (ご自身の環境に合わせて変更してください)
BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"  # 例: Claude 3 Sonnet
# BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0" # 例: Claude 3 Haiku (より高速・低コスト)
# BEDROCK_MODEL_ID = "meta.llama3-8b-instruct-v1:0" # 例: Llama 3 8B Instruct
AWS_REGION = "us-east-1"  # Bedrockが利用可能なリージョン (例: "us-east-1", "ap-northeast-1"など)

# --- LangChainのコアコンポーネントの初期化 ---

@st.cache_resource # LLMインスタンスはリソースとしてキャッシュ
def get_llm():
    """
    ChatBedrock LLMのインスタンスを初期化またはキャッシュから取得します。
    """
    try:
        llm = ChatBedrock(
            model_id=BEDROCK_MODEL_ID,
            region_name=AWS_REGION,
            # credentials_profile_name="your-aws-profile-name", # AWSプロファイル名 (必要な場合)
            model_kwargs={
                "max_tokens": 2048,  # モデルによってパラメータ名が異なる場合あり (例: max_tokens_to_sample)
                "temperature": 0.7,
            }
        )
        return llm
    except Exception as e:
        st.error(f"LLMの初期化に失敗しました: {e}")
        st.stop() # エラーが発生したらアプリを停止

def get_conversation_chain(llm):
    """
    ConversationChainをセッションステートで管理します。
    セッションごとに新しいメモリを持つチェーンが作成されます。
    """
    if "conversation_chain" not in st.session_state:
        memory = ConversationBufferMemory(return_messages=True) # LangChainのMessageオブジェクトで履歴を管理
        st.session_state.conversation_chain = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False # Streamlitアプリでは通常Falseで良い
        )
    return st.session_state.conversation_chain

# --- Streamlit アプリケーションのUIとロジック ---

st.set_page_config(page_title=" Bedrock チャット", layout="wide")
st.title(" Bedrock チャットボット 🤖")
st.caption(f"Powered by LangChain & Streamlit, using {BEDROCK_MODEL_ID}")

# LLMと会話チェーンの準備
llm = get_llm()
conversation_chain = get_conversation_chain(llm)

# セッションステートでチャット履歴を管理
if "messages" not in st.session_state:
    st.session_state.messages = [] # 例: [{"role": "user", "content": "こんにちは"}, {"role": "assistant", "content": "こんにちは！"}]

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力の受付と処理
if user_prompt := st.chat_input("メッセージを入力してください..."):
    # ユーザーのメッセージを履歴に追加し、表示
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # AIの応答を生成し、表示
    with st.chat_message("assistant"):
        try:
            with st.spinner("AIが考えています..."): # 応答待ちの間にスピナーを表示
                # LangChainのConversationChainを呼び出し
                response = conversation_chain.invoke({"input": user_prompt})
                ai_response = response.get('response', "申し訳ありません、応答を取得できませんでした。") # output_keyは 'response' がデフォルト

            st.markdown(ai_response)
            # AIの応答を履歴に追加
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- サイドバー ---
with st.sidebar:
    st.header("オプション")
    if st.button("会話履歴をリセット", key="reset_chat"):
        st.session_state.messages = []
        # ConversationChain とそのメモリもリセット
        if "conversation_chain" in st.session_state:
            del st.session_state.conversation_chain
        st.rerun() # 画面を再描画して変更を反映

    st.markdown("---")
    st.subheader("デバッグ情報")
    if st.checkbox("会話メモリを表示"):
        if "conversation_chain" in st.session_state and hasattr(st.session_state.conversation_chain.memory, "chat_memory"):
            st.write(st.session_state.conversation_chain.memory.chat_memory.messages)
        else:
            st.caption("メモリはまだ初期化されていません。")