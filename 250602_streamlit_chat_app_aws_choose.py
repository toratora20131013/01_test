import streamlit as st
import os
import json
from botocore.config import Config # Boto3のコンフィグ用
import boto3 # AWS SDK
from langchain_aws import ChatBedrock # Bedrock連携用
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 初期設定 ---
# Bedrockモデルのリスト (表示名: モデルID)
AVAILABLE_BEDROCK_MODELS = {
    "Claude 3 Haiku (Anthropic)": "anthropic.claude-3-haiku-20240307-v1:0",
    "Claude 3 Sonnet (Anthropic)": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Claude 3 Opus (Anthropic)": "anthropic.claude-3-opus-20240229-v1:0", # 高性能だが高価・リージョン注意
    "Llama 3 8B Instruct (Meta)": "meta.llama3-8b-instruct-v1:0",
    "Llama 3 70B Instruct (Meta)": "meta.llama3-70b-instruct-v1:0", # 高性能だがリージョン注意
    # "Amazon Titan Text G1 - Express": "amazon.titan-text-express-v1", # 必要に応じて追加
    # "Mistral Large": "mistral.mistral-large-2402-v1:0" # リージョン注意
}
DEFAULT_BEDROCK_MODEL_DISPLAY_NAME = "Claude 3 Haiku (Anthropic)" # デフォルトモデル
AWS_BEDROCK_REGION = "us-east-1"  # Bedrockを利用するAWSリージョン (適宜変更してください)

# --- セッションリセット関数 ---
def reset_chat_and_agent_state():
    st.session_state.messages = []
    st.session_state.search_url_history = []
    keys_to_delete_on_model_change = ["memory", "agent_executor_instance", "cached_llm_instance"]
    for key in keys_to_delete_on_model_change:
        if key in st.session_state:
            del st.session_state[key]
    # @st.cache_resourceでキャッシュされた関数をクリア (引数変更で再実行を促す)

# --- LangChainのコアコンポーネントの初期化 ---

@st.cache_resource # リージョンが変わることも考慮して引数に含める
def get_boto_client(_region_name: str): # キャッシュキーのため引数名にアンダースコア
    """カスタマイズされたBoto3クライアントを取得"""
    boto_config = Config(
        read_timeout=90,    # 読み取りタイムアウトを90秒に設定
        connect_timeout=60, # 接続タイムアウトを60秒に設定
        retries={'max_attempts': 3} # リトライ回数を3回に設定
    )
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=_region_name,
        config=boto_config
    )

@st.cache_resource
def get_llm(model_id: str, region_name: str):
    """選択されたモデルIDとリージョンに基づいてChatBedrock LLMのインスタンスを取得"""
    st.write(f"DEBUG: Initializing Bedrock LLM: {model_id} in {region_name}") # デバッグ用
    try:
        bedrock_boto_client = get_boto_client(region_name)

        # モデルファミリーに応じてmodel_kwargsを調整
        model_kwargs = {"temperature": 0.7}
        if "anthropic.claude" in model_id:
            model_kwargs["max_tokens_to_sample"] = 2048
            # Claude v3はmax_tokensを直接指定できるようになった (LangChainのChatBedrockが対応しているか確認)
            # model_kwargs["max_tokens"] = 2048 # Bedrock APIのClaude3ではこちらが推奨
        elif "meta.llama" in model_id:
            model_kwargs["max_gen_len"] = 2048
            model_kwargs["top_p"] = 0.9
        elif "amazon.titan" in model_id:
            model_kwargs["textGenerationConfig"] = {"maxTokenCount": 2048, "temperature":0.7}
        elif "mistral" in model_id:
            model_kwargs["max_tokens"] = 2048

        llm = ChatBedrock(
            client=bedrock_boto_client, # カスタムBoto3クライアントを使用
            model_id=model_id,
            model_kwargs=model_kwargs,
            # streaming=True, # ストリーミングが必要な場合
        )
        return llm
    except Exception as e:
        st.error(f"LLM (Bedrock: {model_id}) の初期化に失敗しました: {e}")
        st.markdown("""
            **考えられる原因と対処法:**
            - AWS認証情報（環境変数、`~/.aws/credentials`、IAMロールなど）が正しく設定されているか確認してください。
            - 指定したモデルIDが、選択したAWSリージョン (`{region_name}`) で利用可能か確認してください。
            - Bedrockサービスへのアクセス権限がIAMユーザー/ロールに付与されているか確認してください。
        """)
        st.stop()

@st.cache_resource
def get_tools():
    search_tool = DuckDuckGoSearchResults(name="duckduckgo_results_json", max_results=3)
    return [search_tool]

@st.cache_resource
def get_cached_agent_executor(_llm_model_id_for_cache_key: str, _aws_region_for_cache_key: str):
    llm = get_llm(_llm_model_id_for_cache_key, _aws_region_for_cache_key)
    tools = get_tools()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたは親切で役立つアシスタントです。必要に応じてDuckDuckGoを使ってWeb検索を行い、最新情報や特定の情報を調べてユーザーの質問に答えることができます。検索結果を利用した場合は、その情報源について触れるようにしてください。"),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
    return agent_executor

# --- Streamlit アプリケーションのUIとロジック ---

st.set_page_config(page_title="Bedrock LLM選択式 Web検索チャット", layout="wide")
st.title("Bedrock LLM選択式 Web検索チャット 🤖☁️")

# --- サイドバーでのモデル選択と設定 ---
with st.sidebar:
    st.header("設定")

    # セッションステートの初期化 (選択モデルとリージョン)
    if "selected_model_display_name" not in st.session_state:
        st.session_state.selected_model_display_name = DEFAULT_BEDROCK_MODEL_DISPLAY_NAME
    if "selected_bedrock_model_id" not in st.session_state:
        st.session_state.selected_bedrock_model_id = AVAILABLE_BEDROCK_MODELS[DEFAULT_BEDROCK_MODEL_DISPLAY_NAME]
    # リージョンは固定とするが、もし選択式にしたい場合はここに追加
    st.session_state.aws_bedrock_region = AWS_BEDROCK_REGION


    new_selected_model_display_name = st.selectbox(
        "LLMモデルを選択 (AWS Bedrock):",
        options=list(AVAILABLE_BEDROCK_MODELS.keys()),
        index=list(AVAILABLE_BEDROCK_MODELS.keys()).index(st.session_state.selected_model_display_name),
        key="bedrock_model_selector_key"
    )

    if new_selected_model_display_name != st.session_state.selected_model_display_name:
        st.session_state.selected_model_display_name = new_selected_model_display_name
        st.session_state.selected_bedrock_model_id = AVAILABLE_BEDROCK_MODELS[new_selected_model_display_name]
        reset_chat_and_agent_state()
        st.rerun()

st.caption(f"Powered by LangChain & Streamlit, using: {st.session_state.selected_bedrock_model_id} in {st.session_state.aws_bedrock_region}")

# LLMとAgentExecutorの準備
agent_executor = get_cached_agent_executor(st.session_state.selected_bedrock_model_id, st.session_state.aws_bedrock_region)

# セッションステートの初期化 (会話履歴、メモリ、URL履歴)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
if "search_url_history" not in st.session_state:
    st.session_state.search_url_history = []

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力の受付と処理
if user_prompt := st.chat_input("メッセージを入力してください..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner(f"{st.session_state.selected_bedrock_model_id.split('.')[1]} が考えています..."): # モデル名部分を短縮表示
                chat_history_messages = st.session_state.memory.chat_memory.messages
                response_data = agent_executor.invoke(
                    {"input": user_prompt, "chat_history": chat_history_messages}
                )
                ai_response = response_data.get('output', "申し訳ありません、応答を取得できませんでした。")
                intermediate_steps = response_data.get('intermediate_steps', [])

                urls_found_this_turn = []
                for step in intermediate_steps:
                    action, observation = step
                    if action.tool == "duckduckgo_results_json": # ツール名を確認
                        if isinstance(observation, str):
                            try:
                                results_list = json.loads(observation)
                                for res_item in results_list:
                                    if isinstance(res_item, dict) and "link" in res_item:
                                        urls_found_this_turn.append(res_item["link"])
                            except json.JSONDecodeError:
                                st.warning(f"検索結果のJSON解析に失敗: {observation[:100]}...")
                        elif isinstance(observation, list):
                             for res_item in observation:
                                if isinstance(res_item, dict) and "link" in res_item:
                                    urls_found_this_turn.append(res_item["link"])
                for url in urls_found_this_turn:
                    if url not in st.session_state.search_url_history:
                        st.session_state.search_url_history.append(url)

            st.markdown(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            st.session_state.memory.chat_memory.add_user_message(user_prompt)
            st.session_state.memory.chat_memory.add_ai_message(ai_response)

        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- サイドバー (オプションとデバッグ情報) ---
with st.sidebar:
    # ... (モデル選択は上に移動) ...
    st.markdown("---")
    if st.button("会話履歴をリセット", key="reset_chat_button_bedrock"):
        reset_chat_and_agent_state()
        st.rerun()

    st.markdown("---")
    st.subheader("検索されたURL履歴")
    if st.session_state.get("search_url_history"):
        for i, url in enumerate(reversed(st.session_state.search_url_history)):
            try:
                domain = url.split('//')[-1].split('/')[0]
            except:
                domain = "不明なドメイン"
            st.markdown(f"{len(st.session_state.search_url_history) - i}. [{domain}]({url})")
    else:
        st.caption("まだ検索は行われていません。")

    st.markdown("---")
    st.subheader("AWS認証について")
    st.caption(f"このアプリは、設定済みのAWS認証情報とリージョン ({st.session_state.aws_bedrock_region}) を使用してAWS Bedrockに接続します。")
    st.markdown("""
        AWS認証情報が正しく設定されていることを確認してください。一般的な設定方法は以下の通りです:
        - IAMロール (EC2, Lambda, ECSなどで実行する場合)
        - 環境変数 (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`)
        - AWS認証情報ファイル (`~/.aws/credentials` および `~/.aws/config`)
    """)

    st.markdown("---")
    st.subheader("デバッグ情報")
    if st.checkbox("会話メモリを表示 (LangChain)"):
        if "memory" in st.session_state:
            st.write(st.session_state.memory.chat_memory.messages)
        else:
            st.caption("メモリはまだ初期化されていません。")