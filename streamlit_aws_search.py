import streamlit as st
import os
import re
# langchain_google_genaiは不要になったため削除
from langchain_aws import ChatBedrock # GoogleからAWS Bedrock用に変更
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- アプリケーション設定 (Bedrock用に変更) ---
DEFAULT_MODEL_NAME = "anthropic.claude-3-sonnet-20240229-v1:0"
AVAILABLE_MODELS = [
    "anthropic.claude-3-sonnet-20240229-v1:0", # Claude 3 Sonnet (バランス)
    "anthropic.claude-3-haiku-20240307-v1:0",   # Claude 3 Haiku (高速)
    "anthropic.claude-3-opus-20240229-v1:0",    # Claude 3 Opus (高性能・リージョンによっては利用不可)
    "cohere.command-r-plus-v1:0",              # Cohere Command R+
]
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_USE_SEARCH = True

# --- LangChainのコアコンポーネントの初期化 ---

@st.cache_resource
def get_llm(model_name: str, temperature: float, max_tokens: int):
    """
    選択されたモデル、temperature、max_tokensでLLM (Bedrock) を初期化します。
    """
    st.write(f"LLMを初期化中: model={model_name}, temp={temperature}, tokens={max_tokens}") # デバッグ用
    try:
        # AWS認証情報はboto3が環境変数から自動で読み込むため、キーの直接指定は不要
        
        # AWSリージョンを環境変数から取得、なければデフォルト値を使用
        aws_region = os.environ.get("AWS_REGION", "us-east-1")

        llm = ChatBedrock(
            # credentials_profile_name="your-profile-name", # 特定のプロファイルを使いたい場合はコメント解除
            region_name=aws_region,
            model_id=model_name,
            model_kwargs={
                "temperature": temperature,
                # Claude 3, Cohere Command R+ ともに 'max_tokens' というパラメータ名
                "max_tokens": max_tokens,
                # Claude 3の場合、システムプロンプトはこちらに渡すのがより確実
                # "system": "あなたは親切で役立つアシスタントです。"
            },
        )
        return llm
    except Exception as e:
        # 認証情報がない場合のエラーメッセージを分かりやすくする
        if "NoCredentialsError" in str(e):
             st.error("AWSの認証情報が見つかりませんでした。環境変数 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION) を設定してください。")
        else:
            st.error(f"LLM ({model_name}) の初期化に失敗しました: {e}")
        st.stop()

@st.cache_resource
def get_tools(use_search: bool):
    """
    利用可能なツールのリストを取得します。(この関数は変更なし)
    use_searchがTrueの場合のみDuckDuckGo検索ツールを含めます。
    """
    st.write(f"ツールを初期化中: use_search={use_search}") # デバッグ用
    if use_search:
        search_tool = DuckDuckGoSearchResults(name="duckduckgo_search_results", max_results=5)
        search_tool.description = "現在のイベントや、LLMの知識が及ばないトピックについて答えるために、インターネットを検索します。"
        return [search_tool]
    return []

@st.cache_resource
def get_agent_executor(_llm, _tools): # 引数名にアンダースコアを付けてキャッシュがオブジェクトIDではなく内容に依存するように促す
    """
    LLMとツールに基づいてAgentExecutorを初期化します。(この関数は変更なし)
    """
    st.write("AgentExecutorを初期化中...") # デバッグ用
    
    system_message_content = "あなたは親切で役立つアシスタントです。"
    system_message_content += " コードを生成する際は、必ず言語名を指定したMarkdownのコードブロック（例: ```python ... ```）で囲ってください。"
    if _tools: # ツールが提供されている場合のみ検索に関する言及を追加
        system_message_content += " 必要に応じて利用可能なツールを使って情報を調べてユーザーの質問に答えることができます。検索結果を利用した場合は、その情報源（URLなど）について触れるようにしてください。"
    else:
        system_message_content += " ユーザーの質問に、あなたの知識の範囲で直接答えてください。"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message_content),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(_llm, _tools, prompt)
    
    agent_executor_instance = AgentExecutor(
        agent=agent,
        tools=_tools,
        verbose=True,
        handle_parsing_errors=True, 
        return_intermediate_steps=True
    )
    return agent_executor_instance

# --- Streamlit アプリケーションのUIとロジック ---

st.set_page_config(page_title="Bedrock カスタムチャット", layout="wide")
st.title("Bedrock カスタムチャット 🤖💬 (AWS)")

# --- サイドバーでの設定 ---
with st.sidebar:
    st.header("⚙️ 設定")

    st.subheader("LLM設定")
    selected_model = st.selectbox(
        "モデルを選択",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL_NAME)
    )
    current_temperature = st.slider(
        "Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05,
        help="値が低いほど決定的で一貫性のある応答になり、高いほど多様で創造的な応答になります。"
    )
    current_max_tokens = st.number_input(
        "Max Output Tokens", min_value=512, max_value=8192, value=DEFAULT_MAX_TOKENS, step=256,
        help="1回の応答で生成されるトークンの最大数です。"
    )

    st.subheader("ツール設定")
    use_duckduckgo_search = st.toggle(
        "DuckDuckGo検索を有効にする", value=DEFAULT_USE_SEARCH,
        help="有効にすると、AIは必要に応じてWeb検索を行い、最新情報に基づいて回答します。"
    )
    
    st.markdown("---")
    if st.button("会話履歴をリセット", key="reset_chat_button"):
        st.session_state.messages = []
        st.session_state.search_url_history = []
        if "memory" in st.session_state:
            del st.session_state.memory
        get_llm.clear()
        get_tools.clear()
        get_agent_executor.clear()
        st.success("会話履歴とキャッシュをリセットしました。")
        st.rerun()

# --- LLMとツールの初期化 ---
llm = get_llm(selected_model, current_temperature, current_max_tokens)
tools = get_tools(use_duckduckgo_search)
agent_executor = get_agent_executor(llm, tools)

st.caption(f"Model: {selected_model} | Temp: {current_temperature} | MaxTokens: {current_max_tokens} | Search: {'ON' if use_duckduckgo_search else 'OFF'}")


# --- チャット履歴とメモリの管理 ---
# (このセクションは変更なし)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
if "search_url_history" not in st.session_state:
    st.session_state.search_url_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("メッセージを入力してください..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("AIが考えています..."):
                chat_history_messages = st.session_state.memory.chat_memory.messages
                
                response_data = agent_executor.invoke(
                    {
                        "input": user_prompt,
                        "chat_history": chat_history_messages
                    }
                )
                ai_response = response_data.get('output', "申し訳ありません、応答を取得できませんでした。")
                intermediate_steps = response_data.get('intermediate_steps', [])

                # --- URL抽出ロジック (変更なし、ただし正規表現を少し改善) ---
                urls_found_this_turn = []
                if intermediate_steps:
                    for step in intermediate_steps:
                        action, observation = step
                        if action.tool == "duckduckgo_search_results": 
                            if isinstance(observation, str):
                                # 'link:' だけでなく 'url:' などにも対応できるよう柔軟に
                                found_links = re.findall(r"\[source\]:\s*(https?://[^\s)]+)", observation, re.IGNORECASE)
                                if not found_links:
                                     # DuckDuckGoSearchResultsのデフォルト出力形式からURLを抽出
                                     found_links = re.findall(r"link:\s*(https?://[^\s]+)", observation, re.IGNORECASE)
                                
                                urls_found_this_turn.extend(found_links)
                                
                for url in urls_found_this_turn:
                    if url not in st.session_state.search_url_history:
                        st.session_state.search_url_history.append(url)

            st.markdown(ai_response)
            
            with st.expander("Markdownソースをコピー"):
                st.code(ai_response, language='markdown')
                
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            st.session_state.memory.save_context({"input": user_prompt}, {"output": ai_response})

        except Exception as e:
            error_message = f"処理中にエラーが発生しました: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})


# --- サイドバー下部 (URL履歴 & APIキー説明をBedrock用に変更) ---
with st.sidebar:
    st.markdown("---")
    st.subheader("🌐 参照されたURL履歴")
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
    st.subheader("認証について")
    st.caption("このアプリは、実行環境に設定されたAWS認証情報（環境変数など）を使用してAWS Bedrockに接続します。")
    st.markdown("""
    **設定方法:**
    1. AWS IAMでアクセスキーを作成します。
    2. 以下の環境変数を設定します。
       - `AWS_ACCESS_KEY_ID`
       - `AWS_SECRET_ACCESS_KEY`
       - `AWS_REGION` (例: `us-east-1`)
    
    [AWSドキュメント: 認証情報の設定](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)
    """)

    st.markdown("---")
    st.subheader("デバッグ情報")
    if st.checkbox("会話メモリを表示 (LangChain)"):
        if "memory" in st.session_state and hasattr(st.session_state.memory, "chat_memory"):
            st.write(st.session_state.memory.chat_memory.messages)
        else:
            st.caption("メモリはまだ初期化されていません。")