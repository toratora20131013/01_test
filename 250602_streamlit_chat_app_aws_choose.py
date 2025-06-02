import streamlit as st
import os
import json
from langchain_aws import ChatBedrock # Bedrock用
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from botocore.exceptions import NoCredentialsError, ClientError, ProfileNotFound

# --- 定数 ---
AVAILABLE_MODELS = {
    "Anthropic Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Anthropic Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "Anthropic Claude 3 Opus": "anthropic.claude-3-opus-20240229-v1:0", # Opus追加例
    "Amazon Titan Text G1 - Express": "amazon.titan-text-express-v1",
    "Meta Llama 3 8B Instruct": "meta.llama3-8b-instruct-v1:0",
    "Cohere Command R": "cohere.command-r-v1:0",
}
DEFAULT_MODEL_DISPLAY_NAME = "Anthropic Claude 3 Sonnet"
DEFAULT_MODEL_API_NAME = AVAILABLE_MODELS[DEFAULT_MODEL_DISPLAY_NAME]
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_USE_SEARCH = True
DEFAULT_AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# --- セッションステートの初期化 ---
if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = DEFAULT_MODEL_API_NAME
if "selected_max_tokens" not in st.session_state:
    st.session_state.selected_max_tokens = DEFAULT_MAX_TOKENS
if "selected_temperature" not in st.session_state:
    st.session_state.selected_temperature = DEFAULT_TEMPERATURE
if "selected_aws_region" not in st.session_state:
    st.session_state.selected_aws_region = DEFAULT_AWS_REGION
if "use_duckduckgo" not in st.session_state:
    st.session_state.use_duckduckgo = DEFAULT_USE_SEARCH

if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
if "search_url_history" not in st.session_state:
    st.session_state.search_url_history = []

# --- コールバック関数 ---
def on_settings_change():
    if "agent_executor" in st.session_state:
        del st.session_state.agent_executor
    # LLMやToolsのキャッシュもクリアしたい場合は、get_llm.clear() なども検討するが、
    # 通常は引数が変われば @st.cache_resource が再実行されるので不要。

# --- LangChainのコアコンポーネントの初期化 ---
@st.cache_resource
def get_llm(model_id: str, max_tokens_from_ui: int, temperature_from_ui: float, region_name: str):
    try:
        model_kwargs = {}
        if "anthropic.claude" in model_id:
            model_kwargs["max_tokens_to_sample"] = max_tokens_from_ui
            # Claudeの場合、temperatureはChatBedrockのコンストラクタ引数で設定
        elif "amazon.titan" in model_id:
            model_kwargs["textGenerationConfig"] = {
                "maxTokenCount": max_tokens_from_ui,
                "temperature": temperature_from_ui, # Titanはここで温度設定
                "stopSequences": [],
                "topP": 1.0,
            }
        elif "meta.llama" in model_id:
            model_kwargs["max_gen_len"] = max_tokens_from_ui
            # Llamaの場合、temperatureはChatBedrockのコンストラクタ引数で設定
        elif "cohere.command-r" in model_id:
            model_kwargs["max_tokens"] = max_tokens_from_ui
            # Cohereの場合、temperatureはChatBedrockのコンストラクタ引数で設定
        else:
            st.warning(f"モデル {model_id} のための特定の `max_tokens` キーが不明です。デフォルト設定を試みますが、動作しない可能性があります。")

        llm = ChatBedrock(
            region_name=region_name,
            model_id=model_id,
            model_kwargs=model_kwargs if model_kwargs else None,
            temperature=temperature_from_ui if "amazon.titan" not in model_id else None,
        )
        return llm

    except (NoCredentialsError, ProfileNotFound):
        st.error("AWS認証情報が見つかりません。AWS CLIが設定されているか、環境変数 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN (オプション)) を確認してください。または、有効なAWSプロファイルが設定されているか確認してください。")
        st.stop()
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "AccessDeniedException" or "AccessDenied" in str(e) :
            st.error(f"Bedrockへのアクセスが拒否されました。IAM権限とモデルアクセスが有効になっているか確認してください: {e}")
        elif error_code == "ValidationException" and "modelId" in str(e):
            st.error(f"指定されたモデルID '{model_id}' がリージョン '{region_name}' で無効か、アクセス権がありません。Bedrockコンソールでモデルアクセスを有効にしてください。エラー: {e}")
        else:
            st.error(f"AWS Bedrock APIエラーが発生しました: {e}")
        st.stop()
    except Exception as e:
        st.error(f"LLM ({model_id}) の初期化に失敗しました: {e}")
        st.stop()

@st.cache_resource
def get_tools(enable_search: bool):
    if enable_search:
        search_tool = DuckDuckGoSearchResults(name="duckduckgo_results_json", max_results=3)
        return [search_tool]
    return []

def get_agent_executor(llm, tools):
    if "agent_executor" not in st.session_state:
        system_message_parts = ["あなたは親切で役立つアシスタントです。"]
        if tools:
            system_message_parts.append("必要に応じてDuckDuckGoを使ってWeb検索を行い、最新情報や特定の情報を調べてユーザーの質問に答えることができます。検索結果を利用した場合は、その情報源について触れるようにしてください。")
        else:
            system_message_parts.append("Web検索機能は現在オフになっています。")
        
        system_message_parts.append("ユーザーへの最終的な回答は、いかなる場合も通常の文章で、人間が読みやすい平易なテキスト形式で提供してください。JSON形式やその他のプログラムが解釈するような構造化された形式で最終回答を返さないでください。")
        final_system_message = " ".join(system_message_parts)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", final_system_message),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        st.session_state.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,  # ここを修正 (True または False に固定)
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    return st.session_state.agent_executor

# --- Streamlit アプリケーションのUIとロジック ---
st.set_page_config(page_title="Bedrock Web検索チャット (設定可能)", layout="wide")
st.title("AWS Bedrock Web検索チャット 🌐 (設定可能)")

# --- サイドバー ---
with st.sidebar:
    st.header("LLMと言語モデル設定")

    common_aws_regions = [
        "us-east-1", "us-west-2", "ap-northeast-1", "ap-southeast-1", 
        "eu-central-1", "eu-west-1", "eu-west-2", "ap-south-1", 
        "ca-central-1", "sa-east-1" # さらにリージョン追加
    ]
    try:
        default_region_index = common_aws_regions.index(st.session_state.selected_aws_region)
    except ValueError: # 保存されたリージョンがリストにない場合
        st.session_state.selected_aws_region = DEFAULT_AWS_REGION # デフォルトに戻す
        try:
            default_region_index = common_aws_regions.index(st.session_state.selected_aws_region)
        except ValueError: # デフォルトリージョンもリストにない極端なケース
             default_region_index = 0
             st.session_state.selected_aws_region = common_aws_regions[0]


    selected_region_name = st.selectbox(
        "AWS リージョン:",
        options=common_aws_regions,
        index=default_region_index,
        on_change=on_settings_change,
        key="sb_aws_region"
    )
    st.session_state.selected_aws_region = selected_region_name

    current_model_display_name = DEFAULT_MODEL_DISPLAY_NAME
    for display_name, api_name in AVAILABLE_MODELS.items():
        if api_name == st.session_state.selected_model_name:
            current_model_display_name = display_name
            break
    
    selected_display_name = st.selectbox(
        "LLMモデルを選択:",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(current_model_display_name),
        on_change=on_settings_change,
        key="sb_model_display_name"
    )
    st.session_state.selected_model_name = AVAILABLE_MODELS[selected_display_name]

    st.session_state.selected_max_tokens = st.number_input(
        "最大出力トークン数:",
        min_value=256,
        max_value=400000, # Claude 3 Opus は 200K context, Llama3は8Kなど。かなり幅がある。
        value=st.session_state.selected_max_tokens,
        step=128,
        on_change=on_settings_change,
        key="ni_max_tokens"
    )

    st.session_state.selected_temperature = st.slider(
        "Temperature (出力の多様性):",
        min_value=0.0,
        max_value=1.0, # Titanなどは2.0までいけるが、一般的には1.0が上限
        value=st.session_state.selected_temperature,
        step=0.05,
        on_change=on_settings_change,
        key="slider_temperature"
    )

    st.markdown("---")
    st.header("ツール設定")
    st.session_state.use_duckduckgo = st.toggle(
        "DuckDuckGo Web検索を有効にする",
        value=st.session_state.use_duckduckgo,
        on_change=on_settings_change,
        key="toggle_use_search"
    )

    st.markdown("---")
    st.header("会話操作")
    if st.button("会話履歴をリセット", key="reset_chat"):
        st.session_state.messages = []
        st.session_state.search_url_history = []
        if "agent_executor" in st.session_state:
            del st.session_state.agent_executor
        if "memory" in st.session_state:
            del st.session_state.memory
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
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
    st.caption(
        "このアプリはAWS認証情報を利用します。以下のいずれかの方法で認証情報を設定してください:\n"
        "1. 環境変数 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN (オプション), AWS_DEFAULT_REGION)\n"
        "2. AWS CLIのデフォルトプロファイル (~/.aws/credentials と ~/.aws/config)\n"
        "3. (EC2/ECS/Lambdaなど) IAMロール"
    )
    st.markdown("[AWS認証情報の設定詳細](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)")


    st.markdown("---")
    st.subheader("デバッグ情報")
    if st.checkbox("会話メモリを表示 (LangChain)"):
        if "memory" in st.session_state and hasattr(st.session_state.memory, 'chat_memory'):
            st.write(st.session_state.memory.chat_memory.messages)
        else:
            st.caption("メモリはまだ初期化されていないか、アクセスできません。")

# --- メインコンテンツ ---
current_llm = get_llm(
    st.session_state.selected_model_name,
    st.session_state.selected_max_tokens,
    st.session_state.selected_temperature,
    st.session_state.selected_aws_region
)
current_tools = get_tools(st.session_state.use_duckduckgo)
agent_executor = get_agent_executor(current_llm, current_tools)

caption_model_display_name = [k for k, v in AVAILABLE_MODELS.items() if v == st.session_state.selected_model_name][0]
st.caption(f"LLM: Bedrock ({caption_model_display_name} in {st.session_state.selected_aws_region}), Search: DuckDuckGo ({'ON' if st.session_state.use_duckduckgo else 'OFF'})")

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
                
                # --- Bedrock Claude 3などのコンテンツブロック形式の応答を処理 ---
                ai_response_raw = response_data.get('output', "申し訳ありません、応答を取得できませんでした。")
                final_ai_response = ""

                if isinstance(ai_response_raw, list) and len(ai_response_raw) > 0:
                    text_parts = []
                    all_items_are_valid_text_blocks = True
                    for item in ai_response_raw:
                        if isinstance(item, dict) and item.get('type') == 'text' and 'text' in item:
                            text_parts.append(item['text'])
                        else:
                            all_items_are_valid_text_blocks = False
                            break 
                    if all_items_are_valid_text_blocks and text_parts:
                        final_ai_response = "".join(text_parts)
                    else:
                        # 期待するコンテンツブロック形式ではないリストの場合、文字列として結合
                        st.warning(f"AIからの応答が予期せぬリスト形式でした。文字列として結合します: {ai_response_raw}")
                        final_ai_response = " ".join(map(str, ai_response_raw))
                elif isinstance(ai_response_raw, str):
                    final_ai_response = ai_response_raw
                elif ai_response_raw is None:
                    final_ai_response = "応答がありませんでした。"
                else:
                    st.warning(f"AIからの応答が予期せぬ型でした: {type(ai_response_raw)}")
                    final_ai_response = str(ai_response_raw)

                ai_response = final_ai_response
                # --- 処理ここまで ---

                intermediate_steps = response_data.get('intermediate_steps', [])

                urls_found_this_turn = []
                if st.session_state.use_duckduckgo:
                    for step in intermediate_steps:
                        action, observation = step
                        if hasattr(action, 'tool') and action.tool == "duckduckgo_results_json":
                            if isinstance(observation, str):
                                try:
                                    results_list = json.loads(observation)
                                    for res_item in results_list:
                                        if isinstance(res_item, dict) and "link" in res_item:
                                            urls_found_this_turn.append(res_item["link"])
                                except json.JSONDecodeError:
                                    st.warning(f"検索結果のJSON解析に失敗しました: {observation[:200]}...")
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
            error_message = f"処理中にエラーが発生しました: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})