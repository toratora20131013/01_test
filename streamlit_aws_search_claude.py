import streamlit as st
import boto3
import json
from typing import Dict, Any
import re

# ページ設定
st.set_page_config(
    page_title="Bedrock ChatBot",
    page_icon="🤖",
    layout="wide"
)

# CSS スタイル
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #666;
    }
    
    .stCodeBlock {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.25rem;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# AWS Bedrock クライアントの初期化
@st.cache_resource
def init_bedrock_client():
    """AWS Bedrockクライアントを初期化"""
    try:
        return boto3.client(
            'bedrock-runtime',
            region_name=st.secrets.get("AWS_REGION", "us-east-1")
        )
    except Exception as e:
        st.error(f"AWS Bedrock クライアントの初期化に失敗しました: {str(e)}")
        return None

# 利用可能なモデル一覧
AVAILABLE_MODELS = {
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "Claude 3.5 Haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "Claude 3 Opus": "anthropic.claude-3-opus-20240229-v1:0",
    "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "Titan Text G1 - Express": "amazon.titan-text-express-v1",
    "Titan Text G1 - Lite": "amazon.titan-text-lite-v1",
}

def format_message_content(content: str) -> str:
    """メッセージの内容をフォーマット"""
    # Markdownのコードブロックを検出
    code_blocks = re.findall(r'```(\w*)\n(.*?)\n```', content, re.DOTALL)
    
    if code_blocks:
        formatted_content = content
        for lang, code in code_blocks:
            # コードブロックをStreamlitのst.codeに置き換え
            original_block = f"```{lang}\n{code}\n```"
            formatted_content = formatted_content.replace(original_block, f"__CODE_BLOCK_{lang}__")
        return formatted_content, code_blocks
    
    return content, []

def invoke_bedrock_model(client, model_id: str, messages: list, temperature: float, max_tokens: int) -> str:
    """Bedrockモデルを呼び出し"""
    try:
        if "anthropic.claude" in model_id:
            # Claudeモデル用のリクエスト形式
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
        elif "amazon.titan" in model_id:
            # Titanモデル用のリクエスト形式
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9
                }
            }
        else:
            raise ValueError(f"サポートされていないモデル: {model_id}")
        
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        
        response_body = json.loads(response['body'].read())
        
        if "anthropic.claude" in model_id:
            return response_body['content'][0]['text']
        elif "amazon.titan" in model_id:
            return response_body['results'][0]['outputText']
            
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

def display_chat_message(role: str, content: str):
    """チャットメッセージを表示"""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "🧑‍💻" if role == "user" else "🤖"
    
    with st.container():
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <div class="message-header">{icon} {role.title()}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # メッセージ内容の処理
        formatted_content, code_blocks = format_message_content(content)
        
        if code_blocks:
            # コードブロックがある場合
            parts = formatted_content.split("__CODE_BLOCK_")
            st.markdown(parts[0])
            
            for i, (lang, code) in enumerate(code_blocks):
                if i + 1 < len(parts):
                    # コードブロックを表示
                    st.code(code, language=lang if lang else None)
                    # コードブロック後のテキストを表示
                    remaining_text = parts[i + 1].replace(f"{lang}__", "")
                    if remaining_text.strip():
                        st.markdown(remaining_text)
        else:
            # 通常のマークダウン表示
            st.markdown(content)

def main():
    st.title("🤖 Bedrock ChatBot")
    st.markdown("AWS Bedrockを使用したチャットボットアプリです")
    
    # Bedrockクライアントの初期化
    bedrock_client = init_bedrock_client()
    if not bedrock_client:
        st.stop()
    
    # サイドバーでの設定
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # モデル選択
        selected_model_name = st.selectbox(
            "LLMモデルを選択",
            options=list(AVAILABLE_MODELS.keys()),
            index=0
        )
        selected_model_id = AVAILABLE_MODELS[selected_model_name]
        
        # Temperature設定
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="値が高いほど創造的な回答になります"
        )
        
        # 最大トークン数設定
        max_tokens = st.slider(
            "最大出力トークン数",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100,
            help="生成される回答の最大長を制御します"
        )
        
        # 設定情報の表示
        st.info(f"""
        **現在の設定:**
        - モデル: {selected_model_name}
        - Temperature: {temperature}
        - 最大トークン: {max_tokens}
        """)
        
        # チャット履歴をクリア
        if st.button("🗑️ チャット履歴をクリア", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    # セッション状態の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # チャット履歴の表示
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
    
    # ユーザー入力
    if prompt := st.chat_input("メッセージを入力してください..."):
        # ユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # アシスタントの応答を生成
        with st.spinner("回答を生成中..."):
            # Bedrock用のメッセージ形式に変換
            bedrock_messages = []
            for msg in st.session_state.messages:
                bedrock_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # モデルを呼び出し
            response = invoke_bedrock_model(
                bedrock_client,
                selected_model_id,
                bedrock_messages,
                temperature,
                max_tokens
            )
        
        # アシスタントの応答を追加
        st.session_state.messages.append({"role": "assistant", "content": response})
        display_chat_message("assistant", response)
        
        st.rerun()
    
    # フッター
    st.markdown("---")
    st.markdown(
        "💡 **Tips:** コードやMarkdownの内容は自動的にフォーマットされ、コピーしやすく表示されます。"
    )

if __name__ == "__main__":
    main()