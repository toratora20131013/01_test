import streamlit as st
import os
import boto3
from langchain_aws import ChatBedrock

# --- 定数設定 ---
BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"  # 利用可能なBedrockモデルを指定
TEMPERATURE_SETTING = 0.0  # 再現性を高めるため少し低めに設定
AWS_REGION = "us-east-1"  # Bedrockが利用可能なリージョンを指定（必要に応じて変更）

# --- AWS認証情報の確認 ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION_ENV = os.getenv("AWS_DEFAULT_REGION") or AWS_REGION

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    st.error("エラー: 環境変数 `AWS_ACCESS_KEY_ID` と `AWS_SECRET_ACCESS_KEY` が設定されていません。")
    st.stop()

# --- Bedrockクライアントの初期化 ---
try:
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=AWS_REGION_ENV,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    
    llm = ChatBedrock(
        client=bedrock_client,
        model_id=BEDROCK_MODEL_ID,
        model_kwargs={"temperature": TEMPERATURE_SETTING}
    )
except Exception as e:
    st.error(f"Bedrockクライアントの初期化中にエラーが発生しました: {e}")
    st.stop()

# --- 要因特性図生成のためのプロンプトテンプレート ---
def create_fishbone_prompt(product_name, failure_mode):
    prompt = f"""
製品名「{product_name}」の故障モード「{failure_mode}」に関する要因特性図（フィッシュボーン図または石川ダイアグラム）をGraphvizのDOT言語で記述してください。

**まず、この故障モード「{failure_mode}」に最も関連性が高いと考えられる主要な要因カテゴリ（大骨）を、製造プロセスの観点（例：素子成形，化成，重合，陰極塗布，樹脂成形など）を参考にしつつ、3～6個程度特定してください。これらのカテゴリ名は、故障モードとの関連性が分かりやすい具体的な名称にしてください。**
**次に、特定した各主要カテゴリに対して、そのカテゴリに属すると考えられる具体的な要因（中骨・小骨）を、それぞれ3～4個程度挙げてください。**

最終的な出力は、DOT言語のコードのみとしてください。説明文や前置きは不要です。

DOT言語の記述例：
```dot
digraph Fishbone {{
    rankdir=LR; // 左から右へ描画
    node [shape=box, style=rounded, fontname="sans-serif"]; // ノードのデフォルトスタイル
    edge [arrowhead=vee]; // エッジのスタイル

    // 故障モード (背骨の終点)
    FailureMode [label="{failure_mode}", shape=ellipse, style="filled,rounded", fillcolor=lightcoral, fontsize=16];

    // 主要カテゴリ (大骨) - LLMが特定したカテゴリ名とラベルを設定
    // 例: MajorCategory1 [label="<LLMが特定したカテゴリ1の表示名>", shape=plaintext, fontsize=14];
    //     MajorCategory2 [label="<LLMが特定したカテゴリ2の表示名>", shape=plaintext, fontsize=14];
    //     // ... 必要に応じてさらにカテゴリを追加 ...

    // 故障モードへの接続 (特定した全ての主要カテゴリをFailureModeに繋ぐ)
    // 例: MajorCategory1 -> FailureMode;
    //     MajorCategory2 -> FailureMode;
    //     // ...

    // 各カテゴリの小要因 (中骨・小骨) - 各カテゴリに繋げる
    // 例: MajorCategory1 に属する要因
    //     Factor1_1 [label="<カテゴリ1の具体的要因1>"];
    //     Factor1_2 [label="<カテゴリ1の具体的要因2>"];
    //     Factor1_1 -> MajorCategory1;
    //     Factor1_2 -> MajorCategory1;
    //
    // 例: MajorCategory2 に属する要因
    //     Factor2_1 [label="<カテゴリ2の具体的要因1>"];
    //     Factor2_1 -> MajorCategory2;

    // 他の特定したカテゴリについても同様に具体的な要因を追加してください。
    // ノードID (例: MajorCategory1, Factor1_1) は英数字でユニークなものを設定してください。
}}

上記を参考に、製品「{product_name}」の故障「{failure_mode}」に対する具体的な要因特性図のDOTコードを生成してください。
必ず FailureMode のラベルは「{failure_mode}」に、そして他の要因は適切に設定してください。
出力は digraph Fishbone {{ ... }} で始まるDOT言語のコードのみとしてください。
"""
    return prompt

# --- 要因特性図修正のためのプロンプトテンプレート ---
def create_modification_prompt(current_dot_code, modification_request):
    prompt = f"""
あなたはGraphvizのDOT言語に詳しいアシスタントです。
以下の既存のDOT言語コードで記述された要因特性図があります。

【既存のDOTコード】
```dot
{current_dot_code}
```

ユーザーからの以下の修正指示に基づいて、このDOTコードを修正してください。

【修正指示】
{modification_request}

修正後のDOT言語のコードのみを出力してください。説明文や前置きは不要です。
digraph Fishbone {{ ... }} で始まるDOT言語のコードのみとしてください。
元の図の構造やスタイルを可能な限り維持し、指示された変更点のみを正確に反映してください。
もし修正指示が曖昧な場合は、最も適切と思われる解釈で修正を試みてください。
"""
    return prompt

#--- DOTコード抽出関数 ---
def extract_dot_code(response_content):
    if "```dot" in response_content:
        return response_content.split("```dot")[1].split("```")[0].strip()
    elif "```" in response_content and "digraph Fishbone" in response_content:
        # ```のみでdotが指定されていない場合
        code_blocks = response_content.split("```")
        for block in code_blocks:
            if "digraph Fishbone" in block:
                return block.strip()
    elif "digraph Fishbone" in response_content:  # マークダウンなしで直接DOTコードが返る場合
        return response_content.strip()
    return None

#--- Streamlit UI ---
st.set_page_config(layout="wide")  # 画面幅を広げる
st.title("📊 要因特性図ジェネレーター (AWS Bedrock活用)")
st.caption(f"🚀 モデル: {BEDROCK_MODEL_ID} | リージョン: {AWS_REGION_ENV}")

#セッション状態の初期化
if 'dot_code' not in st.session_state:
    st.session_state.dot_code = None
if 'product_name_display' not in st.session_state:
    st.session_state.product_name_display = ""
if 'failure_mode_display' not in st.session_state:
    st.session_state.failure_mode_display = ""

col1, col2 = st.columns(2)

with col1:
    st.subheader("入力")
    product_name = st.text_input("製品名を入力してください:", placeholder="例: 電気ケトル", key="product_name_input")
    failure_mode = st.text_input("故障モードを入力してください:", placeholder="例: 電源が入らない", key="failure_mode_input")

if st.button("要因特性図を生成", type="primary", key="generate_button"):
    if product_name and failure_mode:
        with st.spinner("AIが要因特性図を生成中です..."):
            try:
                prompt = create_fishbone_prompt(product_name, failure_mode)
                response = llm.invoke(prompt)

                dot_code_extracted = ""
                if hasattr(response, 'content'):
                    dot_code_extracted = extract_dot_code(response.content)
                    if not dot_code_extracted:
                        st.error("LLMから有効なDOT言語コードが取得できませんでした。応答内容を確認してください。")
                        st.text_area("LLMの応答:", response.content, height=200)
                        st.session_state.dot_code = None  # エラー時はクリア
                    else:
                         st.session_state.dot_code = dot_code_extracted
                else:
                    st.error("LLMからの応答形式が予期したものではありません。")
                    st.text(response)  # デバッグ用に表示
                    st.session_state.dot_code = None  # エラー時はクリア

                if st.session_state.dot_code and not st.session_state.dot_code.startswith("digraph Fishbone"):
                    st.warning("生成されたDOTコードが期待した形式でない可能性があります。そのまま表示を試みます。")
                    st.text_area("生成されたDOTコード (確認用):", st.session_state.dot_code, height=150)

                # 製品名と故障モードを保存
                st.session_state.product_name_display = product_name
                st.session_state.failure_mode_display = failure_mode
                if st.session_state.dot_code:
                     st.success("要因特性図が生成されました。")

            except Exception as e:
                st.error(f"要因特性図の生成中にエラーが発生しました: {e}")
                st.session_state.dot_code = None  # エラー時はクリア
    else:
        st.warning("製品名と故障モードの両方を入力してください。")

with col2:
    st.subheader("生成された要因特性図")
    if st.session_state.dot_code:
        st.markdown(f"製品名: **{st.session_state.get('product_name_display', '')}**")
        st.markdown(f"故障モード: **{st.session_state.get('failure_mode_display', '')}**")
        try:
            st.graphviz_chart(st.session_state.dot_code)
            st.caption("Graphvizで描画された要因特性図")

            # --- 修正機能 ---
            st.subheader("図の修正")
            modification_request = st.text_area(
                "修正指示を入力してください:",
                placeholder="例: 「人」のカテゴリに「作業者の疲労」を追加してください。\n「材料」カテゴリの「供給者A」を「供給者B」に変更してください。",
                key="modification_input",
                height=100
            )
            if st.button("修正を適用", key="apply_modification"):
                if modification_request:
                    with st.spinner("AIが図を修正中です..."):
                        try:
                            prompt = create_modification_prompt(st.session_state.dot_code, modification_request)
                            response = llm.invoke(prompt)

                            modified_dot_code = ""  # 初期化
                            if hasattr(response, 'content'):
                                modified_dot_code = extract_dot_code(response.content)
                                if not modified_dot_code:
                                    st.error("LLMから有効な修正版DOT言語コードが取得できませんでした。応答内容を確認してください。")
                                    st.text_area("LLMの応答（修正時）:", response.content, height=200)
                                else:
                                    st.session_state.dot_code = modified_dot_code  # DOTコードを更新
                                    st.success("図が修正されました。")
                                    st.rerun()  # 再描画して修正を反映
                            else:
                                st.error("LLMからの応答形式が予期したものではありません（修正時）。")
                                if response is not None:
                                    st.text(response)
                                else:
                                    st.text("LLMからの応答がありませんでした。")

                            # 修正後のDOTコードが有効かどうかの簡易チェック
                            if modified_dot_code and not modified_dot_code.startswith("digraph Fishbone"):
                                st.warning("修正後のDOTコードが期待した形式でない可能性があります。")
                                st.text_area("修正後のDOTコード (確認用):", modified_dot_code, height=150)

                        except Exception as e:
                            st.error(f"図の修正中にエラーが発生しました: {e}")
                else:
                    st.warning("修正指示を入力してください。")
        except Exception as e:
            st.error(f"Graphvizでの描画中にエラーが発生しました: {e}")
            st.info("LLMによって生成されたDOTコードに問題があるか、Graphvizが正しく動作していない可能性があります。")
            st.text_area("エラーが発生したDOTコード:", st.session_state.dot_code, height=200, key="graphviz_error_dot_code_display")

    elif st.session_state.dot_code is None and \
         (st.session_state.get('product_name_display') or st.session_state.get('failure_mode_display')):
        st.warning("要因特性図の生成または表示に失敗しました。入力内容やLLMの応答を確認してください。")
    else:
        st.info("左側のフォームに製品名と故障モードを入力し、「要因特性図を生成」ボタンを押してください。")

#--- サイドバー情報 ---
st.sidebar.header("利用上の注意")
st.sidebar.info(
    "このアプリはAWS BedrockのClaude 3.5 Sonnetモデルを使用して要因特性図のDOT言語コードを生成・修正し、Graphvizで描画します。\n"
    "1. 有効な AWS_ACCESS_KEY_ID と AWS_SECRET_ACCESS_KEY が環境変数に設定されている必要があります。\n"
    "2. 使用するAWSアカウントでBedrockサービスが有効になっている必要があります。\n"
    "3. システムにGraphvizがインストールされ、PATHが通っている必要があります。\n"
    "4. LLMが生成・修正する内容は必ずしも正確・完全であるとは限りません。参考情報として活用してください。"
)
st.sidebar.markdown("---")
st.sidebar.header("AWS Bedrock設定")
st.sidebar.info(
    f"モデルID: {BEDROCK_MODEL_ID}\n"
    f"リージョン: {AWS_REGION_ENV}\n"
    f"温度設定: {TEMPERATURE_SETTING}"
)
st.sidebar.markdown("---")
st.sidebar.header("DOTコードについて")
st.sidebar.info(
    "LLMはGraphvizのDOT言語で要因特性図を表現しようとします。"
    "生成・修正されたDOTコードが複雑すぎたり、書式に誤りがあると、正しく表示されない場合があります。"
)
if st.session_state.dot_code:
    with st.sidebar.expander("現在のDOTコードを見る"):
        st.code(st.session_state.dot_code, language="dot")