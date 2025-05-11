import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from streamlit_agraph import agraph, Node, Edge, Config

# --- 定数設定 ---
GEMINI_MODEL_NAME = "gemini-2.0-flash"  # 例として最新のモデルを指定
TEMPERATURE_SETTING = 0.3 # 再現性を高めるため少し低めに設定

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
        #convert_system_message_to_human=True
    )
except Exception as e:
    st.error(f"LLMの初期化中にエラーが発生しました: {e}")
    st.stop()

# --- 要因特性図生成のためのプロンプトテンプレート ---
def create_fishbone_prompt(product_name, failure_mode):
    prompt = f"""
製品名「{product_name}」の故障モード「{failure_mode}」に関する要因特性図（フィッシュボーン図または石川ダイアグラム）をGraphvizのDOT言語で記述してください。

**まず、この故障モード「{failure_mode}」に最も関連性が高いと考えられる主要な要因カテゴリ（大骨）を、一般的な要因分析の観点（例：人、設備、材料、手順、環境、測定など）を参考にしつつ、3～6個程度特定してください。これらのカテゴリ名は、故障モードとの関連性が分かりやすい具体的な名称にしてください。**
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

#--- Streamlit UI ---
st.set_page_config(layout="wide") # 画面幅を広げる
st.title("📊 要因特性図ジェネレーター (LLM活用)")
st.caption(f"🚀 モデル: {GEMINI_MODEL_NAME}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("入力")
    product_name = st.text_input("製品名を入力してください:", placeholder="例: 電気ケトル")
    failure_mode = st.text_input("故障モードを入力してください:", placeholder="例: 電源が入らない")

if st.button("要因特性図を生成", type="primary"):
    if product_name and failure_mode:
        with st.spinner("AIが要因特性図を生成中です..."):
            try:
                prompt = create_fishbone_prompt(product_name, failure_mode)
                response = llm.invoke(prompt)

                dot_code = ""
                if hasattr(response, 'content'):
                    # response.content からDOTコードを抽出
                    # LLMが```dot ... ``` のようにマークダウンで囲んで返す場合があるため抽出
                    if "```dot" in response.content:
                        dot_code = response.content.split("```dot")[1].split("```")[0].strip()
                    elif "digraph Fishbone" in response.content: # マークダウンなしで直接DOTコードが返る場合
                        dot_code = response.content.strip()
                    else:
                        st.error("LLMから有効なDOT言語コードが取得できませんでした。応答内容を確認してください。")
                        st.text_area("LLMの応答:", response.content, height=200)
                        st.stop()
                else:
                    st.error("LLMからの応答形式が予期したものではありません。")
                    st.text(response) # デバッグ用に表示
                    st.stop()

                if not dot_code.startswith("digraph Fishbone"):
                    st.warning("生成されたDOTコードが期待した形式でない可能性があります。そのまま表示を試みます。")
                    st.text_area("生成されたDOTコード (確認用):", dot_code, height=150)


                # 生成されたDOTコードをセッション状態に保存
                st.session_state.dot_code = dot_code
                st.session_state.product_name_display = product_name
                st.session_state.failure_mode_display = failure_mode


            except Exception as e:
                st.error(f"要因特性図の生成中にエラーが発生しました: {e}")
                st.session_state.dot_code = None # エラー時はクリア
    else:
        st.warning("製品名と故障モードの両方を入力してください。")
        
with col2:
    st.subheader("生成された要因特性図")
    if 'dot_code' in st.session_state and st.session_state.dot_code:
        st.markdown(f"製品名: {st.session_state.get('product_name_display', '')}")
        st.markdown(f"故障モード: {st.session_state.get('failure_mode_display', '')}")
        try:
            # Streamlit標準のgraphviz_chartを使用
            st.graphviz_chart(st.session_state.dot_code)
            st.caption("Graphvizで描画された要因特性図")
        except Exception as e:
            st.error(f"Graphvizでの描画中にエラーが発生しました: {e}")
            st.info("LLMによって生成されたDOTコードに問題があるか、Graphvizが正しく動作していない可能性があります。")
            # 以前 StreamlitDuplicateElementId エラーが出た箇所に key を追加
            st.text_area("エラーが発生したDOTコード:", st.session_state.dot_code, height=200, key="graphviz_error_dot_code_display")



        #     # agraphの設定 (見た目の調整)
        #     config = Config(
        #         width="100%", # 幅をコンテナに合わせる
        #         height=600,     # 高さを固定
        #         directed=True,
        #         physics=True, # ノードが重ならないように物理エンジンを有効化 (動作が重くなる場合あり)
        #         hierarchical=False, # rankdir=LR を活かすためにFalseが良い場合も
        #         layout="dot",
        #         # その他のオプションは streamlit_agraph のドキュメント参照
        #     )
        #     # agraphはNodeとEdgeのリストを期待しない。DOT文字列を直接渡せる。
        #     nodes = []
        #     edges = []
        #     agraph(nodes=nodes, edges=edges, dot=st.session_state.dot_code, config=config)
        # except Exception as e:
        #     st.error(f"要因特性図の描画中にエラーが発生しました: {e}")
        #     st.info("LLMによって生成されたDOTコードに問題があるか、Graphvizが正しく動作していない可能性があります。")
        #     st.text_area("エラーが発生したDOTコード:", st.session_state.dot_code, height=200, key="error_dot_code_display")

    elif 'dot_code' in st.session_state and st.session_state.dot_code is None:
        st.info("エラーが発生したため、図を表示できません。")
    else:
        st.info("左側のフォームに製品名と故障モードを入力し、「要因特性図を生成」ボタンを押してください。")

    
#--- 注意事項 ---
st.sidebar.header("利用上の注意")
st.sidebar.info(
"このアプリはGoogleのGeminiモデルを使用して要因特性図のDOT言語コードを生成し、Graphvizで描画します。\n"
"1. 有効な GEMINI_API_KEY が環境変数に設定されている必要があります。\n"
"2. システムにGraphvizがインストールされ、PATHが通っている必要があります。\n"
"3. LLMが生成する内容は必ずしも正確・完全であるとは限りません。参考情報として活用してください。"
)
st.sidebar.markdown("---")
st.sidebar.header("DOTコードについて")
st.sidebar.info(
"LLMはGraphvizのDOT言語で要因特性図を表現しようとします。"
"生成されたDOTコードが複雑すぎたり、書式に誤りがあると、正しく表示されない場合があります。"
)
if 'dot_code' in st.session_state and st.session_state.dot_code:
    with st.sidebar.expander("生成されたDOTコードを見る"):
        st.code(st.session_state.dot_code, language="dot")