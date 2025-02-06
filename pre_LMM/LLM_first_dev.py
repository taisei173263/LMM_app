import streamlit as st
import os
import sys
import openai
import requests
import tempfile
import platform
from PIL import Image
from bs4 import BeautifulSoup
from transformers import ViTImageProcessor, ViTForImageClassification
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain.agents import initialize_agent, Tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from pdf2image import convert_from_path
import pytesseract
import speech_recognition as sr
from dotenv import load_dotenv

# Webブラウジング用ツールのインポート（toolsディレクトリ内のモジュール）
from tools.search_ddg import get_search_ddg_tool
from tools.fetch_page import get_fetch_page_tool

# ---------------------------
# ページ設定＆カスタムCSS
# ---------------------------
st.set_page_config(
    page_title="AIマルチツール by Chat Taisei 🤖", 
    page_icon="🤖", 
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #f0f2f6;
    font-family: 'Arial', sans-serif;
}
header {
    text-align: center;
}
.sidebar .sidebar-content {
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

CUSTOM_SYSTEM_PROMPT = """
あなたはオンライン調査を行うアシスタントです。ユーザーのリクエストに基づいて利用可能なツールを使用し、調査結果に基づいて説明してください。
既存の知識だけに頼らず、必ず最新の情報を検索してから回答してください。
特に、プログラミング関連の質問の場合は英語で検索して回答してください。
回答の最後に参照したURLをリストアップするようにしてください。
"""


# ---------------------------
# 環境変数の読み込みとAPIキー設定
# ---------------------------
load_dotenv()
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
openai_api_key = st.sidebar.text_input("OpenAI API キーを入力してください", type="password", value=st.session_state["openai_api_key"])
if openai_api_key:
    st.session_state["openai_api_key"] = openai_api_key
    openai.api_key = openai_api_key
st.sidebar.info("※ OpenAI API キーが必要です。まだの場合は取得してください。")
st.session_state["langsmith_api_key"] = st.sidebar.text_input("LangSmith API キー (任意)", type="password")

# ---------------------------
# PDF / Qdrant 用定数
# ---------------------------
QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "pdf_collection"

# ---------------------------
# モデル選択（共通設定）
# ---------------------------
if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-3.5-turbo"
    st.session_state.temperature = 0.0

# *** Move the radio button creation OUTSIDE the function ***
model = st.sidebar.radio("モデルを選択:", ("GPT-3.5", "GPT-4"), key="model_radio_unique")
st.session_state.model_name = "gpt-3.5-turbo" if model == "GPT-3.5" else "gpt-4"
st.session_state.temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.01, key="temp_slider_unique")


def select_model(): # Function now only returns the LLM
    return ChatOpenAI(
        temperature=st.session_state.temperature,
        model_name=st.session_state.model_name,
        openai_api_key=st.session_state["openai_api_key"]
    )

if "llm" not in st.session_state:  # Use "llm" instead of "selected_model" for clarity
    st.session_state["llm"] = select_model()
llm = st.session_state["llm"]


# ---------------------------
# 各機能の実装
# ---------------------------
def chat_bot(llm):
    st.header("チャットボット 🤖")
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [SystemMessage(content="あなたは有能なアシスタントです。")]
    if user_input := st.chat_input("メッセージを入力してください"):
        st.session_state["chat_messages"].append(HumanMessage(content=user_input))
        with st.spinner("AIが考え中です..."):
            response = llm(st.session_state["chat_messages"])
        st.session_state["chat_messages"].append(AIMessage(content=response.content))
    for message in st.session_state["chat_messages"]:
        if isinstance(message, AIMessage):
            st.chat_message("assistant").markdown(message.content)
        elif isinstance(message, HumanMessage):
            st.chat_message("user").markdown(message.content)

def website_summarizer(llm):
    st.header("ウェブサイト要約 🌐")
    url = st.text_input("要約するウェブサイトのURLを入力してください", key="website_url")
    if url:
        with st.spinner("ウェブサイトからコンテンツを取得中..."):
            try:
                res = requests.get(url)
                res.raise_for_status()
                soup = BeautifulSoup(res.text, "html.parser")
                content = soup.body.get_text() if soup.body else "コンテンツがありません。"
            except Exception as e:
                st.error(f"ウェブサイトの取得に失敗しました: {e}")
                return
        st.subheader("取得したテキスト（一部）")
        st.write(content[:1000])
        prompt = f"以下の文章を要約してください:\n{content[:1000]}"
        with st.spinner("要約中..."):
            response = llm([HumanMessage(content=prompt)])
        st.markdown("### 要約結果")
        st.write(response.content)

def youtube_summarizer(llm):
    st.header("YouTube 要約 🎥")
    url = st.text_input("YouTube のURLを入力してください", key="youtube_url")
    if url:
        with st.spinner("YouTubeのトランスクリプトを取得中..."):
            try:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=True, language='ja')
                docs = loader.load()
            except Exception as e:
                st.error(f"YouTubeの取得に失敗しました: {e}")
                return
        prompt_template = "以下のYouTube動画のトランスクリプトを要約してください:\n{text}"
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="stuff", verbose=True, prompt=PROMPT)
        with st.spinner("要約中..."):
            summary = chain.run(input_documents=docs)
        st.markdown("### 要約結果")
        st.write(summary)

def image_recognition():
    st.header("画像認識＆要約 🖼️")
    uploaded_file = st.file_uploader("画像をアップロードしてください (jpg, jpeg, png)", type=["jpg", "jpeg", "png"], key="img_upload")
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="アップロードされた画像", use_container_width=True)
        with st.spinner("画像を解析中..."):
            image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
            inputs = image_processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            predicted_class_idx = outputs.logits.argmax(-1).item()
            label_map = model.config.id2label
            predicted_label = label_map.get(predicted_class_idx, "不明")
        st.markdown(f"### 予測結果: {predicted_label} (クラス {predicted_class_idx})")
        st.info("※ ここに追加の説明や要約を実装できます。")

def speech_recognition():
    st.header("音声認識 🎙️")
    uploaded_audio = st.file_uploader("音声ファイルをアップロードしてください (wav, mp3, m4a)", type=["wav", "mp3", "m4a"], key="audio_upload")
    if uploaded_audio:
        st.success("音声ファイルがアップロードされました！")
        # ※ 以下はダミー実装です。実際には speech_recognition ライブラリなどを使用して処理してください。
        st.info("※ 音声認識処理は現在未実装です。")

def image_generation():
    st.header("画像生成 🎨")
    prompt = st.text_input("生成する画像のプロンプトを入力してください", key="gen_prompt")
    if prompt:
        with st.spinner("画像生成中..."):
            # ※ ここに実際の画像生成API（例: OpenAI DALL-E）を呼び出すコードを実装してください。
            # 以下はダミーの画像URLです。
            image_url = "https://via.placeholder.com/1024"
        st.image(image_url, caption="生成された画像", use_container_width=True)
        st.success("画像生成が完了しました！")

def get_pdf_text():
    uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type="pdf", key="pdf_upload")
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = "\n\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        if not text.strip():
            st.warning("選択可能なテキストが見つかりませんでした。OCR処理を実行します...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_file.read())
                temp_pdf_path = temp_pdf.name
            poppler_path = r"C:\Program Files\poppler-23.01.0\bin" if platform.system() == "Windows" else None
            images = convert_from_path(temp_pdf_path, dpi=300, poppler_path=poppler_path)
            text = "\n\n".join([pytesseract.image_to_string(img) for img in images])
            os.remove(temp_pdf_path)
        if not text.strip():
            st.error("PDFからテキストを抽出できませんでした。")
            return None
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-davinci-003",
            chunk_size=250,
            chunk_overlap=0,
        )
        return text_splitter.split_text(text)
    return None

def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME, 
        embeddings=OpenAIEmbeddings()
    )

def pdf_upload_and_build_vector_db():
    st.header("PDFアップロード＆ベクトル検索 📄")
    pdf_text = get_pdf_text()
    if pdf_text:
        with st.spinner("PDFを処理中..."):
            qdrant = load_qdrant()
            qdrant.add_texts(pdf_text)
        st.success("PDFの処理とインデックス作成が完了しました！")

def ask_my_pdf():
    st.header("PDFに関する質問 📜")
    qdrant = load_qdrant()
    existing_docs = qdrant.similarity_search("test", k=1)
    if not existing_docs:
        st.warning("⚠️ インデックスされたデータが見つかりませんでした。先にPDFをアップロードしてください。")
        return
    query = st.text_input("PDFに関する質問を入力してください", key="pdf_query_input")
    if query:
        with st.spinner("回答を検索中..."):
            try:
                retriever = qdrant.as_retriever()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=select_model(), chain_type="stuff", retriever=retriever
                )
                response = qa_chain.run(query)
                st.markdown("### 回答")
                st.write(response if response else "該当する情報が見つかりませんでした。")
            except Exception as e:
                st.error(f"検索中にエラーが発生しました: {e}")

def chat_taisei_with_web_browsing():
    st.header("Webブラウジング付き Chat Taisei 🤗")
    if "web_messages" not in st.session_state:
        st.session_state["web_messages"] = [{"role": "assistant", "content": "こんにちは、Chat Taiseiです。ご質問は？"}]
    # ツール関数をToolオブジェクトでラッピング
    search_tool = Tool(
        name="duckduckgo_search",
        func=get_search_ddg_tool,
        description="DuckDuckGoでウェブ検索を行います。入力は検索クエリ文字列です。"
    )
    fetch_page_tool = Tool(
        name="fetch_page",
        func=get_fetch_page_tool,
        description="指定されたウェブページの内容を取得します。入力はURLです。"
    )
    tools = [search_tool, fetch_page_tool]
    if prompt := st.chat_input("ウェブブラウジング機能付きで質問してください"):
        st.session_state["web_messages"].append({"role": "user", "content": prompt})
        with st.spinner("Chat Taiseiが考え中..."):
            llm = select_model()
            search_agent = initialize_agent(
                agent="openai-functions",
                tools=tools,
                llm=llm,
                max_iterations=5,
                agent_kwargs={"system_message": SystemMessage(content=CUSTOM_SYSTEM_PROMPT)}
            )
            response = search_agent.run(st.session_state["web_messages"])
            st.session_state["web_messages"].append({"role": "assistant", "content": response})
            st.markdown(response)

# ---------------------------
# メインレイアウト
# ---------------------------
def main():
    st.sidebar.header("🔧 機能選択")
    option = st.sidebar.radio("利用する機能を選んでください:", [
        "チャットボット 🤖",
        "ウェブサイト要約 🌐",
        "YouTube 要約 🎥",
        "画像認識＆要約 🖼️",
        "音声認識 🎙️",
        "画像生成 🎨",
        "PDFアップロード＆ベクトル検索 📄",
        "PDFに関する質問 📜",
        "Webブラウジング付き Chat Taisei 🤗"
    ], key="feature_select")
    
    # 共通のモデル選択（必要に応じて各機能内で再選択可）
    llm = select_model()
    
    col_main, col_info = st.columns([2, 1])
    with col_main:
        if option == "チャットボット 🤖":
            chat_bot(llm)
        elif option == "ウェブサイト要約 🌐":
            website_summarizer(llm)
        elif option == "YouTube 要約 🎥":
            youtube_summarizer(llm)
        elif option == "画像認識＆要約 🖼️":
            image_recognition()
        elif option == "音声認識 🎙️":
            speech_recognition()
        elif option == "画像生成 🎨":
            image_generation()
        elif option == "PDFアップロード＆ベクトル検索 📄":
            pdf_upload_and_build_vector_db()
        elif option == "PDFに関する質問 📜":
            ask_my_pdf()
        elif option == "Webブラウジング付き Chat Taisei 🤗":
            chat_taisei_with_web_browsing()
    with col_info:
        st.markdown("## 使い方ガイド")
        st.info("""
        この **AIマルチツール** は、以下の機能を提供します：
        - **チャットボット**: AIと対話し、質問や相談に応答します。
        - **ウェブサイト要約**: 指定したウェブサイトの内容を要約します。
        - **YouTube 要約**: YouTube動画のトランスクリプトを取得し、要約します。
        - **画像認識＆要約**: アップロードした画像を解析し、その内容を説明します。
        - **音声認識**: アップロードした音声ファイルを解析します。
        - **画像生成**: 入力したプロンプトに基づき画像を生成します。
        - **PDFアップロード＆ベクトル検索**: PDFのテキスト抽出とインデックス作成を行います。
        - **PDFに関する質問**: インデックスされたPDFから情報を検索し回答します。
        - **Webブラウジング付き Chat Taisei**: オンライン検索機能を備えたチャットボットです。
        """)
        st.markdown("### © Chat Taisei")
        
if __name__ == '__main__':
    main()
