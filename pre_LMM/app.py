import streamlit as st
import os
import openai
import requests
import tempfile
import torch
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.callbacks.manager import get_openai_callback
from bs4 import BeautifulSoup
import speech_recognition as sr
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
import platform
from pdf2image import convert_from_path



# **ストリームリットのページ設定を最初に配置**
st.set_page_config(page_title="AI Multi-Tool", page_icon="🤖")

# **環境変数の読み込み**
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print(f"✅ APIキーが正しく設定されています: {openai_api_key[:5]}...********")
else:
    print("🚨 APIキーが設定されていません！")

st.sidebar.text(f"API Key: {'✅ 設定済み' if openai_api_key else '❌ 未設定'}")

# **Qdrantの設定**
QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "pdf_collection"

# **モデル選択**
def select_model():
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gpt-3.5-turbo"
        st.session_state.temperature = 0.0

    with st.sidebar:
        if "model_selection" not in st.session_state:
            model = st.radio("Choose a model:", ("GPT-3.5", "GPT-4"), key="model_selection")
            st.session_state.model_name = "gpt-3.5-turbo" if model == "GPT-3.5" else "gpt-4"

        if "temperature_selection" not in st.session_state:
            st.session_state.temperature = st.slider(
                "Temperature:", 0.0, 2.0, 0.0, 0.01, key="temperature_selection"
            )

    return ChatOpenAI(
        temperature=st.session_state.temperature,
        model_name=st.session_state.model_name,
        openai_api_key=openai.api_key
    )


# **Chat Bot**
def chat_bot(llm):
    st.header("ChatGPT 🤗")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [SystemMessage(content="You are a helpful assistant.")]
    if user_input := st.chat_input("Ask me anything!"):
        st.session_state["messages"].append(HumanMessage(content=user_input))
        with st.spinner("ChatGPT is typing..."):
            response = llm(st.session_state["messages"])
        st.session_state["messages"].append(AIMessage(content=response.content))
    for message in st.session_state["messages"]:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

# **ウェブサイト要約**
def website_summarizer(llm):
    st.header("Website Summarizer 🌐")
    url = st.text_input("Enter a website URL:")
    if url:
        with st.spinner("Fetching content..."):
            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                content = soup.body.get_text() if soup.body else "No content available."
            except requests.RequestException as e:
                st.error(f"Failed to fetch the website: {e}")
                return
        prompt = f"Summarize the following text:\n{content[:1000]}"
        with st.spinner("Summarizing..."):
            response = llm([HumanMessage(content=prompt)])
        st.markdown("## Summary")
        st.write(response.content)

# **YouTube要約**
def youtube_summarizer(llm):
    st.header("YouTube Summarizer 🎥")
    url = st.text_input("Enter a YouTube URL:")
    if url:
        with st.spinner("Fetching video transcript..."):
            try:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=True, language='ja')
                docs = loader.load()
            except Exception as e:
                st.error(f"Failed to fetch YouTube transcript: {e}")
                return
        prompt_template = "Summarize the following YouTube video transcript:\n{text}"
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="stuff", verbose=True, prompt=PROMPT)
        response = chain.run(input_documents=docs)
        st.markdown("## Summary")
        st.write(response)

def image_recognition():
    st.header("Image Recognition & Summary 🖼️")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            # ViTの分類モデルをロード
            feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            # クラス予測
            predicted_class_idx = outputs.logits.argmax(-1).item()
            label_map = model.config.id2label  # モデルのラベルマップを取得
            predicted_label = label_map.get(predicted_class_idx, "Unknown")

        st.markdown(f"### Prediction: {predicted_label} (Class {predicted_class_idx})")

        # AIによる説明生成
        llm = select_model()
        prompt = f"This image appears to be a {predicted_label}. Provide a concise and informative description of such an object."
        with st.spinner("Generating description..."):
            response = llm([HumanMessage(content=prompt)])

        st.markdown("## Summary")
        st.write(response.content)


# **音声認識**
def speech_recognition():
    st.header("Speech Recognition 🎙️")
    uploaded_audio = st.file_uploader("Upload an audio file...", type=["wav", "mp3", "m4a"])
    if uploaded_audio:
        st.success("Audio file uploaded successfully!")

# **画像生成**
def image_generation():
    st.header("Image Generation 🎨")
    prompt = st.text_input("Enter a prompt for image generation:")
    if prompt:
        with st.spinner("Generating image..."):
            response = openai.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            image_url = response.data[0].url
            st.image(image_url, caption="Generated Image", use_column_width=True)

# **PDFテキスト取得**
def get_pdf_text():
    uploaded_file = st.file_uploader("Upload your PDF here😇", type='pdf')

    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)

        # **通常のテキスト抽出**
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

        # **OCR処理の必要性を確認**
        if not text.strip():
            st.warning("No selectable text found. Using OCR for text extraction...")

            # **アップロードされたPDFを一時ファイルに保存**
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_file.read())
                temp_pdf_path = temp_pdf.name  # 一時ファイルのパス

            # **WindowsのPopplerパスを設定（必要なら）**
            poppler_path = None
            if platform.system() == "Windows":
                poppler_path = r"C:\Program Files\poppler-23.01.0\bin"  # 適宜変更

            # **PDFを画像に変換**
            images = convert_from_path(temp_pdf_path, dpi=300, poppler_path=poppler_path)

            # **OCRでテキスト抽出**
            text = "\n\n".join([pytesseract.image_to_string(img) for img in images])

            # **一時ファイルを削除**
            os.remove(temp_pdf_path)

        # **OCRでもテキストが取れなかった場合**
        if not text.strip():
            st.error("Could not extract text from the PDF. It might be encrypted or unreadable.")
            return None

        # **テキストをチャンクに分割**
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-davinci-003",
            chunk_size=250,
            chunk_overlap=0,
        )
        return text_splitter.split_text(text)

    return None


# **Qdrantのロード**
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

# **PDFアップロードとベクトルDB作成**
def pdf_upload_and_build_vector_db():
    st.title("PDF Upload & Vector Search 📄")
    pdf_text = get_pdf_text()
    if pdf_text:
        with st.spinner("Loading PDF ..."):
            qdrant = load_qdrant()
            qdrant.add_texts(pdf_text)
        st.success("PDF successfully processed and indexed!")

def ask_my_pdf():
    st.title("Ask My PDF 📜")

    # Qdrant のロード
    qdrant = load_qdrant()
    
    # 既存のデータがあるか確認
    existing_docs = qdrant.similarity_search("test", k=1)
    if not existing_docs:
        st.warning("⚠️ No indexed data found. Please upload and process a PDF first.")
        return

    # 質問を受け付ける（キーを追加）
    query = st.text_input("🔍 Ask a question about the PDF:", key="pdf_query_input")
    
    if query:
        with st.spinner("🤖 Searching for the answer..."):
            try:
                retriever = qdrant.as_retriever()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=select_model(), chain_type="stuff", retriever=retriever
                )
                response = qa_chain.run(query)

                # 回答を表示
                st.markdown("### 📝 Answer:")
                st.write(response if response else "No relevant information found in the PDF.")

            except Exception as e:
                st.error(f"❌ Error during retrieval: {e}")





# **メイン関数**
def main():
    st.sidebar.header("🔧 Select a Tool")

    # Call select_model() only once
    llm = select_model()

    option = st.sidebar.radio("Select a function:", (
        "Chat Bot 🤖", "Website Summarizer 🌐", "YouTube Summarizer 🎥",
        "Image Recognition & Summary 🖼️", "Speech Recognition 🎙️",
        "Image Generation 🎨", "PDF Upload & Vector Search 📄", "Ask My PDF 📜"
    ), key="function_selection")
    
    if option == "Chat Bot 🤖":
        chat_bot(llm)
    elif option == "Website Summarizer 🌐":
        website_summarizer(llm)
    elif option == "YouTube Summarizer 🎥":
        youtube_summarizer(llm)
    elif option == "Image Recognition & Summary 🖼️":
        image_recognition()  # No need to call select_model() inside
    elif option == "Speech Recognition 🎙️":
        speech_recognition()
    elif option == "Image Generation 🎨":
        image_generation()
    elif option == "PDF Upload & Vector Search 📄":
        pdf_upload_and_build_vector_db()
    elif option == "Ask My PDF 📜":
        ask_my_pdf()

if __name__ == '__main__':
    main()
