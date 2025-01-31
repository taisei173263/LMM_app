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
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    model_name = "gpt-3.5-turbo" if model == "GPT-3.5" else "gpt-4"
    temperature = st.sidebar.slider("Temperature:", 0.0, 2.0, 0.0, 0.01)
    return ChatOpenAI(temperature=temperature, model_name=model_name, openai_api_key=openai.api_key)

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

# **画像認識**
def image_recognition():
    st.header("Image Recognition 🖼️")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

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
        st.success("Image generation function is currently under development.")

# **PDFアップロード & ベクトル検索**
def pdf_upload():
    st.header("PDF Upload & Vector Search 📄")
    st.success("PDF upload function is currently under development.")

# **PDFに質問**
def ask_my_pdf():
    st.header("Ask My PDF 📜")
    st.success("PDF Q&A function is currently under development.")

# **メイン関数**
def main():
    llm = select_model()
    option = st.sidebar.radio("Select a function:", (
        "Chat Bot", "Website Summarizer", "YouTube Summarizer", "Image Recognition", "Speech Recognition", "Image Generation", "PDF Upload & Vector Search", "Ask My PDF"
    ))
    if option == "Chat Bot":
        chat_bot(llm)
    elif option == "Website Summarizer":
        website_summarizer(llm)
    elif option == "YouTube Summarizer":
        youtube_summarizer(llm)
    elif option == "Image Recognition":
        image_recognition()
    elif option == "Speech Recognition":
        speech_recognition()
    elif option == "Image Generation":
        image_generation()
    elif option == "PDF Upload & Vector Search":
        pdf_upload()
    elif option == "Ask My PDF":
        ask_my_pdf()

if __name__ == '__main__':
    main()
