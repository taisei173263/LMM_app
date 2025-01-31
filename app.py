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

# 環境変数の設定
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit ページ設定
def init_page():
    st.set_page_config(
        page_title="AI Multi-Tool",
        page_icon="🤖"
    )
    st.sidebar.title("Options")
    st.sidebar.subheader("Choose a feature:")

# モデル選択
def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    model_name = "gpt-3.5-turbo" if model == "GPT-3.5" else "gpt-4"
    temperature = st.sidebar.slider("Temperature:", 0.0, 2.0, 0.0, 0.01)

    return ChatOpenAI(
        temperature=temperature,
        model_name=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    ) 

# チャットボット機能
def chat_bot(llm):
    st.header("ChatGPT 🤗")

    if st.sidebar.button("Clear Conversation", key="clear"):
        st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]
        st.session_state.costs = []

    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]
        st.session_state.costs = []

    if user_input := st.chat_input("Ask me anything!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("ChatGPT is typing ..."):
            with get_openai_callback() as cb:
                response = llm(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response.content))
            st.session_state.costs.append(cb.total_cost)

    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)

    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(st.session_state.costs):.5f}**")
    for cost in st.session_state.costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

# ウェブサイト要約機能
def website_summarizer(llm):
    st.header("Website Summarizer 🤗")
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

# YouTube動画要約機能
def youtube_summarizer(llm):
    st.header("YouTube Summarizer 🤗")
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

# 画像認識機能
def image_recognition():
    st.header("Image Recognition 🖼️")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing Image..."):
            model_name = "google/vit-base-patch16-224"
            feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
            model = ViTForImageClassification.from_pretrained(model_name)

            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            predicted_label = outputs.logits.argmax(-1).item()

            st.success(f"Prediction: {model.config.id2label[predicted_label]}")

# 音声認識機能
def speech_recognition():
    st.header("Speech Recognition 🎙️")
    uploaded_audio = st.file_uploader("Upload an audio file...", type=["wav", "mp3", "m4a"])

    if uploaded_audio is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_audio.read())
            temp_audio_path = temp_audio.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio = recognizer.record(source)

        with st.spinner("Processing audio..."):
            try:
                transcription = openai.Audio.transcribe(
                    "whisper-1",
                    temp_audio_path
                )
                st.success("Transcription:")
                st.write(transcription["text"])
            except Exception as e:
                st.error(f"Error: {e}")

# 画像生成機能
def image_generation():
    st.header("Image Generation 🎨")
    prompt = st.text_input("Enter a prompt for image generation:")

    if prompt:
        with st.spinner("Generating image..."):
            try:
                response = openai.Image.create(
                    prompt=prompt,
                    n=1,
                    size="1024x1024"
                )
                image_url = response["data"][0]["url"]
                st.image(image_url, caption="Generated Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

# メイン関数
def main():
    init_page()
    llm = select_model()
    
    option = st.sidebar.radio("Select a function:", (
        "Chat Bot", 
        "Website Summarizer", 
        "YouTube Summarizer", 
        "Image Recognition", 
        "Speech Recognition", 
        "Image Generation"
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

if __name__ == '__main__':
    main()
