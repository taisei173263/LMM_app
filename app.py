import streamlit as st
from streamlit_chat import message
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os

# .env ファイルの読み込み
load_dotenv()

# 環境変数の取得（Google Gemini APIキー）
google_api_key = os.getenv("GOOGLE_API_KEY")

# APIキーの確認
if google_api_key is None:
    raise ValueError("ERROR: GOOGLE_API_KEY が取得できませんでした。`.env` ファイルを確認してください。")


def init_page():
    """Streamlit ページの設定"""
    st.set_page_config(
        page_title="Website Summarizer",
        page_icon="🤗"
    )
    st.header("Website Summarizer 🤗")
    st.sidebar.title("Options")


def init_messages():
    """チャット履歴の初期化"""
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]


def select_model():
    """Google Gemini モデルを選択"""
    model = st.sidebar.radio("Choose a model:", ("Gemini-Pro",))
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0,
        google_api_key=google_api_key  # APIキーを明示的に指定
    )


def get_url_input():
    """ユーザーが入力する URL を取得"""
    url = st.text_input("URL: ", key="input")
    return url


def validate_url(url):
    """入力された URL が正しいか検証"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_content(url):
    """Webページの内容を取得"""
    try:
        with st.spinner("Fetching Content ..."):
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # HTTPエラーをキャッチ
            soup = BeautifulSoup(response.text, 'html.parser')
            # fetch text from main (change the below code to filter page)
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            elif soup.body:
                return soup.body.get_text()
            else:
                return None
    except requests.exceptions.RequestException as e:
        st.write(f'Error fetching the page: {e}')
        return None


def build_prompt(content, n_chars=300):
    """要約用のプロンプトを作成"""
    return f"""Here is the content of a web page. Please provide a concise summary of around {n_chars} characters.

========

{content[:1000]}

"""


def get_answer(llm, prompt):
    """Gemini API を使用して要約を取得"""
    response = llm.invoke(prompt)  # **修正: `text` を渡す**
    return response.content


def main():
    """Streamlit アプリのメイン処理"""
    init_page()

    llm = select_model()
    init_messages()

    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write('Please input a valid URL')
            answer = None
        else:
            content = get_content(url)
            if content:
                prompt = build_prompt(content)
                st.session_state.messages.append(HumanMessage(content=prompt))
                with st.spinner("Google AI is typing ..."):
                    answer = get_answer(llm, prompt)
            else:
                answer = None

    if answer:
        with response_container:
            st.markdown("## Summary")
            st.write(answer)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(content)


if __name__ == '__main__':
    main()
