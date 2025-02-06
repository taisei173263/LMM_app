import streamlit as st

# `set_page_config()` をスクリプトの最初の Streamlit コマンドとして設定
st.set_page_config(page_title="Chat Taisei", page_icon="🤗")

import os
import sys
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.schema import SystemMessage
from dotenv import load_dotenv

# 必要に応じて、`tools` ディレクトリを Python の検索パスに追加するコードをここに記述

# `tools` 内のモジュールをインポート
from tools.search_ddg import get_search_ddg_tool
from tools.fetch_page import get_fetch_page_tool

# 環境変数の読み込み
load_dotenv()
st.session_state["openai_api_key"] = os.getenv("OPENAI_API_KEY")

# カスタムプロンプトの設定
CUSTOM_SYSTEM_PROMPT = """
You are an assistant that conducts online research based on user requests. Using available tools, please explain the researched information.
Please don't answer based solely on what you already know. Always perform a search before providing a response.

In special cases, such as when the user specifies a page to read, there's no need to search.
Please read the provided page and answer the user's question accordingly.

If you find that there's not much information just by looking at the search results page, consider these two options and try them out:
- Try clicking on the links of the search results to access and read the content of each page.
- Change your search query and perform a new search.

Users are extremely busy and not as free as you are.
Therefore, to save the user's effort, please provide direct answers.

GOOD ANSWER EXAMPLE
- This is sample code:  -- sample code here --
- The answer to your question is -- answer here --

Please make sure to list the URLs of the pages you referenced at the end of your answer. (This will allow users to verify your response.)

Please make sure to answer in the language used by the user. If the user asks in Japanese, please answer in Japanese. If the user asks in Spanish, please answer in Spanish.
But, you can go ahead and search in English, especially for programming-related questions. PLEASE MAKE SURE TO ALWAYS SEARCH IN ENGLISH FOR THOSE.
"""

# Streamlit のページ設定
def init_page():
    st.title("Chat Taisei 🤗")
    st.sidebar.title("Options")
    st.session_state["openai_api_key"] = st.sidebar.text_input("OpenAI API Key", type="password")
    st.session_state["langsmith_api_key"] = st.sidebar.text_input("LangSmith API Key (optional)", type="password")

# メッセージの初期化
def init_messages():
    if "messages" not in st.session_state or st.sidebar.button("Clear Conversation"):
        st.session_state.messages = [{"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}]

# モデルの選択
def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-4", "GPT-3.5-16k", "GPT-3.5 (not recommended)"))
    model_mapping = {
        "GPT-4": "gpt-4",
        "GPT-3.5-16k": "gpt-3.5-turbo-16k",
        "GPT-3.5 (not recommended)": "gpt-3.5-turbo"
    }
    
    return ChatOpenAI(
        temperature=0,
        openai_api_key=st.session_state["openai_api_key"],
        model_name=model_mapping.get(model, "gpt-4"),
        streaming=True
    )

# メイン関数
def main():
    init_page()
    init_messages()
    
    # ツール関数を Tool オブジェクトでラッピング（関数参照を渡す）
    search_tool = Tool(
        name="duckduckgo_search",  # ツール名はアルファベット、数字、アンダースコア、ハイフンのみ使用
        func=get_search_ddg_tool,   # 関数参照を渡す（実行時に必要な引数が与えられます）
        description="Search the web using DuckDuckGo. Input should be a search query string."
    )
    
    fetch_page_tool = Tool(
        name="fetch_page",  # 同様に名前を修正
        func=get_fetch_page_tool,   # 関数参照を渡す
        description="Fetch the content of a given web page. Input should be a URL."
    )
    
    tools = [search_tool, fetch_page_tool]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if not st.session_state["openai_api_key"]:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    
    llm = select_model()
    
    if st.session_state["langsmith_api_key"]:
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
        os.environ['LANGCHAIN_API_KEY'] = st.session_state["langsmith_api_key"]
    
    if prompt := st.chat_input(placeholder="Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        search_agent = initialize_agent(
            agent='openai-functions',
            tools=tools,
            llm=llm,
            max_iterations=5,
            agent_kwargs={"system_message": SystemMessage(content=CUSTOM_SYSTEM_PROMPT)}
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

if __name__ == '__main__':
    main()
