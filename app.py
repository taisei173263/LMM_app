import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# .env ファイルの読み込み
load_dotenv()

# 環境変数の取得
google_api_key = os.getenv("GOOGLE_API_KEY")

# APIキーが取得できていない場合のエラーハンドリング
if google_api_key is None:
    raise ValueError("ERROR: GOOGLE_API_KEY が取得できませんでした。`.env` ファイルを確認してください。")

# 環境変数をセット
os.environ["GOOGLE_API_KEY"] = google_api_key


def main():
    # Google Gemini モデルを使用
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0  # ✅ `developer_instructions` は削除！
    )

    st.set_page_config(
        page_title="My Google AI Chatbot",
        page_icon="🤖"
    )
    st.header("Google AI Chatbot 🤖")

    # チャット履歴を初期化
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful AI assistant powered by Google.")
        ]

    # ユーザーの入力を監視
    if user_input := st.chat_input("Input your question here:"):
        st.session_state.messages.append(HumanMessage(content=user_input))

        # **🔹 修正: `messages` を単純なテキストとして渡す**
        prompt_text = "\n".join([msg.content for msg in st.session_state.messages if isinstance(msg, HumanMessage)])

        with st.spinner("Google AI is typing ..."):
            response = llm.invoke(prompt_text)  # **修正: `text` を渡す**

        st.session_state.messages.append(AIMessage(content=response.content))

    # チャット履歴を表示
    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:
            st.write(f"System message: {message.content}")

if __name__ == '__main__':
    main()
