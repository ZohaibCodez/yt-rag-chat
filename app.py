# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api.proxies import WebshareProxyConfig
import asyncio
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime


# Load environment variables
load_dotenv()

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="""You are a helpful AI assistant specialized in answering questions about YouTube videos. 
        Use the provided context chunks as your main source of truth. 
        If the context does not contain the answer, politely say you don’t know instead of making things up.
        If the question is not related to the video, politely inform the user that you can only answer questions related to the video.
        """)
        ]
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_video_id' not in st.session_state:
        st.session_state.current_video_id = None

def configure_page():
    st.set_page_config(
        page_title="YouTube RAG Chat",
        page_icon="🎥",
        layout="centered",
    )

    st.title("⚡YouTube x RAG Assistant")
    st.markdown("### Transform any YouTube video into an interactive conversation")

def handle_sidebar():
    # Sidebar for API key
    st.sidebar.header("🔑 Configuration")
    
    api_key = st.sidebar.text_input(
        "Your Google Gemini API Key",
        type="password",
        placeholder="Enter your API key...",
        help="Your key is kept only in your current browser session.",
        value=st.session_state.get("api_key", ""),
    )
    if api_key:
        st.session_state.api_key = api_key
        if len(api_key) < 20:
            st.sidebar.error("⚠️ This API key looks too short. Please check it.")
        elif not api_key.startswith("AIza"):
            st.sidebar.warning(
                "⚠️ This doesn't look like a Google API key. Double-check it."
            )
        else:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.sidebar.success("✅ API key set for this session")
    else:
        st.sidebar.info("💡 Enter your API key to start chatting")

    st.sidebar.divider()

    selected_model = st.sidebar.selectbox(
            "Generation Models",
            [
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-2.5-flash-image-preview",
                "gemini-live-2.5-flash-preview",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-2.0-flash-001",
                "gemini-2.0-flash-lite-001",
                "gemini-2.0-flash-live-001",
                "gemini-2.0-flash-live-preview-04-09",
                "gemini-2.0-flash-preview-image-generation",
                "gemini-1.5-flash",
                "gemini-1.5-pro"
            ],
            index=0,
            help="Choose the Gemini model for generation"
        )
    
    st.session_state.model = selected_model

    st.sidebar.divider()

    st.sidebar.subheader("💬 Chat Controls")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [
                SystemMessage(content="""You are a helpful AI assistant specialized in answering questions about YouTube videos. 
            Use the provided context chunks as your main source of truth. 
            If the context does not contain the answer, politely say you don’t know instead of making things up.
            If the question is not related to the video, politely inform the user that you can only answer questions related to the video."""
            )
            ]
            st.rerun()

    with col2:
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")

    st.sidebar.divider()
    st.sidebar.subheader("📊 Session Info")

    message_count = len(st.session_state.messages) - 1  # Exclude system message
    st.sidebar.metric("Messages", message_count)

    st.sidebar.info(f"**Current Model:**\n{selected_model}")

    if message_count > 0:
        st.sidebar.divider()
        chat_text = ""
        for msg in st.session_state.messages[1:]:  # Skip system message
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            chat_text += f"{role}: {msg.content}\n\n"

        st.sidebar.download_button(
            "📥 Download Chat",
            chat_text,
            f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain",
            use_container_width=True,
            help="Download your conversation history",
        )

    # Main interface
    video_url = st.text_input(
        "🔗 YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID"
    )
    video_id = extract_video_id(video_url) if video_url else ""
    st.session_state.current_video_id = video_id
    return selected_model,video_id,st.session_state.get("api_key")

def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from YouTube URL or return ID if already provided"""
    if 'youtube.com/watch?v=' in url_or_id:
        return url_or_id.split('v=')[1].split('&')[0]
    elif 'youtu.be/' in url_or_id:
        return url_or_id.split('youtu.be/')[1].split('?')[0]
    else:
        return url_or_id  # Assume it's already an ID

def handle_video_processing(video_id=""):
    if st.button("🚀 Process Video", type="primary"):
        if not video_id:
            st.error("❌ Please enter a YouTube URL!")
        else:
            with st.spinner("Processing video..."):
                st.info("🔄 Extracting transcript...")
                # try:
                #     ytt_api = YouTubeTranscriptApi()
                #     transcript_list = ytt_api.fetch(video_id)
                #     transcript = " ".join(snippet.text for snippet in transcript_list)
                # except TranscriptsDisabled:
                #     st.error("❌ Transcripts are disabled for this video.")
                #     st.stop()
                # except Exception as e:
                #     st.error(f"❌ An error occurred: {e}")
                #     st.stop()
                with open("transcript.txt", "r") as f:
                    transcript = f.read()
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                chunks = splitter.create_documents([transcript])
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.from_documents(chunks,embeddings)
                # vector_store.index_to_docstore_id
                retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})
                st.session_state['retriever'] = retriever
                st.rerun()
                st.success("✅ Video processed! Ready for questions.")

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

@st.cache_resource()
def get_chat_model(model_name:str, api_key_keyed_for_cache: str | None):
    # api_key_keyed_for_cache is unused except for cache key isolation across different keys
    return ChatGoogleGenerativeAI(model=model_name)

def display_chat_messages():
    for message in st.session_state.messages[1:]:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)

        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

def handle_user_input(chat_model, input_disabled: bool = False):
    if prompt := st.chat_input("Ask a question about the video...", disabled=input_disabled):
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        prompt_template = PromptTemplate(
            template = """
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}

                Question: {question}
                """,
                input_variables=["context","question"]
            )

        with st.chat_message("user"):
            retriever = st.session_state.get('retriever')
            if prompt and prompt.strip():
                st.write(prompt)
            elif prompt == "":
                st.warning("Please type a message before sending!")
            if not retriever:
                st.error("❌ Please process a video first!")
            else:
                retrieved_docs = retriever.invoke(prompt)
                parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                }
                )
                parser = StrOutputParser()
                main_chain = parallel_chain | prompt_template | chat_model | parser

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    message_placeholder = st.empty()
                    full_response = main_chain.invoke(prompt)

                    message_placeholder.markdown(full_response)
                    # Only add the message once, and only if there's content
                    if full_response.strip():
                        st.session_state.messages.append(
                            AIMessage(content=full_response)
                        )
                    else:
                        st.error(
                            "🚫 No response received. This model might not be working. Please try a different model."
                        )
                        
                except Exception as e:
                    error_message = str(e).lower()
                    if "not found" in error_message or "invalid" in error_message:
                        st.error(
                            "❌ This model is not available or has been deprecated. Please select a different model."
                        )
                    elif "quota" in error_message or "limit" in error_message:
                        st.error(
                            "📊 API quota exceeded. Please try again later or use a different model."
                        )
                    elif "timeout" in error_message:
                        st.error(
                            "⏱️ Request timed out. This model might be overloaded. Try a different model."
                        )
                st.rerun()

init_session_state()
configure_page()
selected_model,video_id,user_api_key = handle_sidebar()
handle_video_processing(video_id)
chat_model = None
if user_api_key:
    # Ensure env var is set for the underlying client
    os.environ["GOOGLE_API_KEY"] = user_api_key
    chat_model = get_chat_model(selected_model,user_api_key)

if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="""You are a helpful AI assistant specialized in answering questions about YouTube videos. 
        Use the provided context chunks as your main source of truth. 
        If the context does not contain the answer, politely say you don’t know instead of making things up.
        If the question is not related to the video, politely inform the user that you can only answer questions related to the video.
        """)]

    
display_chat_messages()

if chat_model is None:
    st.warning(
        "Please enter your Google Gemini API key in the sidebar to start chatting."
    )

handle_user_input(chat_model,input_disabled=(chat_model is None))