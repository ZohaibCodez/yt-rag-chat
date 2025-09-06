# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
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
import time

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/embedding-001"
RETRIEVER_K = 4
DEFAULT_SYSTEM_MESSAGE = """
You are YouTube RAG Assistant 📺🤖. 
Your role is to help users understand and explore the content of YouTube videos by using the retrieved transcript context. 

Follow these rules:
1. Always prioritize the transcript/context when answering questions about the video. 
   - Summarize, explain, or extract details only from the retrieved text.
   - If the answer is not present in the transcript, clearly say you don’t know.
   
2. Also maintain awareness of the ongoing chat history (previous user and assistant messages in this session). 
   - If the user asks about their previous messages, use the chat history instead of the transcript.
   - For example, if the user asks “what is my name” and they told you earlier, answer from the chat history.

3. Never invent facts. If the context or chat history does not contain the answer, politely say you don’t know. 

4. Keep your tone friendly, clear, and concise. 
   - Use bullet points or short paragraphs if the answer is long. 
   - Do not repeat the system instructions in your answers.
"""

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
        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_video_id" not in st.session_state:
        st.session_state.current_video_id = None


def configure_page():
    st.set_page_config(
        page_title="YouTube RAG Chat",
        page_icon="🎥",
        layout="centered",
    )

    st.title("⚡YouTube x RAG Assistant")
    st.markdown("### Transform any YouTube video into an interactive conversation")

def center_app():
    st.markdown(
        """
        <style>
        /* Center the main content */
        .block-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            # text-align: center;
            max-width: 800px; /* prevent content from stretching too wide */
            margin: auto;
        }

        #transform-any-you-tube-video-into-an-interactive-conversation,#you-tube-x-rag-assistant{
        text-align:center;
        }

        /* Center text inputs and buttons */
        .stTextInput, .stButton {
            width: 100% !important;
            # max-width: 500px;
            margin: auto;
        }

        @media (max-width: 768px) {
                .stVerticalBlock{
                align-items:center;
                }
            }
        
        </style>
        """,
        unsafe_allow_html=True,
    )



def handle_new_video_button():
    """Clear current video and start fresh"""
    if st.sidebar.button("🔄 New Video", use_container_width=True):
        # Clear video-related session state
        if "retriever" in st.session_state:
            del st.session_state["retriever"]
        if "current_video_id" in st.session_state:
            st.session_state.current_video_id = None

        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]

        st.success("🔄 Ready for new video!")
        time.sleep(1)
        st.rerun()


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
            "gemini-1.5-pro",
        ],
        index=0,
        help="Choose the Gemini model for generation",
    )

    st.session_state.model = selected_model

    st.sidebar.divider()

    st.sidebar.subheader("💬 Chat Controls")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
            st.rerun()

    with col2:
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")

    handle_new_video_button()

    st.sidebar.divider()
    st.sidebar.subheader("📊 Session Info")

    message_count = len(st.session_state.messages) - 1  # Exclude system message
    video_processed = (
        "retriever" in st.session_state
        and st.session_state.get("retriever") is not None
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Messages", message_count)
    with col2:
        st.metric("Video", "✅" if video_processed else "❌")

    if video_processed:
        st.sidebar.success("🎥 Video ready for chat")
    else:
        st.sidebar.info("📹 No video processed yet")

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
        "🔗 YouTube Video URL", placeholder="https://www.youtube.com/watch?v=VIDEO_ID"
    )
    video_id = extract_video_id(video_url) if video_url else ""
    st.session_state.current_video_id = video_id

    if video_id:
        display_video_info(video_id)
    elif video_url and not video_id:
        st.error("❌ Invalid YouTube URL format")
        st.info("💡 Please use: youtube.com/watch?v=... or youtu.be/...")

    return selected_model, video_id, st.session_state.get("api_key")


def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from YouTube URL or return ID if already provided"""
    if "youtube.com/watch?v=" in url_or_id:
        video_id = url_or_id.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url_or_id:
        video_id = url_or_id.split("youtu.be/")[1].split("?")[0]
    else:
        video_id = url_or_id  # Assume it's already an ID

    if len(video_id) == 11:
        return video_id
    else:
        return ""


def display_video_info(video_id: str):
    """Display basic video information"""
    if video_id:
        col1, col2 = st.columns([1, 2])
        with col1:
            # Show video thumbnail
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            st.image(thumbnail_url, width="stretch")

        with col2:
            st.success("✅ Valid YouTube URL detected!")
            st.info(f"📺 Video ID: `{video_id}`")
            st.markdown(
                f"🔗 [Open in YouTube](https://www.youtube.com/watch?v={video_id})"
            )


def handle_video_processing(video_id=""):
    if st.button("🚀 Process Video", type="primary"):
        user_api_key = st.session_state.get("api_key", "")
        if not user_api_key:
            st.error("❌ Please enter your Google Gemini API key in the sidebar first!")
            st.info("💡 You need a valid API key to process videos and chat")
            return
        elif not video_id:
            st.error("❌ Please enter a valid YouTube URL!")
            st.info(
                "💡 Supported formats:\n- https://www.youtube.com/watch?v=VIDEO_ID\n- https://youtu.be/VIDEO_ID\n- VIDEO_ID"
            )
            return
        else:
            with st.spinner("Processing video..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Extract transcript
                status_text.text("🔄 Step 1/4: Extracting transcript...")
                progress_bar.progress(25)
                try:
                    ytt_api = YouTubeTranscriptApi()
                    transcript_list = ytt_api.fetch(video_id)
                    transcript = " ".join(snippet.text for snippet in transcript_list)
                except TranscriptsDisabled:
                    st.error("❌ Transcripts are disabled for this video.")
                    st.stop()
                except Exception as e:
                    st.error(
                        f"❌ An error occurred. This video is not transcribed:(\nWe couldn’t fetch the transcript. YouTube may be blocking requests from your current network. Please try again later or switch to another connection"
                    )
                    st.stop()

                # Step 2: Split into chunks and create vector store
                status_text.text("📄 Step 2/4: Splitting into chunks...")
                progress_bar.progress(50)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )
                chunks = splitter.create_documents([transcript])

                # Step 3: Create embeddings
                status_text.text("🧠 Step 3/4: Creating embeddings...")
                progress_bar.progress(75)
                embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
                # Step 4: Build vector store
                status_text.text("🗂️ Step 4/4: Building search index...")
                progress_bar.progress(100)
                vector_store = FAISS.from_documents(chunks, embeddings)
                # vector_store.index_to_docstore_id
                retriever = vector_store.as_retriever(
                    search_type="similarity", search_kwargs={"k": RETRIEVER_K}
                )
                st.session_state["retriever"] = retriever
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Re-run to update UI state
                st.success("✅ Video processed! Ready for questions.")
                time.sleep(2)
                st.rerun()


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


@st.cache_resource()
def get_chat_model(model_name: str, api_key_keyed_for_cache: str | None):
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
    if prompt := st.chat_input(
        "Ask a question about the video...", disabled=input_disabled
    ):
        if not prompt.strip():
            st.warning("Please type a message before sending!")
            return

        st.session_state.messages.append(HumanMessage(content=prompt))

        prompt_template = PromptTemplate(
            template="""Based on this video transcript content:

            {context}

            Question: {question}""",
            input_variables=["context", "question"],
        )

        with st.chat_message("user"):
            st.write(prompt)

        retriever = st.session_state.get("retriever")
        if not retriever:
            with st.chat_message("assistant"):
                error_msg = (
                    "❌ Please process a video first to enable question answering."
                )
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(content=error_msg))
            return
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing video content..."):
                try:
                    retrieved_docs = retriever.invoke(prompt)
                    if not retrieved_docs:
                        no_context_msg = "🤷‍♂️ I couldn't find relevant information in the video transcript for your question."
                        st.warning(no_context_msg)
                        st.session_state.messages.append(
                            AIMessage(content=no_context_msg)
                        )
                        return
                    parallel_chain = RunnableParallel(
                        {
                            "context": retriever | RunnableLambda(format_docs),
                            "question": RunnablePassthrough(),
                        }
                    )
                    parser = StrOutputParser()
                    main_chain = parallel_chain | prompt_template | chat_model | parser

                    message_placeholder = st.empty()
                    full_response = ""

                    # Stream the response using stream method (synchronous)
                    for chunk in main_chain.stream(prompt):
                        if chunk and chunk.strip():
                            full_response += chunk
                            message_placeholder.markdown(
                                full_response + "▌"
                            )  # Cursor indicator

                    # Remove cursor and display final response
                    if full_response and full_response.strip():
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append(
                            AIMessage(content=full_response)
                        )
                    else:
                        error_msg = (
                            "🚫 No response received. Please try a different model."
                        )
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append(AIMessage(content=error_msg))

                    # Rerun to refresh the UI after streaming
                    st.rerun()

                except Exception as e:
                    error_message = str(e).lower()
                    if "not found" in error_message or "invalid" in error_message:
                        error_msg = "❌ This model is not available. Please select a different model."
                    elif "quota" in error_message or "limit" in error_message:
                        error_msg = "📊 API quota exceeded. Please try again later or use a different model."
                    elif "timeout" in error_message:
                        error_msg = (
                            "⏱️ Request timed out. Try a different model or try again."
                        )
                    else:
                        error_msg = f"❌ An error occurred. Try selecting different model or check your api key:("

                    st.error(error_msg)
                    st.session_state.messages.append(AIMessage(content=error_msg))
            # st.rerun()


init_session_state()
configure_page()
center_app()
selected_model, video_id, user_api_key = handle_sidebar()
handle_video_processing(video_id)
chat_model = None
if user_api_key:
    # Ensure env var is set for the underlying client
    os.environ["GOOGLE_API_KEY"] = user_api_key
    chat_model = get_chat_model(selected_model, user_api_key)


display_chat_messages()

if chat_model is None:
    st.warning(
        "Please enter your Google Gemini API key in the sidebar to start chatting."
    )

handle_user_input(chat_model, input_disabled=(chat_model is None))
