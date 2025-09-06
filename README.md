# ⚡ YouTube RAG Assistant

> Transform any YouTube video into an interactive AI conversation using RAG (Retrieval Augmented Generation)

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![Issues](https://img.shields.io/github/issues/ZohaibCodez/yt-rag-chat.svg)](https://github.com/ZohaibCodez/yt-rag-chat/issues)
![Last Commit](https://img.shields.io/github/last-commit/ZohaibCodez/yt-rag-chat)
[![Stars](https://img.shields.io/github/stars/ZohaibCodez/yt-rag-chat?style=social)](https://github.com/ZohaibCodez/yt-rag-chat/stargazers)
[![Forks](https://img.shields.io/github/forks/ZohaibCodez/yt-rag-chat?style=social)](https://github.com/ZohaibCodez/yt-rag-chat/network/members)
[![Live Demo](https://img.shields.io/badge/demo-online-green.svg)](https://sragchat.streamlit.app)

## 🎯 Overview

YouTube RAG Assistant is a Streamlit web application that enables you to have intelligent conversations with YouTube video content. Simply provide a YouTube URL, and the app will process the video's transcript to create a searchable knowledge base that you can query using natural language.

## 🌐 Live Demo

Try it out here: [YouTube RAG Assistant Live Demo](https://sragchat.streamlit.app)

## ✨ Features

- 🎥 **YouTube Integration**: Process any YouTube video with available transcripts
- 🤖 **Multiple AI Models**: Support for various Google Gemini models (2.5 Pro, Flash, etc.)
- 💬 **Interactive Chat**: Natural language conversation with video content
- 🔍 **Smart Search**: Vector-based similarity search using FAISS
- 📊 **Session Management**: Chat history, export functionality, and session persistence
- 🎨 **Modern UI**: Clean, responsive Streamlit interface with real-time updates
- 📈 **Progress Tracking**: Visual feedback during video processing
- 🔄 **Streaming Responses**: Real-time AI response streaming

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI/ML**: Google Gemini API, LangChain
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Google Generative AI Embeddings
- **Video Processing**: YouTube Transcript API
- **Environment**: Python 3.12+, Docker support, UV package manager

## 📋 Prerequisites

- Python 3.12 or higher
- UV package manager ([Install UV](https://docs.astral.sh/uv/getting-started/installation/))
- Google Gemini API Key ([Get one here](https://ai.google.dev/))
- Internet connection for YouTube video processing

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/youtube-rag-assistant.git
cd youtube-rag-assistant
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Run the Application

```bash
uv run streamlit run app.py
```

### 4. Access the App

Open your browser and navigate to `http://localhost:8501`

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

Alternatively, you can enter your API key directly in the app's sidebar.

### Supported Models

- `gemini-2.5-pro` (Recommended)
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `gemini-2.0-flash`
- `gemini-1.5-pro`
- And more...

## 📱 How to Use

1. **Enter API Key**: Add your Google Gemini API key in the sidebar
2. **Paste YouTube URL**: Enter any YouTube video URL in the input field
3. **Process Video**: Click "🚀 Process Video" to extract and index the transcript
4. **Start Chatting**: Ask questions about the video content in natural language
5. **Export Chat**: Download your conversation history anytime

### Supported URL Formats

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `VIDEO_ID` (direct video ID)

## ⚠️ Current Limitations

- **Language Support**: Currently only supports YouTube videos with **English captions** (manual or auto-generated)
- **Transcript Dependency**: Videos must have available transcripts/captions
- **Processing Time**: Large videos may take longer to process
- **API Limits**: Subject to Google Gemini API rate limits and quotas

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   YouTube API   │───▶│  Text Splitter   │───▶│   Embeddings    │
│   (Transcripts) │    │  (Chunking)      │    │  (Google AI)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Streamlit UI   │◀───│   Chat Chain     │◀───│   FAISS Store   │
│   (Frontend)    │    │  (LangChain)     │    │ (Vector Search) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                       ┌──────────────────┐
                       │  Gemini Models   │
                       │ (Generation AI)  │
                       └──────────────────┘
```

## 🐳 Docker Support

### Build Image

```bash
docker build -t youtube-rag-assistant .
```

### Run Container

```bash
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_api_key youtube-rag-assistant
```

## 📁 Project Structure

```
yt-rag-chat/
├── .devcontainer/          # VS Code dev container config
├── .env                   # Environment variables (create this)
├── .env.example           # Environment variables template
├── .git/                  # Git repository data
├── .gitignore             # Git ignore rules
├── .python-version        # Python version specification
├── .venv/                 # Virtual environment
├── app.py                 # Main Streamlit application
├── data/                  # Data directory
│   └── demo_transcript.txt # Demo transcript file
├── Dockerfile             # Docker container config
├── Dockerfile.dev         # Development Docker config
├── LICENSE                # MIT License file
├── pyproject.toml         # Project configuration
├── README.md              # Project documentation
└── uv.lock                # Dependency lock file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Future Roadmap

- [ ] Multi-language transcript support
- [ ] Video playlist processing
- [ ] Conversation memory persistence
- [ ] Advanced chunking strategies
- [ ] Video summarization features
- [ ] Support for other video platforms
- [ ] Batch processing capabilities
- [ ] Custom embedding models

## 🐛 Known Issues

- Some videos may not have transcripts available
- Large videos (2+ hours) may take significant processing time
- API rate limiting may affect performance during peak usage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [LangChain](https://langchain.com/) for RAG implementation tools
- [Google AI](https://ai.google.dev/) for Gemini API access
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) for transcript extraction

## 📞 Support

If you encounter any issues or have questions:

- Open an [Issue](https://github.com/yourusername/youtube-rag-assistant/issues)
- Check existing issues for solutions
- Review the documentation above

---

⭐ **Star this repository if you found it helpful!**


Built with ❤️ using Streamlit and Google Gemini AI