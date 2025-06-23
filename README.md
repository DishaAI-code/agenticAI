# 🎤 Voice Assistant Chatbot with Sentiment, Intent & PDF-RAG

An intelligent **voice-based assistant** powered by OpenAI, Azure Speech-to-Text, and ElevenLabs Text-to-Speech. The system understands voice input, classifies **sentiment and intent**, optionally leverages a **PDF for contextual grounding**, and replies both via **text and speech**.

---

## 🚀 Features

- 🎙️ Voice input via microphone (Streamlit + `audio_recorder_streamlit`)
- 🧠 Sentiment and intent detection using OpenAI GPT
- 📄 Document-aware responses via Retrieval-Augmented Generation (RAG) with FAISS
- 🔊 Real-time TTS playback using ElevenLabs
- 🛡️ OpenAI Moderation to ensure content safety
- 💬 Chat memory to preserve recent context
- 🌐 Streamlit interface with expandable conversation history

---

## 🛠️ Tech Stack

| Layer            | Technology                                  |
|------------------|---------------------------------------------|
| LLM              | OpenAI GPT-3.5                              |
| Speech-to-Text   | Azure Cognitive Services                    |
| Text-to-Speech   | ElevenLabs API                              |
| Embeddings       | HuggingFace `all-MiniLM-L6-v2`              |
| Vector Store     | FAISS                                       |
| Frontend         | Streamlit                                   |
| Memory           | Streamlit `st.session_state`                |
| Moderation       | OpenAI Moderation API                       |

---

## 📁 Folder Structure

voice-assistant-chatbot/
├── app.py # Main Streamlit app
├── audio_utils.py # Handles STT (Azure) & TTS (ElevenLabs)
├── conversational_memory.py # In-memory chat history logic
├── context.py # CLI-based testing interface (optional)
├── moderation_utils.py # OpenAI moderation integration
├── rag_utils.py # PDF-based RAG pipeline
├── sentiment_utils.py # Sentiment & intent analysis using GPT
├── .env # Environment variables (excluded from Git)
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 📥 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/voice-assistant-chatbot.git
cd voice-assistant-chatbot
