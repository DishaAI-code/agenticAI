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

### Repository Structure

- `app.py`  
  Main Streamlit application that runs the chatbot interface.

- `audio_utils.py`  
  Handles speech-to-text (STT) via Azure and text-to-speech (TTS) via ElevenLabs.

- `conversational_memory.py`  
  Manages in-memory chat history to maintain context during conversations.

- `context.py`  
  Provides a CLI-based testing interface (optional for debugging or testing purposes).

- `moderation_utils.py`  
  Integrates OpenAI moderation tools to filter or analyze content.

- `rag_utils.py`  
  Implements a PDF-based Retrieval-Augmented Generation (RAG) pipeline for enhanced knowledge retrieval.

- `sentiment_utils.py`  
  Performs sentiment and intent analysis using GPT models.

- `.env`  
  Environment variables (sensitive data, excluded from Git).

- `requirements.txt`  
  Lists the Python dependencies required to run the project.

- `README.md`  
  Project documentation and overview.


---

## 📥 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/DishaAI-code/agenticAI


### 2️⃣ Create .env File
- Create a .env file in the root directory and paste the following:

OPENAI_API_KEY=your_openai_key
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=your_azure_region
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_voice_id  # Optional: uses default if not provided

### 3️⃣ Install Dependencies
- Ensure Python ≥ 3.9 is installed, then run:

pip install -r requirements.txt

### 4️⃣ Run the Application

streamlit run app.py

