"""
📁 app.py

🎯 Purpose:
Main entry-point Streamlit application that powers a voice-driven conversational assistant
with sentiment & intent detection, document-aware (RAG) query handling, and audio playback.

🧩 Core Features:

1. 🌍 Environment Setup:
   - Loads API keys using `dotenv` for OpenAI, Azure, ElevenLabs.

2. 📄 PDF Upload (Optional Context):
   - Allows users to upload PDFs.
   - Enables Retrieval-Augmented Generation via `rag_utils.py`.

3. 🎙 Voice Input via `audio_recorder_streamlit`:
   - Captures microphone input.
   - Saves audio as temp `.wav` file for transcription.

4. 🧾 Speech Transcription:
   - Uses Azure Cognitive Services STT to convert audio to text.
   - Automatically deletes temporary audio file after use.

5. ⚠️ Content Moderation:
   - Filters unsafe/inappropriate input using OpenAI’s moderation API via `moderation_utils.py`.

6. 💬 Sentiment and Intent Analysis:
   - Calls `analyze_sentiment_intent()` from `sentiment_utils.py` to classify:
     - Sentiment: Positive / Negative / Neutral
     - Intent: e.g., Question, Request, Complaint

7. 🧠 RAG or Standard GPT Response:
   - If PDF provided: uses `process_pdf_and_ask()` to generate response from document context.
   - Else: falls back to GPT-based chat via `process_text_with_llm()`.

8. 🔊 Text-to-Speech Playback:
   - Uses ElevenLabs TTS to synthesize AI responses.
   - Encodes audio in base64 and plays it via in-browser `<audio>` tag.

9. 🧠 Conversational Memory:
   - Maintains chat history in `st.session_state["conversation_history"]`
   - Shows message log in collapsible UI section for traceability.

✅ End Result:
- Seamless voice-to-AI interaction.
- Document-aware answers (if needed).
- Feedback in both visual and audible form.

"""
import os
import time
import streamlit as st
from openai import OpenAI
from moderation_utils import moderate_text
from rag_utils import process_text_with_llm, process_pdf_and_ask
from audio_recorder_streamlit import audio_recorder
import base64
import requests
import tempfile
from dotenv import load_dotenv
from utils.sentiment_utils import analyze_sentiment_intent
from utils.audio_utils import text_to_speech_elevenlabs, transcribe_audio
from conversational_memory import get_conversation_history

# ----------------------------------------------------------
# ENVIRONMENT SETUP
# ----------------------------------------------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------
# PERSISTENT MEMORY HANDLING
# --------------------------

def load_history():
    """Initialize or load conversation history from persistent storage"""
    return []  # Currently using in-memory storage

def save_history(history):
    """Placeholder for saving conversation history to persistent storage"""
    pass  # Implement database/file storage if needed

# --------------------------
# RESULT DISPLAY
# --------------------------

def display_results():
    """Display transcribed text, analysis results, and AI response"""
    current_input = st.session_state.get("user_text", "")
    if not current_input or str(current_input).strip() == "":
        return

    # Display transcribed text
    st.subheader("Transcribed Text")
    st.text_area("Your speech as text", 
                value=current_input, 
                height=100, 
                disabled=True,
                key=f"transcribed_{int(time.time())}")

    # Analyze sentiment and intent if not already done
    if not st.session_state.get("sentiment") or not st.session_state.get("intent"):
        with st.spinner("Analyzing sentiment and intent..."):
            sentiment, intent = analyze_sentiment_intent(current_input)
            st.session_state.sentiment = sentiment
            st.session_state.intent = intent

    # Display analysis results in columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Intent")
        st.info(st.session_state.get("intent", "Pending..."))
    with col2:
        st.subheader("Sentiment")
        st.success(st.session_state.get("sentiment", "Pending..."))

    # Generate AI response
    with st.spinner("Generating AI response..."):
        if st.session_state.get("pdf_path"):
            with open(st.session_state.pdf_path, "rb") as f:
                response = process_pdf_and_ask(f, current_input)
        else:
            response = process_text_with_llm(current_input)

        # Only update if we have a valid response
        if response and str(response).strip() != "":
            st.session_state.ai_response = response
            print(f"New Response Generated: {response[:100]}...")

            # Generate new audio
            audio_base64 = text_to_speech_elevenlabs(response)
            if audio_base64:
                st.session_state.audio_base64 = audio_base64
                st.session_state.audio_available = True
                print("New audio generated")
            else:
                st.session_state.audio_available = False

            # Update conversation history only with valid entries
            st.session_state.conversation_history.append({
                "user": current_input,
                "assistant": response
            })
            save_history(st.session_state.conversation_history)

    # Display AI response
    if st.session_state.get("ai_response"):
        st.subheader("AI Response")
        st.write(st.session_state.ai_response)

        # Play audio if available
        if st.session_state.get("audio_available"):
            audio_key = f"audio_{int(time.time())}"
            audio_html = f"""
                <audio id="{audio_key}" controls autoplay>
                    <source src="data:audio/mpeg;base64,{st.session_state.audio_base64}" type="audio/mpeg">
                </audio>
                <script>
                    document.getElementById("{audio_key}").play().catch(e => console.log("Audio play failed:", e));
                </script>
            """
            st.components.v1.html(audio_html, height=50)

# --------------------------
# SESSION MANAGEMENT
# --------------------------

def clear_previous_results():
    """Clear temporary session state variables while preserving conversation history"""
    keys_to_keep = ['conversation_history', 'pdf_path']
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    print("🧹 Cleared temporary session state")

# --------------------------
# UI SETUP
# --------------------------

def page_setup():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title=" Voice Bot",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    st.header("Voice Assistant with Sentiment + Intent + RAG")

# --------------------------
# MAIN APPLICATION
# --------------------------

def main():
    """Main application logic"""
    page_setup()

    # Initialize conversation history if not exists
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = load_history()

    # PDF upload for RAG context
    uploaded_pdf = st.file_uploader("Upload PDF (optional for context)", type=["pdf"])
    if uploaded_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_pdf.read())
            st.session_state.pdf_path = f.name
        st.success("PDF loaded for RAG context")
    elif st.session_state.get("pdf_path"):
        st.info("Using previously uploaded PDF for context")

    # Audio recording interface
    audio_bytes = audio_recorder(
        text=" Start Recording",
        key="audio_recorder"
    )

    # Process audio when recorded
    if audio_bytes:
        clear_previous_results()

        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_audio_path = f.name

        # Transcribe audio
        with st.spinner(" Processing your voice..."):
            transcribed_text = transcribe_audio(temp_audio_path)

        # Clean up temp file
        try:
            os.unlink(temp_audio_path)
        except Exception as e:
            print(f" Failed to delete temp file: {e}")

        # Validate transcription
        if not transcribed_text or str(transcribed_text).strip() == "":
            st.warning(" No speech detected or transcription failed")
            return

        # Store and moderate input
        st.session_state.user_text = transcribed_text.strip()

        # Content moderation
        flagged, reasons = moderate_text(transcribed_text)
        if flagged:
            st.error(f" Blocked due to: {', '.join(reasons)}")
            return

        # Process and display results
        display_results()

    # Display conversation history (filtering empty entries)
    with st.expander(" Show Conversation History"):
        clean_history = [
            turn for turn in st.session_state.conversation_history
            if turn.get("user", "").strip() != "" and turn.get("assistant", "").strip() != ""
        ]
        
        if not clean_history:
            st.info("No conversation history yet")
        else:
            for idx, turn in enumerate(clean_history):
                st.markdown(f"** User {idx+1}:** {turn.get('user', '').strip()}")
                st.markdown(f"** Assistant {idx+1}:** {turn.get('assistant', '').strip()}")
                st.markdown("---")

if __name__ == "__main__":
    main()




