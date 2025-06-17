"""
Voice Assistant with Sentiment Analysis, Intent Detection, and RAG Capabilities
Streamlit application that processes voice input, generates AI responses, and maintains conversation history.
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

# Load environment variables from .env file
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
    st.subheader("📝 Transcribed Text")
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
        st.subheader("🧠 Intent")
        st.info(st.session_state.get("intent", "Pending..."))
    with col2:
        st.subheader("📊 Sentiment")
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
            print(f"🔵 New Response Generated: {response[:100]}...")

            # Generate new audio
            audio_base64 = text_to_speech_elevenlabs(response)
            if audio_base64:
                st.session_state.audio_base64 = audio_base64
                st.session_state.audio_available = True
                print("🎵 New audio generated")
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
        st.subheader("🤖 AI Response")
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
        page_title="🎙 Voice Bot",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    st.header("🎤 Voice Assistant with Sentiment + Intent + RAG")

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
        st.success("📄 PDF loaded for RAG context")
    elif st.session_state.get("pdf_path"):
        st.info("ℹ️ Using previously uploaded PDF for context")

    # Audio recording interface
    audio_bytes = audio_recorder(
        text="🎙 Start Recording",
        pause_threshold=2.0,
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
        with st.spinner("🔊 Processing your voice..."):
            transcribed_text = transcribe_audio(temp_audio_path)

        # Clean up temp file
        try:
            os.unlink(temp_audio_path)
        except Exception as e:
            print(f"⚠️ Failed to delete temp file: {e}")

        # Validate transcription
        if not transcribed_text or str(transcribed_text).strip() == "":
            st.warning("⚠️ No speech detected or transcription failed")
            return

        # Store and moderate input
        st.session_state.user_text = transcribed_text.strip()

        # Content moderation
        flagged, reasons = moderate_text(transcribed_text)
        if flagged:
            st.error(f"⚠️ Blocked due to: {', '.join(reasons)}")
            return

        # Process and display results
        display_results()

    # Display conversation history (filtering empty entries)
    with st.expander("🗂️ Show Conversation History"):
        clean_history = [
            turn for turn in st.session_state.conversation_history
            if turn.get("user", "").strip() != "" and turn.get("assistant", "").strip() != ""
        ]
        
        if not clean_history:
            st.info("No conversation history yet")
        else:
            for idx, turn in enumerate(clean_history):
                st.markdown(f"**🧑 User {idx+1}:** {turn.get('user', '').strip()}")
                st.markdown(f"**🤖 Assistant {idx+1}:** {turn.get('assistant', '').strip()}")
                st.markdown("---")

if __name__ == "__main__":
    main()




#3..
# """
# Voice Assistant with Sentiment Analysis, Intent Detection, and RAG Capabilities
# Streamlit application that processes voice input, generates AI responses, and maintains conversation history.
# """

# import os
# import time
# import streamlit as st
# from openai import OpenAI
# from moderation_utils import moderate_text
# from rag_utils import process_text_with_llm, process_pdf_and_ask
# from audio_recorder_streamlit import audio_recorder
# import base64
# import requests
# import tempfile
# from dotenv import load_dotenv
# from azure.cognitiveservices.speech import SpeechConfig, AudioConfig, SpeechRecognizer
# from conversational_memory import get_conversation_history

# # Load environment variables from .env file
# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # --------------------------
# # PERSISTENT MEMORY HANDLING
# # --------------------------

# def load_history():
#     """Initialize or load conversation history from persistent storage"""
#     return []  # Currently using in-memory storage

# def save_history(history):
#     """Placeholder for saving conversation history to persistent storage"""
#     pass  # Implement database/file storage if needed

# # --------------------------
# # SPEECH SYNTHESIS (TTS)
# # --------------------------

# def text_to_speech_elevenlabs(text: str):
#     """
#     Convert text to speech using ElevenLabs API
#     Args:
#         text: The text to convert to speech
#     Returns:
#         Base64 encoded audio data or None if failed
#     """
#     try:
#         xi_api_key = os.getenv("ELEVENLABS_API_KEY")
#         voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
#         url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

#         headers = {
#             "Accept": "audio/mpeg",
#             "Content-Type": "application/json",
#             "xi-api-key": xi_api_key
#         }

#         data = {
#             "text": text,
#             "model_id": "eleven_monolingual_v1",
#             "voice_settings": {
#                 "stability": 0.5,
#                 "similarity_boost": 0.5
#             }
#         }

#         print("🔊 Sending to ElevenLabs TTS:", text[:100] + "..." if len(text) > 100 else text)
#         response = requests.post(url, json=data, headers=headers)
        
#         if response.status_code == 200:
#             return base64.b64encode(response.content).decode("utf-8")
#         else:
#             st.error(f"ElevenLabs TTS failed: {response.text}")
#             return None
#     except Exception as e:
#         st.error(f"TTS Error: {str(e)}")
#         return None

# # --------------------------
# # SPEECH RECOGNITION (STT)
# # --------------------------

# def transcribe_audio(file_path):
#     """
#     Transcribe audio file using Azure Speech-to-Text
#     Args:
#         file_path: Path to audio file
#     Returns:
#         Transcribed text or None if failed
#     """
#     try:
#         speech_key = os.getenv("AZURE_SPEECH_KEY")
#         service_region = os.getenv("AZURE_SPEECH_REGION")
#         speech_config = SpeechConfig(subscription=speech_key, region=service_region)
#         audio_config = AudioConfig(filename=file_path)
#         recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
#         result = recognizer.recognize_once()
#         return result.text if result.text and result.text.strip() != "" else None
#     except Exception as e:
#         st.error(f"Transcription error: {str(e)}")
#         return None

# # --------------------------
# # TEXT ANALYSIS
# # --------------------------

# def analyze_sentiment_intent(text):
#     """
#     Analyze text for sentiment and intent using OpenAI
#     Args:
#         text: Input text to analyze
#     Returns:
#         Tuple of (sentiment, intent) or (None, None) if failed
#     """
#     try:
#         # Sentiment analysis
#         sentiment_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "Classify sentiment as Positive/Negative/Neutral only"},
#                 {"role": "user", "content": f"Text: {text}"}
#             ]
#         )
#         sentiment = sentiment_response.choices[0].message.content.strip()

#         # Intent analysis
#         intent_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "Classify intent in 2-3 words (e.g., 'Question', 'Request', 'Complaint')"},
#                 {"role": "user", "content": f"Text: {text}"}
#             ]
#         )
#         intent = intent_response.choices[0].message.content.strip()

#         return sentiment, intent
#     except Exception as e:
#         st.error(f"Analysis error: {e}")
#         return None, None

# # --------------------------
# # RESULT DISPLAY
# # --------------------------

# def display_results():
#     """Display transcribed text, analysis results, and AI response"""
#     current_input = st.session_state.get("user_text", "")
#     if not current_input or str(current_input).strip() == "":
#         return

#     # Display transcribed text
#     st.subheader("📝 Transcribed Text")
#     st.text_area("Your speech as text", 
#                 value=current_input, 
#                 height=100, 
#                 disabled=True,
#                 key=f"transcribed_{int(time.time())}")

#     # Analyze sentiment and intent if not already done
#     if not st.session_state.get("sentiment") or not st.session_state.get("intent"):
#         with st.spinner("Analyzing sentiment and intent..."):
#             sentiment, intent = analyze_sentiment_intent(current_input)
#             st.session_state.sentiment = sentiment
#             st.session_state.intent = intent

#     # Display analysis results in columns
#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("🧠 Intent")
#         st.info(st.session_state.get("intent", "Pending..."))
#     with col2:
#         st.subheader("📊 Sentiment")
#         st.success(st.session_state.get("sentiment", "Pending..."))

#     # Generate AI response
#     with st.spinner("Generating AI response..."):
#         if st.session_state.get("pdf_path"):
#             with open(st.session_state.pdf_path, "rb") as f:
#                 response = process_pdf_and_ask(f, current_input)
#         else:
#             response = process_text_with_llm(current_input)

#         # Only update if we have a valid response
#         if response and str(response).strip() != "":
#             st.session_state.ai_response = response
#             print(f"🔵 New Response Generated: {response[:100]}...")

#             # Generate new audio
#             audio_base64 = text_to_speech_elevenlabs(response)
#             if audio_base64:
#                 st.session_state.audio_base64 = audio_base64
#                 st.session_state.audio_available = True
#                 print("🎵 New audio generated")
#             else:
#                 st.session_state.audio_available = False

#             # Update conversation history only with valid entries
#             st.session_state.conversation_history.append({
#                 "user": current_input,
#                 "assistant": response
#             })
#             save_history(st.session_state.conversation_history)

#     # Display AI response
#     if st.session_state.get("ai_response"):
#         st.subheader("🤖 AI Response")
#         st.write(st.session_state.ai_response)

#         # Play audio if available
#         if st.session_state.get("audio_available"):
#             audio_key = f"audio_{int(time.time())}"
#             audio_html = f"""
#                 <audio id="{audio_key}" controls autoplay>
#                     <source src="data:audio/mpeg;base64,{st.session_state.audio_base64}" type="audio/mpeg">
#                 </audio>
#                 <script>
#                     document.getElementById("{audio_key}").play().catch(e => console.log("Audio play failed:", e));
#                 </script>
#             """
#             st.components.v1.html(audio_html, height=50)

# # --------------------------
# # SESSION MANAGEMENT
# # --------------------------

# def clear_previous_results():
#     """Clear temporary session state variables while preserving conversation history"""
#     keys_to_keep = ['conversation_history', 'pdf_path']
#     for key in list(st.session_state.keys()):
#         if key not in keys_to_keep:
#             del st.session_state[key]
#     print("🧹 Cleared temporary session state")

# # --------------------------
# # UI SETUP
# # --------------------------

# def page_setup():
#     """Configure the Streamlit page settings"""
#     st.set_page_config(
#         page_title="🎙 Voice Bot",
#         layout="centered",
#         initial_sidebar_state="expanded"
#     )
#     st.header("🎤 Voice Assistant with Sentiment + Intent + RAG")

# # --------------------------
# # MAIN APPLICATION
# # --------------------------

# def main():
#     """Main application logic"""
#     page_setup()

#     # Initialize conversation history if not exists
#     if "conversation_history" not in st.session_state:
#         st.session_state.conversation_history = load_history()

#     # PDF upload for RAG context
#     uploaded_pdf = st.file_uploader("Upload PDF (optional for context)", type=["pdf"])
#     if uploaded_pdf:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
#             f.write(uploaded_pdf.read())
#             st.session_state.pdf_path = f.name
#         st.success("📄 PDF loaded for RAG context")
#     elif st.session_state.get("pdf_path"):
#         st.info("ℹ️ Using previously uploaded PDF for context")

#     # Audio recording interface
#     audio_bytes = audio_recorder(
#         text="🎙 Start Recording",
#         pause_threshold=2.0,
#         key="audio_recorder"
#     )

#     # Process audio when recorded
#     if audio_bytes:
#         clear_previous_results()

#         # Save audio to temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#             f.write(audio_bytes)
#             temp_audio_path = f.name

#         # Transcribe audio
#         with st.spinner("🔊 Processing your voice..."):
#             transcribed_text = transcribe_audio(temp_audio_path)

#         # Clean up temp file
#         try:
#             os.unlink(temp_audio_path)
#         except Exception as e:
#             print(f"⚠️ Failed to delete temp file: {e}")

#         # Validate transcription
#         if not transcribed_text or str(transcribed_text).strip() == "":
#             st.warning("⚠️ No speech detected or transcription failed")
#             return

#         # Store and moderate input
#         st.session_state.user_text = transcribed_text.strip()

#         # Content moderation
#         flagged, reasons = moderate_text(transcribed_text)
#         if flagged:
#             st.error(f"⚠️ Blocked due to: {', '.join(reasons)}")
#             return

#         # Process and display results
#         display_results()

#     # Display conversation history (filtering empty entries)
#     with st.expander("🗂️ Show Conversation History"):
#         clean_history = [
#             turn for turn in st.session_state.conversation_history
#             if turn.get("user", "").strip() != "" and turn.get("assistant", "").strip() != ""
#         ]
        
#         if not clean_history:
#             st.info("No conversation history yet")
#         else:
#             for idx, turn in enumerate(clean_history):
#                 st.markdown(f"**🧑 User {idx+1}:** {turn.get('user', '').strip()}")
#                 st.markdown(f"**🤖 Assistant {idx+1}:** {turn.get('assistant', '').strip()}")
#                 st.markdown("---")

# if __name__ == "__main__":
#     main()