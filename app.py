import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

from rag_utils import process_pdf_and_ask
from audio_utils import (
    transcribe_audio, 
    text_to_speech,
    recognize_speech_azure
)
from moderation_utils import moderate_text

# === Load Keys ===
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
eleven_key = os.getenv("ELEVENLABS_API_KEY")
client = OpenAI(api_key=openai_key)

st.set_page_config(page_title="Multimodal AI QnA")
st.title("🎙️ Multimodal RAG + Voice Bot")



# === PDF Upload Section ===
st.header("📄 Upload PDF for Q&A")
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file:
    st.success("PDF uploaded. Ask a question below.")
    
    # Initialize session state for question input
    if 'pdf_question' not in st.session_state:
        st.session_state.pdf_question = ""
    
    # Text input that syncs with session state
    question_input = st.text_input(
        "Type your question here",
        value=st.session_state.pdf_question,
        key="question_input_widget"
    )
    
    # Update session state when text input changes
    if question_input != st.session_state.pdf_question:
        st.session_state.pdf_question = question_input

    # Audio recording option for PDF questions
    if st.button("🎙️ Record Question (Azure)"):
        st.info("Listening with Azure... (Speak now)")
        transcribed_text = recognize_speech_azure()
        
        if transcribed_text and transcribed_text != "No speech recognized.":
            st.session_state.pdf_question = transcribed_text
            st.success("Question recorded successfully!")
            st.rerun()
        else:
            st.warning("No speech detected or error occurred.")

    if st.button("🧠 Ask PDF"):
        # Get the current question from session state
        current_question = st.session_state.pdf_question
        
        if not current_question.strip():
            st.warning("Provide a question via text or voice.")
        else:
            flagged, reasons = moderate_text(current_question)
            if flagged is None:
                st.error("Moderation failed.")
            elif flagged:
                st.error("Blocked by moderation.")
                st.info(f"Reason: {', '.join(reasons)}")
            else:
                try:
                    answer = process_pdf_and_ask(pdf_file, current_question)
                    st.subheader("📚 Answer from Document")
                    st.success(answer)
                    st.audio(text_to_speech(answer), format="audio/mp3")
                except Exception as e:
                    st.error(f"Error: {e}")

# === Free Text Analysis ===
st.header("💬 Analyze Free Text")

# Initialize session state for message input
if 'message_input' not in st.session_state:
    st.session_state.message_input = ""

# Text area that syncs with session state
user_input = st.text_area(
    "Enter your message", 
    value=st.session_state.message_input, 
    height=100, 
    key="message_input_widget"
)

# Update session state when text area changes
if user_input != st.session_state.message_input:
    st.session_state.message_input = user_input

# Add Azure microphone recording option
if st.button("🎙️ Record Audio for Analysis (Azure)"):
    st.info("Listening with Azure... (Speak now)")
    transcribed_text = recognize_speech_azure()
    
    if transcribed_text and transcribed_text != "No speech recognized.":
        st.session_state.message_input = transcribed_text
        st.success("Audio transcribed successfully!")
        st.rerun()
    else:
        st.warning("No speech detected or error occurred.")

if st.button("🔍 Analyze Text"):
    # Get the current text from session state
    analysis_text = st.session_state.message_input
    
    if not analysis_text.strip():
        st.warning("Enter some text or record audio.")
    else:
        flagged, reasons = moderate_text(analysis_text)
        if flagged is None:
            st.error("Moderation failed.")
        elif flagged:
            st.error("Blocked by moderation.")
            st.info(f"Reason: {', '.join(reasons)}")
        else:
            try:
                intent_prompt = [
                    {"role": "system", "content": "You're an intent classifier."},
                    {"role": "user", "content": analysis_text}
                ]
                intent_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=intent_prompt
                )
                intent_label = intent_response.choices[0].message.content.strip()

                sentiment_prompt = [
                    {"role": "system", "content": "You're a sentiment classifier."},
                    {"role": "user", "content": analysis_text}
                ]
                sentiment_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=sentiment_prompt
                )
                sentiment = sentiment_response.choices[0].message.content.strip()

                st.subheader("🧭 Intent:")
                st.info(intent_label)

                st.subheader("📈 Sentiment:")
                st.success(sentiment)

            except Exception as e:
                st.error(f"Error: {e}")
                
                
                
                

#-------------------------------------------updated---------------------------
# 2. updation 

# import os
# import tempfile
# import sounddevice as sd
# import numpy as np
# from scipy.io.wavfile import write
# from dotenv import load_dotenv
# from openai import OpenAI
# import streamlit as st
# import webrtcvad

# from rag_utils import process_pdf_and_ask
# from audio_utils import transcribe_audio, text_to_speech
# from moderation_utils import moderate_text

# # === Load environment variables ===
# load_dotenv()
# openai_key = os.getenv("OPENAI_API_KEY")
# eleven_key = os.getenv("ELEVENLABS_API_KEY")
# ELEVENLABS_VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"  # Roger voice

# print(" OpenAI Key Loaded:", openai_key)
# print(" ElevenLabs Key Loaded:", eleven_key)

# # === Initialize OpenAI client ===
# client = OpenAI(api_key=openai_key)

# # === Record Audio with VAD ===
# def record_audio_with_vad(duration=10, aggressiveness=2):
#     fs = 16000
#     vad = webrtcvad.Vad(aggressiveness)
#     frame_duration = 30  # ms
#     frame_size = int(fs * frame_duration / 1000)

#     def is_speech(frame):
#         return vad.is_speech(frame, sample_rate=fs)

#     recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
#     sd.wait()

#     audio_bytes = recording.flatten().tobytes()
#     frames = [audio_bytes[i:i + frame_size * 2] for i in range(0, len(audio_bytes), frame_size * 2)]
#     voiced = [f for f in frames if is_speech(f)]

#     if not voiced:
#         return None

#     audio_array = np.frombuffer(b"".join(voiced), dtype=np.int16)

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#         write(f.name, fs, audio_array.reshape(-1, 1))
#         return f.name

# # === Streamlit UI ===
# st.set_page_config(page_title="Multimodal AI QnA")
# st.title("🎙️ Multimodal RAG + Voice Bot")
# st.markdown("""
# Handles **Text**, **Audio**, and **PDFs** with:
# - Intent detection ✅ (via OpenAI)
# - Sentiment classification ✅
# - OpenAI moderation ✅
# - Real-time voice transcription & TTS 🔊
# """)

# # === PDF Upload Section ===
# st.header(" Upload PDF for Q&A")
# pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

# if pdf_file:
#     st.success(" PDF uploaded. Ask a question below.")
    
#     st.markdown("###  Type Your Question")
#     question = st.text_input("Enter a question based on the PDF")

#     st.markdown("###  Or Use Microphone (with VAD)")

#     if st.button(" Record Audio"):
#         st.info("Listening... Speak now.")
#         audio_path = record_audio_with_vad()
#         if audio_path:
#             transcribed = transcribe_audio(audio_path)
#             st.success(f" Transcribed: {transcribed}")
#             question = transcribed
#         else:
#             st.warning(" No speech detected.")

#     uploaded_audio = st.file_uploader("Or upload your own audio (WAV)", type=["wav"])
#     if uploaded_audio:
#         with st.spinner(" Transcribing uploaded audio..."):
#             question = transcribe_audio(uploaded_audio)
#             st.success(f" Transcribed: {question}")

#     if st.button(" Ask PDF"):
#         if not question.strip():
#             st.warning(" Provide a question via text or audio.")
#         else:
#             flagged, reasons = moderate_text(question)
#             if flagged is None:
#                 st.error(" Moderation system failed.")
#             elif flagged:
#                 st.error(" Blocked by moderation.")
#                 st.info(f"Reason: {', '.join(reasons)}")
#             else:
#                 try:
#                     answer = process_pdf_and_ask(pdf_file, question)
#                     st.subheader(" Answer from Document:")
#                     st.success(answer)
#                     st.audio(text_to_speech(answer), format="audio/mp3")
#                 except Exception as e:
#                     st.error(f" Error: {e}")

# # === Free Text Analysis Section ===
# st.header(" Analyze Free Text")
# user_input = st.text_area("Enter your message", "", height=100)

# if st.button("Analyze Text"):
#     if not user_input.strip():
#         st.warning(" Please enter text.")
#     else:
#         flagged, reasons = moderate_text(user_input)
#         if flagged is None:
#             st.error(" Moderation failed.")
#         elif flagged:
#             st.error(" Blocked by moderation.")
#             st.info(f"Reason: {', '.join(reasons)}")
#         else:
#             try:
#                 # === Intent classification via OpenAI ===
#                 intent_prompt = [
#                     {"role": "system", "content": "You're an intent classification engine. Based on the user message, return only the intent like 'travel', 'product inquiry', 'greeting', 'complaint', 'support', 'finance', 'healthcare', 'goodbye', etc. Respond with only the intent label."},
#                     {"role": "user", "content": user_input}
#                 ]
#                 intent_response = client.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=intent_prompt
#                 )
#                 intent_label = intent_response.choices[0].message.content.strip()

#                 # === Sentiment classification ===
#                 sentiment_prompt = [
#                     {"role": "system", "content": "You're a sentiment classifier. Respond with 'Positive', 'Negative', or 'Neutral' and explain briefly."},
#                     {"role": "user", "content": user_input}
#                 ]
#                 sentiment_response = client.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=sentiment_prompt
#                 )
#                 sentiment = sentiment_response.choices[0].message.content.strip()

#                 st.subheader(" Detected Intent:")
#                 st.info(intent_label)

#                 st.subheader(" Predicted Sentiment:")
#                 st.success(sentiment)

#             except Exception as e:
#                 st.error(f" Error during analysis: {e}")



#---------------------------------------------3.

# import os
# from dotenv import load_dotenv
# from openai import OpenAI
# import streamlit as st

# from rag_utils import process_pdf_and_ask
# from audio_utils import (
#     transcribe_audio, 
#     text_to_speech, 
#     record_audio_with_vad,
#     recognize_speech_azure
# )
# from moderation_utils import moderate_text

# # === Load Keys ===
# load_dotenv()
# openai_key = os.getenv("OPENAI_API_KEY")
# eleven_key = os.getenv("ELEVENLABS_API_KEY")
# client = OpenAI(api_key=openai_key)

# st.set_page_config(page_title="Multimodal AI QnA")
# st.title("🎙️ Multimodal RAG + Voice Bot")

# st.markdown("""
# Handles **Text**, **Audio**, and **PDFs** with:
# - Intent detection ✅
# - Sentiment classification ✅
# - OpenAI moderation ✅
# - Whisper & Azure STT 🔊
# """)

# # === PDF Upload Section ===
# st.header("📄 Upload PDF for Q&A")
# pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

# if pdf_file:
#     st.success("PDF uploaded. Ask a question below.")
    
#     question = st.text_input("Type your question here")

#     st.markdown("### 🎤 Or use your voice")

#     col1, col2 = st.columns(2)

#     with col1:
#         if st.button("🎙️ Record (VAD + Whisper)"):
#             st.info("Listening with VAD...")
#             audio_path = record_audio_with_vad()
#             if audio_path:
#                 question = transcribe_audio(audio_path)
#                 st.success(f"Transcribed: {question}")
#             else:
#                 st.warning("No speech detected.")

#     with col2:
#         if st.button("🎙️ Record (Azure Mic)"):
#             st.info("Listening with Azure...")
#             question = recognize_speech_azure()
#             st.success(f"Transcribed: {question}")

#     if st.button("🧠 Ask PDF"):
#         if not question.strip():
#             st.warning("Provide a question via text or voice.")
#         else:
#             flagged, reasons = moderate_text(question)
#             if flagged is None:
#                 st.error("Moderation failed.")
#             elif flagged:
#                 st.error("Blocked by moderation.")
#                 st.info(f"Reason: {', '.join(reasons)}")
#             else:
#                 try:
#                     answer = process_pdf_and_ask(pdf_file, question)
#                     st.subheader("📚 Answer from Document")
#                     st.success(answer)
#                     st.audio(text_to_speech(answer), format="audio/mp3")
#                 except Exception as e:
#                     st.error(f"Error: {e}")

# # === Free Text Analysis ===
# st.header("💬 Analyze Free Text")

# # Initialize session state for message input
# if 'message_input' not in st.session_state:
#     st.session_state.message_input = ""

# # Text area that displays and updates the session state
# user_input = st.text_area(
#     "Enter your message", 
#     value=st.session_state.message_input, 
#     height=100, 
#     key="message_input_widget"
# )

# # Add Azure microphone recording option
# if st.button("🎙️ Record Audio for Analysis (Azure)"):
#     st.info("Listening with Azure... (Speak now)")
#     transcribed_text = recognize_speech_azure()
    
#     if transcribed_text and transcribed_text != "No speech recognized.":
#         # Update the session state
#         st.session_state.message_input = transcribed_text
#         st.success("Audio transcribed successfully!")
#         # Force a rerun to update the text area
#         st.rerun()  # Changed from st.experimental_rerun() to st.rerun()
#     else:
#         st.warning("No speech detected or error occurred.")

# if st.button("🔍 Analyze Text"):
#     # Get the current text from session state
#     analysis_text = st.session_state.message_input
    
#     if not analysis_text.strip():
#         st.warning("Enter some text or record audio.")
#     else:
#         flagged, reasons = moderate_text(analysis_text)
#         if flagged is None:
#             st.error("Moderation failed.")
#         elif flagged:
#             st.error("Blocked by moderation.")
#             st.info(f"Reason: {', '.join(reasons)}")
#         else:
#             try:
#                 intent_prompt = [
#                     {"role": "system", "content": "You're an intent classifier."},
#                     {"role": "user", "content": analysis_text}
#                 ]
#                 intent_response = client.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=intent_prompt
#                 )
#                 intent_label = intent_response.choices[0].message.content.strip()

#                 sentiment_prompt = [
#                     {"role": "system", "content": "You're a sentiment classifier."},
#                     {"role": "user", "content": analysis_text}
#                 ]
#                 sentiment_response = client.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=sentiment_prompt
#                 )
#                 sentiment = sentiment_response.choices[0].message.content.strip()

#                 st.subheader("🧭 Intent:")
#                 st.info(intent_label)

#                 st.subheader("📈 Sentiment:")
#                 st.success(sentiment)

#             except Exception as e:
#                 st.error(f"Error: {e}")



# import os
# from dotenv import load_dotenv
# from openai import OpenAI
# import streamlit as st

# from rag_utils import process_pdf_and_ask
# from audio_utils import (
#     transcribe_audio, 
#     text_to_speech,
#     recognize_speech_azure
# )
# from moderation_utils import moderate_text

# # === Load Keys ===
# load_dotenv()
# openai_key = os.getenv("OPENAI_API_KEY")
# eleven_key = os.getenv("ELEVENLABS_API_KEY")
# client = OpenAI(api_key=openai_key)

# st.set_page_config(page_title="Multimodal AI QnA")
# st.title("🎙️ Multimodal RAG + Voice Bot")

# st.markdown("""
# Handles **Text**, **Audio**, and **PDFs** with:
# - Intent detection ✅
# - Sentiment classification ✅
# - OpenAI moderation ✅
# - Azure STT 🔊
# """)

# # === PDF Upload Section ===
# st.header("📄 Upload PDF for Q&A")
# pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

# if pdf_file:
#     st.success("PDF uploaded. Ask a question below.")
    
#     # Initialize session state for question input
#     if 'pdf_question' not in st.session_state:
#         st.session_state.pdf_question = ""
    
#     # Text input that displays and updates the session state
#     question_input = st.text_input(
#         "Type your question here",
#         value=st.session_state.pdf_question,
#         key="question_input_widget"
#     )

#     # Audio recording option for PDF questions
#     if st.button("🎙️ Record Question (Azure)"):
#         st.info("Listening with Azure... (Speak now)")
#         transcribed_text = recognize_speech_azure()
        
#         if transcribed_text and transcribed_text != "No speech recognized.":
#             # Update the session state
#             st.session_state.pdf_question = transcribed_text
#             st.success("Question recorded successfully!")
#             # Force a rerun to update the text input
#             st.rerun()
#         else:
#             st.warning("No speech detected or error occurred.")

#     if st.button("🧠 Ask PDF"):
#         # Get the current question from session state
#         current_question = st.session_state.pdf_question
        
#         if not current_question.strip():
#             st.warning("Provide a question via text or voice.")
#         else:
#             flagged, reasons = moderate_text(current_question)
#             if flagged is None:
#                 st.error("Moderation failed.")
#             elif flagged:
#                 st.error("Blocked by moderation.")
#                 st.info(f"Reason: {', '.join(reasons)}")
#             else:
#                 try:
#                     answer = process_pdf_and_ask(pdf_file, current_question)
#                     st.subheader("📚 Answer from Document")
#                     st.success(answer)
#                     st.audio(text_to_speech(answer), format="audio/mp3")
#                 except Exception as e:
#                     st.error(f"Error: {e}")

# # === Free Text Analysis ===
# st.header("💬 Analyze Free Text")

# # Initialize session state for message input
# if 'message_input' not in st.session_state:
#     st.session_state.message_input = ""

# # Text area that displays and updates the session state
# user_input = st.text_area(
#     "Enter your message", 
#     value=st.session_state.message_input, 
#     height=100, 
#     key="message_input_widget"
# )

# # Add Azure microphone recording option
# if st.button("🎙️ Record Audio for Analysis (Azure)"):
#     st.info("Listening with Azure... (Speak now)")
#     transcribed_text = recognize_speech_azure()
    
#     if transcribed_text and transcribed_text != "No speech recognized.":
#         # Update the session state
#         st.session_state.message_input = transcribed_text
#         st.success("Audio transcribed successfully!")
#         # Force a rerun to update the text area
#         st.rerun()
#     else:
#         st.warning("No speech detected or error occurred.")

# if st.button("🔍 Analyze Text"):
#     # Get the current text from session state
#     analysis_text = st.session_state.message_input
    
#     if not analysis_text.strip():
#         st.warning("Enter some text or record audio.")
#     else:
#         flagged, reasons = moderate_text(analysis_text)
#         if flagged is None:
#             st.error("Moderation failed.")
#         elif flagged:
#             st.error("Blocked by moderation.")
#             st.info(f"Reason: {', '.join(reasons)}")
#         else:
#             try:
#                 intent_prompt = [
#                     {"role": "system", "content": "You're an intent classifier."},
#                     {"role": "user", "content": analysis_text}
#                 ]
#                 intent_response = client.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=intent_prompt
#                 )
#                 intent_label = intent_response.choices[0].message.content.strip()

#                 sentiment_prompt = [
#                     {"role": "system", "content": "You're a sentiment classifier."},
#                     {"role": "user", "content": analysis_text}
#                 ]
#                 sentiment_response = client.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=sentiment_prompt
#                 )
#                 sentiment = sentiment_response.choices[0].message.content.strip()

#                 st.subheader("🧭 Intent:")
#                 st.info(intent_label)

#                 st.subheader("📈 Sentiment:")
#                 st.success(sentiment)

#             except Exception as e:
#                 st.error(f"Error: {e}")