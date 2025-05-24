# import streamlit as st
# import streamlit.components.v1 as components
# from openai import OpenAI
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
# import os
# from dotenv import load_dotenv
# from rag_utils import process_pdf_and_ask
# from audio_utils import transcribe_audio, text_to_speech
# from moderation_utils import moderate_text

# # Load environment variables
# load_dotenv()
# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# # Load intent classifier
# @st.cache_resource
# def load_intent_model():
#     model = AutoModelForSequenceClassification.from_pretrained("yeniguno/bert-uncased-intent-classification")
#     tokenizer = AutoTokenizer.from_pretrained("yeniguno/bert-uncased-intent-classification")
#     return pipeline("text-classification", model=model, tokenizer=tokenizer)

# intent_pipe = load_intent_model()

# # === UI ===
# st.set_page_config(page_title="Multimodal RAG + Moderation", page_icon="🤖")
# st.title("🎙️ Multimodal RAG + Intent Bot")
# st.write("Handles **text**, **audio**, and **PDFs** with intent detection, sentiment analysis, and OpenAI moderation.")

# # === PDF Upload + Ask ===
# st.header("📄 Upload PDF for Question Answering")
# pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# if pdf_file:
#     st.success("✅ PDF uploaded. Ask a question via text or record your voice.")
#     st.markdown("### ✍️ Type Your Question")
#     question = st.text_input("Ask a question based on the PDF content:")

#     st.markdown("### 🎤 Or Record Your Question")
#     components.html(open("audio_recorder.html", "r").read(), height=300)

#     uploaded_audio = st.file_uploader("Or upload recorded audio (WAV)", type=["wav"])
#     final_question = question

#     if st.button("Ask PDF"):
#         if uploaded_audio:
#             with st.spinner("🔊 Transcribing audio..."):
#                 final_question = transcribe_audio(uploaded_audio, client)
#                 st.info(f"Transcribed audio: {final_question}")

#         if not final_question.strip():
#             st.warning("⚠️ Please provide a text or audio question.")
#         else:
#             flagged, reasons = moderate_text(final_question)
#             if flagged is None:
#                 st.error("⚠️ Moderation system failed. Please try again later.")
#             elif flagged:
#                 st.error("🚫 Your query has been blocked due to safety violations.")
#                 st.info(f"🛑 Reason(s): {', '.join(reasons)}")
#             else:
#                 try:
#                     answer = process_pdf_and_ask(pdf_file, final_question)
#                     st.subheader("📚 Answer from Document:")
#                     st.success(answer)
#                     st.audio(text_to_speech(answer), format="audio/mp3")
#                 except Exception as e:
#                     st.error(f"❌ Error processing PDF: {e}")

# # === Text Input ===
# st.header("📝 Intent + Sentiment Analysis (Text Only)")
# user_input = st.text_area("Enter a message", "", height=100)

# if st.button("Analyze Text"):
#     if user_input.strip() == "":
#         st.warning("⚠️ Please enter some text to analyze.")
#     else:
#         flagged, reasons = moderate_text(user_input)

#         print("flagged user input is after moderation ai is", flagged)
#         print("moderation user input reason", reasons)

#         if flagged is None:
#             st.error("⚠️ Moderation system failed. Please try again later.")
#         elif flagged:
#             st.error("🚫 Your query has been blocked due to safety violations.")
#             st.info(f"🛑 Reason(s): {', '.join(reasons)}")
#         else:
#             try:
#                 intent_result = intent_pipe(user_input)
#                 intent_label = intent_result[0]["label"]

#                 response = client.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=[
#                         {
#                             "role": "system",
#                             "content": "You are a sentiment analysis assistant. Respond with 'Positive', 'Negative', or 'Neutral' and give a brief reason."
#                         },
#                         {"role": "user", "content": user_input}
#                     ]
#                 )
#                 sentiment = response.choices[0].message.content.strip()

#                 st.subheader("🧠 Detected Intent:")
#                 st.info(intent_label)

#                 st.subheader("💬 Predicted Sentiment:")
#                 st.success(sentiment)

#             except Exception as e:
#                 st.error(f"❌ Error during analysis: {e}")


#2...........................

import os
import tempfile
import requests
import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from rag_utils import process_pdf_and_ask
from audio_utils import transcribe_audio, text_to_speech
from moderation_utils import moderate_text
import webrtcvad

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
eleven_key = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"  # Roger voice

# Debug prints to verify keys (you can comment these out later)
print("✅ OpenAI Key Loaded:", openai_key)
print("✅ ElevenLabs Key Loaded:", eleven_key)

# Initialize OpenAI client
client = OpenAI(api_key=openai_key)

# Load intent classifier
@st.cache_resource
def load_intent_model():
    model = AutoModelForSequenceClassification.from_pretrained("yeniguno/bert-uncased-intent-classification")
    tokenizer = AutoTokenizer.from_pretrained("yeniguno/bert-uncased-intent-classification")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

intent_pipe = load_intent_model()

# === Record Audio with VAD (Voice Activity Detection) ===
def record_audio_with_vad(duration=10, aggressiveness=2):
    fs = 16000
    vad = webrtcvad.Vad(aggressiveness)
    frame_duration = 30  # ms
    frame_size = int(fs * frame_duration / 1000)

    def is_speech(frame):
        return vad.is_speech(frame, sample_rate=fs)

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()

    audio_bytes = recording.flatten().tobytes()
    frames = [audio_bytes[i:i + frame_size * 2] for i in range(0, len(audio_bytes), frame_size * 2)]
    voiced = [f for f in frames if is_speech(f)]

    if not voiced:
        return None

    audio_array = np.frombuffer(b"".join(voiced), dtype=np.int16)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        write(f.name, fs, audio_array.reshape(-1, 1))
        return f.name

# === UI ===
st.set_page_config(page_title="Multimodal AI QnA", page_icon="🤖")
st.title("🎙️ Multimodal RAG + Voice Bot")
st.markdown("""Handles **Text**, **Audio**, and **PDFs** with:
- Intent detection ✅
- Sentiment classification ✅
- OpenAI moderation ✅
- Real-time voice transcription & TTS 🔊
""")

# === PDF Upload + Ask ===
st.header("📄 Upload PDF for Q&A")
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file:
    st.success("✅ PDF uploaded. Ask a question below.")
    
    st.markdown("###  Type Your Question")
    question = st.text_input("Enter a question based on the PDF")

    st.markdown("### 🎤 Or Use Microphone (with VAD)")

    if st.button("🎙️ Record Audio"):
        st.info("Listening... Speak now.")
        audio_path = record_audio_with_vad()
        if audio_path:
            transcribed = transcribe_audio(audio_path)
            st.success(f"📝 Transcribed: {transcribed}")
            question = transcribed
        else:
            st.warning("❌ No speech detected.")

    uploaded_audio = st.file_uploader("Or upload your own audio (WAV)", type=["wav"])
    if uploaded_audio:
        with st.spinner("🔊 Transcribing uploaded audio..."):
            question = transcribe_audio(uploaded_audio)
            st.success(f"📝 Transcribed: {question}")

    if st.button("🧠 Ask PDF"):
        if not question.strip():
            st.warning("⚠️ Provide a question via text or audio.")
        else:
            flagged, reasons = moderate_text(question)
            if flagged is None:
                st.error("⚠️ Moderation system failed.")
            elif flagged:
                st.error("🚫 Blocked by moderation.")
                st.info(f"Reason: {', '.join(reasons)}")
            else:
                try:
                    answer = process_pdf_and_ask(pdf_file, question)
                    st.subheader("📚 Answer from Document:")
                    st.success(answer)
                    st.audio(text_to_speech(answer), format="audio/mp3")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# === Text + Intent + Sentiment ===
st.header("📝 Analyze Free Text")
user_input = st.text_area("Enter your message", "", height=100)

if st.button("Analyze Text"):
    if not user_input.strip():
        st.warning("⚠️ Please enter text.")
    else:
        flagged, reasons = moderate_text(user_input)
        if flagged is None:
            st.error("⚠️ Moderation failed.")
        elif flagged:
            st.error("🚫 Blocked by moderation.")
            st.info(f"Reason: {', '.join(reasons)}")
        else:
            try:
                intent_result = intent_pipe(user_input)
                intent_label = intent_result[0]["label"]

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": "You're a sentiment classifier. Respond with 'Positive', 'Negative', or 'Neutral' and explain briefly."},
                              {"role": "user", "content": user_input}]
                )
                sentiment = response.choices[0].message.content.strip()

                st.subheader("🧠 Detected Intent:")
                st.info(intent_label)

                st.subheader("💬 Predicted Sentiment:")
                st.success(sentiment)

            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")


#-------------------------------------------updated---------------------------

# import streamlit as st
# import streamlit.components.v1 as components
# from openai import OpenAI
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
# import os
# import tempfile
# import base64
# from rag_utils import process_pdf_and_ask
# from audio_utils import transcribe_audio, text_to_speech
# from langchain_community.document_loaders import PyPDFLoader


# # 🟰 ADD THESE LINES TO HANDLE LOCAL .env
# from dotenv import load_dotenv
# load_dotenv()

# #loading api key from dotenv
# client = client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# # Initialize OpenAI client with Streamlit secrets
# # openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# # Load BERT intent classification model
# @st.cache_resource
# def load_intent_model():
#     model = AutoModelForSequenceClassification.from_pretrained("yeniguno/bert-uncased-intent-classification")
#     tokenizer = AutoTokenizer.from_pretrained("yeniguno/bert-uncased-intent-classification")
#     return pipeline("text-classification", model=model, tokenizer=tokenizer)

# intent_pipe = load_intent_model()

# # === UI ===
# st.title("🎙️ Multimodal RAG + Intent Bot")
# st.write("Handles **text**, **audio**, and **PDFs** for intent, sentiment, and document Q&A.")

# # === PDF Upload + Ask ===
# st.header("📄 Upload PDF for Question Answering")
# pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# if pdf_file:
#     st.success("✅ PDF uploaded. Ask a question via text or record your voice.")
#     st.markdown("### ✍️ Type Your Question")
#     question = st.text_input("Ask a question based on the PDF content:")

#     st.markdown("### 🎤 Or Record Your Question")
#     components.html(open("audio_recorder.html", "r").read(), height=300)

#     uploaded_audio = st.file_uploader("Or upload recorded audio (WAV)", type=["wav"])

#     final_question = question

#     if st.button("Ask PDF"):
#         if uploaded_audio:
#             with st.spinner("🔊 Transcribing audio..."):
#                 final_question = transcribe_audio(uploaded_audio, openai_client)
#                 st.info(f"Transcribed audio: {final_question}")
#         if not final_question.strip():
#             st.warning("⚠️ Please provide a text or audio question.")
#         else:
#             try:
#                 answer = process_pdf_and_ask(pdf_file, final_question)  # Updated to match the function signature
#                 st.subheader("📚 Answer from Document:")
#                 st.success(answer)
#                 st.audio(text_to_speech(answer), format="audio/mp3")
#             except Exception as e:
#                 st.error(f"❌ Error processing PDF: {e}")

# # === Text Input ===
# st.header("📝 Intent + Sentiment Analysis (Text Only)")
# user_input = st.text_area("Enter a message", "", height=100)

# if st.button("Analyze Text"):
#     if user_input.strip() == "":
#         st.warning("⚠️ Please enter some text to analyze.")
#     else:
#         try:
#             intent_result = intent_pipe(user_input)
#             intent_label = intent_result[0]["label"]

#             response = openai_client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[ 
#                     {"role": "system", "content": "You are a sentiment analysis assistant. Respond with 'Positive', 'Negative', or 'Neutral' and give a brief reason."},
#                     {"role": "user", "content": user_input}
#                 ]
#             )
#             sentiment = response.choices[0].message.content.strip()

#             st.subheader("🧠 Detected Intent:")
#             st.info(intent_label)

#             st.subheader("💬 Predicted Sentiment:")
#             st.success(sentiment)

#         except Exception as e:
#             st.error(f"❌ Error: {e}")
