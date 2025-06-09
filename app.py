# #In this below code both audio part and text part is working
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



# # === PDF Upload Section ===
# st.header(" Upload PDF for Q&A")
# pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

# if pdf_file:
#     st.success("PDF uploaded. Ask a question below.")
    
#     # Initialize session state for question input
#     if 'pdf_question' not in st.session_state:
#         st.session_state.pdf_question = ""
    
#     # Text input that syncs with session state
#     question_input = st.text_input(
#         "Type your question here",
#         value=st.session_state.pdf_question,
#         key="question_input_widget"
#     )
    
#     # Update session state when text input changes
#     if question_input != st.session_state.pdf_question:
#         st.session_state.pdf_question = question_input

#     # Audio recording option for PDF questions
#     if st.button("🎙️ Record Question (Azure)"):
#         st.info("Listening with Azure... (Speak now)")
#         transcribed_text = recognize_speech_azure()
        
#         if transcribed_text and transcribed_text != "No speech recognized.":
#             st.session_state.pdf_question = transcribed_text
#             st.success("Question recorded successfully!")
#             st.rerun()
#         else:
#             st.warning("No speech detected or error occurred.")

#     if st.button(" Ask PDF"):
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
#                     st.subheader(" Answer from Document")
#                     st.success(answer)
#                     st.audio(text_to_speech(answer), format="audio/mp3")
#                 except Exception as e:
#                     st.error(f"Error: {e}")

# # === Free Text Analysis ===
# st.header("💬 Analyze Free Text")

# # Initialize session state for message input
# if 'message_input' not in st.session_state:
#     st.session_state.message_input = ""

# # Text area that syncs with session state
# user_input = st.text_area(
#     "Enter your message", 
#     value=st.session_state.message_input, 
#     height=100, 
#     key="message_input_widget"
# )

# # Update session state when text area changes
# if user_input != st.session_state.message_input:
#     st.session_state.message_input = user_input

# # Add Azure microphone recording option
# if st.button(" Record Audio for Analysis (Azure)"):
#     st.info("Listening with Azure... (Speak now)")
#     transcribed_text = recognize_speech_azure()
    
#     if transcribed_text and transcribed_text != "No speech recognized.":
#         st.session_state.message_input = transcribed_text
#         st.success("Audio transcribed successfully!")
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
                
                
                
                

#-------------------------------------------updated---------------------------
# 2. updation 
# in this code below only after giving the query as audio automatically sentiment and intent function is runnign 
# but rag part is not running we basically remove the anlyze butoon 

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
# st.title("🎙 Multimodal RAG + Voice Bot")



# # === PDF Upload Section ===
# st.header(" Upload PDF for Q&A")
# pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

# if pdf_file:
#     st.success("PDF uploaded. Ask a question below.")
    
#     # Initialize session state for question input
#     if 'pdf_question' not in st.session_state:
#         st.session_state.pdf_question = ""
    
#     # Text input that syncs with session state
#     question_input = st.text_input(
#         "Type your question here",
#         value=st.session_state.pdf_question,
#         key="question_input_widget"
#     )
    
#     # Update session state when text input changes
#     if question_input != st.session_state.pdf_question:
#         st.session_state.pdf_question = question_input

#     # Audio recording option for PDF questions
#     if st.button("🎙 Record Question (Azure)"):
#         st.info("Listening with Azure... (Speak now)")
#         transcribed_text = recognize_speech_azure()
        
#         if transcribed_text and transcribed_text != "No speech recognized.":
#             st.session_state.pdf_question = transcribed_text
#             st.success("Question recorded successfully!")
#             st.rerun()
#         else:
#             st.warning("No speech detected or error occurred.")

#     if st.button(" Ask PDF"):
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
#                     st.subheader(" Answer from Document")
#                     st.success(answer)
#                     st.audio(text_to_speech(answer), format="audio/mp3")
#                 except Exception as e:
#                     st.error(f"Error: {e}")

# # === Free Text Analysis ===
# st.header("💬 Analyze Free Text")

# # Initialize session state for message input
# if 'message_input' not in st.session_state:
#     st.session_state.message_input = ""

# # Text area that syncs with session state
# user_input = st.text_area(
#     "Enter your message", 
#     value=st.session_state.message_input, 
#     height=100, 
#     key="message_input_widget"
# )

# # Update session state when text area changes
# if user_input != st.session_state.message_input:
#     st.session_state.message_input = user_input

# # Add Azure microphone recording option
# if st.button(" Record Audio for Analysis (Azure)"):
#     st.info("Listening with Azure... (Speak now)")
#     transcribed_text = recognize_speech_azure()
    
#     if transcribed_text and transcribed_text != "No speech recognized.":
#         st.session_state.message_input = transcribed_text
#         st.success("Audio transcribed successfully!")
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


#---------------------------------------------3. --------------------------------------
#current updated code in github


# import os
# from dotenv import load_dotenv
# from openai import OpenAI
# import streamlit as st

# from rag_utils import process_pdf_and_ask, process_text_with_llm  # Updated rag_utils function
# from audio_utils import recognize_speech_azure, text_to_speech
# from moderation_utils import moderate_text

# # === Load Keys ===
# load_dotenv()
# openai_key = os.getenv("OPENAI_API_KEY")
# eleven_key = os.getenv("ELEVENLABS_API_KEY")
# client = OpenAI(api_key=openai_key)

# st.set_page_config(page_title="Multimodal AI QnA")
# st.title("🎙️ Multimodal RAG  Voice Bot")

# def analyze_text(text):
#     """Function to analyze text for intent and sentiment"""
#     flagged, reasons = moderate_text(text)
#     if flagged is None:
#         st.error("Moderation failed.")
#         return None, None
#     elif flagged:
#         st.error("Blocked by moderation.")
#         st.info(f"Reason: {', '.join(reasons)}")
#         return None, None
    
#     try:
#         # Intent analysis
#         intent_prompt = [
#             {"role": "system", "content": "You're an intent classifier."},
#             {"role": "user", "content": text}
#         ]
#         intent_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=intent_prompt
#         )
#         intent_label = intent_response.choices[0].message.content.strip()

#         # Sentiment analysis
#         sentiment_prompt = [
#             {"role": "system", "content": "You're a sentiment classifier."},
#             {"role": "user", "content": text}
#         ]
#         sentiment_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=sentiment_prompt
#         )
#         sentiment = sentiment_response.choices[0].message.content.strip()

#         return intent_label, sentiment

#     except Exception as e:
#         st.error(f"Analysis Error: {e}")
#         return None, None

# def process_user_input(text):
#     """Process user input through all required pipelines"""
#     if not text.strip():
#         return
    
#     # 1. Always analyze sentiment and intent
#     st.subheader("💬 Message Analysis")
#     intent, sentiment = analyze_text(text)
    
#     if intent is not None and sentiment is not None:
#         st.subheader("🧭 Intent:")
#         st.info(intent)
#         st.subheader("📈 Sentiment:")
#         st.success(sentiment)
    
#     # 2. Process with RAG/LLM (works with or without PDF)
#     st.subheader("🤖 AI Response")
#     try:
#         with st.spinner("Generating response..."):
#             if 'pdf_file' in st.session_state and st.session_state.pdf_file is not None:
#                 # Process with RAG if PDF exists
#                 answer = process_pdf_and_ask(st.session_state.pdf_file, text)
#             else:
#                 # Process with basic LLM if no PDF
#                 answer = process_text_with_llm(text)
            
#             st.success(answer)
#             st.audio(text_to_speech(answer), format="audio/mp3")
#     except Exception as e:
#         st.error(f"Response Error: {e}")

# # === Main Audio Input Section ===
# st.header("🎤 Voice Input")

# # Initialize session state
# if 'user_input' not in st.session_state:
#     st.session_state.user_input = ""
#     st.session_state.new_audio = False

# # Display area for transcribed text
# st.text_area("Transcribed Text", 
#             value=st.session_state.user_input, 
#             height=100, 
#             key="transcribed_display",
#             disabled=True)

# # Single audio recording button
# if st.button("🎙️ Start Recording (Azure)"):
#     st.info("Listening with Azure... (Speak now)")
#     transcribed_text = recognize_speech_azure()
    
#     if transcribed_text and transcribed_text != "No speech recognized.":
#         st.session_state.user_input = transcribed_text
#         st.session_state.new_audio = True
#         st.rerun()
#     else:
#         st.warning("No speech detected or error occurred.")

# # === PDF Upload Section ===
# st.header("📄 PDF Upload (Optional)")
# pdf_file = st.file_uploader("Upload a PDF document", type=["pdf"], key="pdf_uploader")
# if pdf_file:
#     st.session_state.pdf_file = pdf_file
#     st.success("PDF loaded! Questions will use document context.")
# elif 'pdf_file' in st.session_state:
#     del st.session_state.pdf_file

# # Process audio input when received
# if st.session_state.new_audio:
#     st.session_state.new_audio = False
#     process_user_input(st.session_state.user_input)


#4th updation 


# import os
# import tempfile
# from dotenv import load_dotenv
# from openai import OpenAI
# import streamlit as st
# from streamlit_audio_recorder import audio_recorder

# from rag_utils import process_pdf_and_ask, process_text_with_llm
# from audio_utils import recognize_speech_azure, text_to_speech
# from moderation_utils import moderate_text

# # === Load Keys ===
# load_dotenv()
# openai_key = os.getenv("OPENAI_API_KEY")
# eleven_key = os.getenv("ELEVENLABS_API_KEY")
# client = OpenAI(api_key=openai_key)

# st.set_page_config(page_title="Multimodal AI QnA")
# st.title("🎙️ Multimodal RAG Voice Bot")


# def analyze_text(text):
#     """Function to analyze text for intent and sentiment"""
#     flagged, reasons = moderate_text(text)
#     if flagged is None:
#         st.error("Moderation failed.")
#         return None, None
#     elif flagged:
#         st.error("Blocked by moderation.")
#         st.info(f"Reason: {', '.join(reasons)}")
#         return None, None

#     try:
#         # Intent analysis
#         intent_prompt = [
#             {"role": "system", "content": "You're an intent classifier."},
#             {"role": "user", "content": text}
#         ]
#         intent_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=intent_prompt
#         )
#         intent_label = intent_response.choices[0].message.content.strip()

#         # Sentiment analysis
#         sentiment_prompt = [
#             {"role": "system", "content": "You're a sentiment classifier."},
#             {"role": "user", "content": text}
#         ]
#         sentiment_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=sentiment_prompt
#         )
#         sentiment = sentiment_response.choices[0].message.content.strip()

#         return intent_label, sentiment

#     except Exception as e:
#         st.error(f"Analysis Error: {e}")
#         return None, None


# def process_user_input(text):
#     """Process user input through all required pipelines"""
#     if not text.strip():
#         return

#     # 1. Always analyze sentiment and intent
#     st.subheader("💬 Message Analysis")
#     intent, sentiment = analyze_text(text)

#     if intent is not None and sentiment is not None:
#         st.subheader("🧭 Intent:")
#         st.info(intent)
#         st.subheader("📈 Sentiment:")
#         st.success(sentiment)

#     # 2. Process with RAG/LLM (works with or without PDF)
#     st.subheader("🤖 AI Response")
#     try:
#         with st.spinner("Generating response..."):
#             if 'pdf_file' in st.session_state and st.session_state.pdf_file is not None:
#                 answer = process_pdf_and_ask(st.session_state.pdf_file, text)
#             else:
#                 answer = process_text_with_llm(text)

#             st.success(answer)
#             st.audio(text_to_speech(answer), format="audio/mp3")
#     except Exception as e:
#         st.error(f"Response Error: {e}")


# # === Main Audio Input Section ===
# st.header("🎤 Voice Input")

# # Initialize session state
# if 'user_input' not in st.session_state:
#     st.session_state.user_input = ""
#     st.session_state.new_audio = False

# # Audio Recorder from browser
# audio_bytes = audio_recorder(
#     text="🎙️ Start Recording (Browser Mic)",
#     recording_color="#e8b62c",
#     neutral_color="#6aa36f",
#     icon_size="2x"
# )

# # Save and process audio if recorded
# if audio_bytes:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#         f.write(audio_bytes)
#         audio_path = f.name
#         st.success("Recording saved! Transcribing...")
#         transcribed_text = recognize_speech_azure(audio_path)

#         if transcribed_text and transcribed_text != "No speech recognized.":
#             st.session_state.user_input = transcribed_text
#             st.session_state.new_audio = True
#             st.rerun()
#         else:
#             st.warning("No speech detected or error occurred.")

# # Display area for transcribed text
# st.text_area("Transcribed Text",
#              value=st.session_state.user_input,
#              height=100,
#              key="transcribed_display",
#              disabled=True)

# # === PDF Upload Section ===
# st.header("📄 PDF Upload (Optional)")
# pdf_file = st.file_uploader("Upload a PDF document", type=["pdf"], key="pdf_uploader")
# if pdf_file:
#     st.session_state.pdf_file = pdf_file
#     st.success("PDF loaded! Questions will use document context.")
# elif 'pdf_file' in st.session_state:
#     del st.session_state.pdf_file

# # Process audio input when received
# if st.session_state.new_audio:
#     st.session_state.new_audio = False
#     process_user_input(st.session_state.user_input)


#5ht update

# import os
# import streamlit as st
# from dotenv import load_dotenv
# from openai import OpenAI
# from audio_utils import recognize_from_mic
# from moderation_utils import moderate_text  # Your moderation function
# import base64
# import requests

# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # ElevenLabs TTS function
# def text_to_speech_elevenlabs(text: str):
#     try:
#         xi_api_key = os.getenv("ELEVENLABS_API_KEY")
#         voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

#         url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

#         headers = {
#             "Accept": "audio/mpeg",
#             "Content-Type": "application/json",
#             "xi-api-key": xi_api_key
#         }

#         data = {
#             "text": text,
#             "model_id": "eleven_monolingual_v2",
#             "voice_settings": {
#                 "stability": 0.5,
#                 "similarity_boost": 0.5
#             }
#         }

#         response = requests.post(url, json=data, headers=headers)

#         if response.status_code == 200:
#             audio_bytes = response.content
#             audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
#             return audio_base64
#         else:
#             st.error(f"ElevenLabs TTS failed: {response.text}")
#             return None

#     except Exception as e:
#         st.error(f"TTS Error: {str(e)}")
#         return None

# def analyze_sentiment_intent(text: str):
#     try:
#         sentiment_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "Classify sentiment as Positive/Negative/Neutral only"},
#                 {"role": "user", "content": f"Text: {text}"}
#             ]
#         )
#         sentiment = sentiment_response.choices[0].message.content.strip()

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

# def display_and_process_results():
#     st.subheader("Transcribed Text")
#     st.text_area("", value=st.session_state.user_text, height=100, disabled=True)

#     if 'sentiment' not in st.session_state:
#         with st.spinner("Analyzing sentiment and intent..."):
#             sentiment, intent = analyze_sentiment_intent(st.session_state.user_text)
#             st.session_state.sentiment = sentiment
#             st.session_state.intent = intent

#     if 'sentiment' in st.session_state:
#         col1, col2 = st.columns(2)
#         with col1:
#             st.subheader("🧠 Intent")
#             st.info(st.session_state.intent)
#         with col2:
#             st.subheader("📊 Sentiment")
#             st.success(st.session_state.sentiment)

#     if 'ai_response' not in st.session_state:
#         with st.spinner("Generating AI response..."):
#             if 'pdf_path' in st.session_state:
#                 from rag_utils import process_pdf_and_ask
#                 with open(st.session_state.pdf_path, "rb") as f:
#                     response = process_pdf_and_ask(f, st.session_state.user_text)
#             else:
#                 from rag_utils import process_text_with_llm
#                 response = process_text_with_llm(st.session_state.user_text)

#             st.session_state.ai_response = response

#             with st.spinner("Generating audio..."):
#                 audio_base64 = text_to_speech_elevenlabs(response)
#                 if audio_base64:
#                     st.session_state.audio_available = True
#                     st.session_state.audio_base64 = audio_base64
#                 else:
#                     st.session_state.audio_available = False

#             # Finished processing, hide listening
#             st.session_state.listening = False
#             st.experimental_rerun()

#     if 'ai_response' in st.session_state:
#         st.subheader("🤖 AI Response")
#         st.write(st.session_state.ai_response)

#         if st.session_state.get('audio_available', False):
#             st.markdown(f"""
#                 <audio controls autoplay>
#                     <source src="data:audio/mpeg;base64,{st.session_state.audio_base64}" type="audio/mpeg">
#                     Your browser does not support the audio element.
#                 </audio>
#                 """, unsafe_allow_html=True)
#         else:
#             st.warning("Audio playback not available for this response")

# def page_setup():
#     st.set_page_config(page_title="Voice Analysis Bot", layout="centered")
#     st.header("🎤 Voice Assistant with Sentiment+Intent+RAG")
#     st.markdown("""
#         <style>
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#         .stButton button {
#             background-color: #4CAF50;
#             color: white;
#             padding: 10px 24px;
#             border: none;
#             border-radius: 4px;
#             cursor: pointer;
#             font-size: 16px;
#         }
#         .stButton button:hover {
#             background-color: #45a049;
#         }
#         audio {
#             width: 100%;
#             margin-top: 20px;
#         }
#         </style>
#     """, unsafe_allow_html=True)

# def main():
#     page_setup()

#     # Initialize session state variables
#     if 'processing' not in st.session_state:
#         st.session_state.processing = False
#     if 'listening' not in st.session_state:
#         st.session_state.listening = False

#     # PDF Upload
#     uploaded_pdf = st.file_uploader("Upload PDF (Optional)", type=["pdf"])
#     if uploaded_pdf:
#         with open("temp.pdf", "wb") as f:
#             f.write(uploaded_pdf.read())
#         st.session_state.pdf_path = "temp.pdf"
#         st.success("PDF loaded for RAG analysis")

#     # Start Recording button
#     if st.button("🎤 Start Recording", disabled=st.session_state.processing):
#         # Reset states for new recording
#         for key in ['user_text', 'sentiment', 'intent', 'ai_response', 'audio_available', 'audio_base64']:
#             if key in st.session_state:
#                 del st.session_state[key]

#         st.session_state.processing = True
#         st.session_state.listening = True

#         # Run recognition from mic
#         user_text = recognize_from_mic()

#         st.session_state.listening = False  # Hide "Listening..." after done

#         if not user_text:
#             st.warning("No speech detected. Please try again.")
#             st.session_state.processing = False
#             return

#         st.session_state.user_text = user_text

#         # Moderation check
#         flagged, reasons = moderate_text(user_text)
#         if flagged:
#             st.error(f"⚠️ Content blocked due to: {', '.join(reasons)}")
#             st.session_state.processing = False
#             return  # Stop further processing

#         # Proceed with analysis and response generation
#         st.session_state.processing = False
#         st.experimental_rerun()

#     # Show "Listening..." only when listening
#     if st.session_state.listening:
#         st.info("Listening... Please speak now.")

#     # If text available, process and display results
#     if 'user_text' in st.session_state and not st.session_state.listening:
#         display_and_process_results()

# if __name__ == "__main__":
#     main()



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


from azure.cognitiveservices.speech import SpeechConfig, AudioConfig, SpeechRecognizer
import azure.cognitiveservices.speech as speechsdk

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ElevenLabs TTS
def text_to_speech_elevenlabs(text: str):
    try:
        xi_api_key = os.getenv("ELEVENLABS_API_KEY")
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": xi_api_key
        }

        data = {
            "text": text,
            "model_id": "eleven_monolingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            audio_bytes = response.content
            return base64.b64encode(audio_bytes).decode("utf-8")
        else:
            st.error(f"ElevenLabs TTS failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

# Transcribe audio using Azure
def transcribe_audio(file_path):
    try:
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        service_region = os.getenv("AZURE_SPEECH_REGION")
        speech_config = SpeechConfig(subscription=speech_key, region=service_region)
        audio_config = AudioConfig(filename=file_path)
        recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        result = recognizer.recognize_once()
        return result.text
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def analyze_sentiment_intent(text):
    try:
        sentiment_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Classify sentiment as Positive/Negative/Neutral only"},
                {"role": "user", "content": f"Text: {text}"}
            ]
        )
        sentiment = sentiment_response.choices[0].message.content.strip()

        intent_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Classify intent in 2-3 words (e.g., 'Question', 'Request', 'Complaint')"},
                {"role": "user", "content": f"Text: {text}"}
            ]
        )
        intent = intent_response.choices[0].message.content.strip()

        return sentiment, intent
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return None, None

def display_results():
    st.subheader("📝 Transcribed Text")
    st.text_area("", value=st.session_state.user_text, height=100, disabled=True)

    if 'sentiment' not in st.session_state:
        with st.spinner("Analyzing sentiment and intent..."):
            sentiment, intent = analyze_sentiment_intent(st.session_state.user_text)
            st.session_state.sentiment = sentiment
            st.session_state.intent = intent

    if 'sentiment' in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🧠 Intent")
            st.info(st.session_state.intent)
        with col2:
            st.subheader("📊 Sentiment")
            st.success(st.session_state.sentiment)

    if 'ai_response' not in st.session_state:
        with st.spinner("Generating AI response..."):
            if 'pdf_path' in st.session_state:
                with open(st.session_state.pdf_path, "rb") as f:
                    response = process_pdf_and_ask(f, st.session_state.user_text)
            else:
                response = process_text_with_llm(st.session_state.user_text)

            st.session_state.ai_response = response

            audio_base64 = text_to_speech_elevenlabs(response)
            if audio_base64:
                st.session_state.audio_base64 = audio_base64
                st.session_state.audio_available = True
            else:
                st.session_state.audio_available = False

            st.experimental_rerun()

    if 'ai_response' in st.session_state:
        st.subheader("🤖 AI Response")
        st.write(st.session_state.ai_response)

        if st.session_state.get("audio_available", False):
            st.markdown(f"""
                <audio controls autoplay>
                    <source src="data:audio/mpeg;base64,{st.session_state.audio_base64}" type="audio/mpeg">
                    Your browser does not support the audio element.
                </audio>
                """, unsafe_allow_html=True)

def page_setup():
    st.set_page_config(page_title="🎙 Voice Bot", layout="centered")
    st.header("🎤 Voice Assistant with Sentiment + Intent + RAG")

def main():
    page_setup()

    # Handle uploaded PDF (for optional RAG)
    uploaded_pdf = st.file_uploader("Upload PDF (optional for context)", type=["pdf"])
    if uploaded_pdf:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_pdf.read())
        st.session_state.pdf_path = "temp.pdf"
        st.success("📄 PDF loaded for RAG.")

    # Audio Recorder (browser-based)
    audio_bytes = audio_recorder(text="🎙 Start Recording")

    if audio_bytes:
        # Save audio to file safely
        with open("temp.wav", "wb") as f:
            f.write(audio_bytes)

        time.sleep(0.5)  # Ensure file is released by OS
        transcribed_text = transcribe_audio("temp.wav")

        if not transcribed_text:
            st.warning("⚠️ No speech detected or transcription failed.")
            return

        st.session_state.user_text = transcribed_text

        # Moderation check
        flagged, reasons = moderate_text(transcribed_text)
        if flagged:
            st.error(f"⚠️ Blocked due to: {', '.join(reasons)}")
            return

        display_results()


if __name__ == "__main__":
    main()
