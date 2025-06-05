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

import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
from rag_utils import process_pdf_and_ask, process_text_with_llm
from audio_utils import recognize_speech_azure, text_to_speech
from moderation_utils import moderate_text

# Initialize
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Session state
if 'audio_data' not in st.session_state:
    st.session_state.update({
        'audio_data': None,
        'transcribed_text': "",
        'processing': False,
        'pdf_file': None,
        'is_recording': False
    })

# UI Config
st.set_page_config(page_title="Voice RAG Bot", layout="wide")
st.title("🎙️ Smart Voice Assistant")

def analyze_text(text):
    """Enhanced text analysis with single API call"""
    flagged, reasons = moderate_text(text)
    if flagged:
        st.error(f"Content blocked: {reasons}")
        return None, None
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Analyze this text. Respond with:
                Intent: <detected intent>
                Sentiment: <positive/neutral/negative>"""},
                {"role": "user", "content": text}
            ]
        )
        analysis = response.choices[0].message.content
        return analysis.split("Intent: ")[1].split("\n")[0], analysis.split("Sentiment: ")[1]
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None, None

def process_query():
    """Handle the full processing pipeline"""
    if not st.session_state.transcribed_text:
        return
    
    st.session_state.processing = True
    
    # Analysis
    with st.expander("💬 Analysis"):
        intent, sentiment = analyze_text(st.session_state.transcribed_text)
        if intent and sentiment:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🧭 Intent")
                st.info(intent)
            with col2:
                st.subheader("📈 Sentiment")
                st.success(sentiment)
    
    # Response Generation
    with st.spinner("Generating response..."):
        try:
            if st.session_state.pdf_file:
                answer = process_pdf_and_ask(st.session_state.pdf_file, st.session_state.transcribed_text)
            else:
                answer = process_text_with_llm(st.session_state.transcribed_text)
            
            st.subheader("🤖 Response")
            st.write(answer)
            
            # Audio Response
            audio = text_to_speech(answer)
            if audio:
                st.audio(audio, format="audio/mp3")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            st.session_state.processing = False

# Main UI
st.header("🎤 Voice Input")

if st.button("🎙️ Start Recording") and not st.session_state.processing:
    st.session_state.is_recording = True
    st.session_state.transcribed_text = recognize_speech_azure()
    st.rerun()

if st.session_state.transcribed_text:
    st.text_area("Transcription", 
                value=st.session_state.transcribed_text, 
                height=100,
                disabled=True)
    process_query()

# PDF Handling
st.header("📄 Document Context")
pdf_file = st.file_uploader("Upload PDF (optional)", type=["pdf"])
if pdf_file:
    st.session_state.pdf_file = pdf_file
    st.success("PDF loaded for context!")
elif st.session_state.pdf_file:
    if st.button("Remove PDF"):
        st.session_state.pdf_file = None
        st.rerun()