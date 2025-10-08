
"""
📁 app.py

🎯 Updated Purpose:
Voice assistant with LPU course integration, combining:
- Voice interaction (STT/TTS)
- Sentiment/intent analysis
- LPU course information from web scraping
- Conversational memory
- RAG functionality
- Langfuse tracing
"""

import os
import time
import streamlit as st
from openai import OpenAI
# from moderation_utils import moderate_text
from rag_utils import process_text_with_llm, process_pdf_and_ask, generate_lpu_response,generate_general_response
from audio_recorder_streamlit import audio_recorder
import base64
import tempfile
from dotenv import load_dotenv
from utils.sentiment_utils import analyze_sentiment_intent
from utils.audio_utils import text_to_speech_elevenlabs, transcribe_audio
from conversational_memory import get_conversation_history
# from utils.scraper import scrape_lpu_courses
from datetime import datetime
from utils.api_monitor import monitor
from langfuse import get_client, observe
# from langfuse.decorators import langfuse_context

# ----------------------------------------------------------
# ENVIRONMENT SETUP
# ----------------------------------------------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Langfuse ===
LANGFUSE_SECRET_KEY = "sk-lf-d5a4ee4d-c355-451d-aaae-5e61b4a8e84d"
LANGFUSE_PUBLIC_KEY = "pk-lf-ab02e5cf-d477-42d0-b355-1599838ab141"
LANGFUSE_HOST = "https://us.cloud.langfuse.com"

# set env vars programmatically (so you don't need to export in shell)
os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST

# initialize client
langfuse = get_client()

# ----------------------------------------------------------
# LPU COURSE HANDLING
# ----------------------------------------------------------

class CourseDatabase:
    def __init__(self):
        self.courses = []
        self.last_updated = None
    
    def has_courses(self):
        return len(self.courses) > 0
    
    def update_courses(self):
        """Scrape and update course data (runs weekly)"""
        if not self.has_courses() or (datetime.now().weekday() == 0 and 
                                    (self.last_updated is None or 
                                     (datetime.now() - self.last_updated).days >= 7)):
            st.session_state.course_status = "Loading courses from LPU website..."
            # new_courses = scrape_lpu_courses()
            # if new_courses:
            #     self.courses = new_courses
            #     self.last_updated = datetime.now()
            #     st.session_state.course_status = "Course data updated successfully"
            # else:
            #     st.session_state.course_status = "Warning: Scraping failed - using existing data"
    
    def search_courses(self, query, n_results=3):
        """Simple keyword-based course search"""
        query = query.lower()
        return [course for course in self.courses if query in course.lower()][:n_results]

# Initialize course database
if "course_db" not in st.session_state:
    st.session_state.course_db = CourseDatabase()
    st.session_state.course_db.update_courses()

# ----------------------------------------------------------
# UI & MAIN APPLICATION
# ----------------------------------------------------------

def page_setup():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="Voice Assistant",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    st.header("Voice AI Agent")

@observe
def display_results():
    """Display transcribed text, analysis results, and AI response"""
    current_input = st.session_state.get("user_text", "")
    if not current_input or str(current_input).strip() == "":
        return

    # Display transcribed text
    st.subheader(" Transcription ")
    st.text_area(".", 
                value=current_input, 
                height=100, 
                disabled=True,
                key=f"transcribed_{int(time.time())}")

    # Analyze sentiment and intent
    if not st.session_state.get("sentiment") or not st.session_state.get("intent"):
        with st.spinner("Analyzing sentiment and intent..."):
            sentiment, intent = analyze_sentiment_intent(current_input)
            st.session_state.sentiment = sentiment
            st.session_state.intent = intent

    # Display analysis
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" Intent")
        st.info(st.session_state.get("intent", "Pending..."))
    with col2:
        st.subheader(" Sentiment")
        st.success(st.session_state.get("sentiment", "Pending..."))

    # Generate response
    response = None
    with st.spinner("Generating response..."):
        if st.session_state.get("pdf_path"):
            with open(st.session_state.pdf_path, "rb") as f:
                response = process_pdf_and_ask(f, current_input)
        else:
            # Check if query is LPU-related
            if any(kw in current_input.lower() for kw in ["lpu", "lovely", "university", "admission", "course"]):
                response = generate_lpu_response(current_input, st.session_state.course_db)
            else:
                response = generate_general_response(current_input)

        if response and str(response).strip() != "":
            st.session_state.ai_response = response
            print(f" New Response Generated: {response[:100]}...")

            # Generate audio
            audio_base64 = text_to_speech_elevenlabs(response)
            if audio_base64:
                st.session_state.audio_base64 = audio_base64
                st.session_state.audio_available = True
                print(" New audio generated")
            else:
                st.session_state.audio_available = False

            # Update conversation history
            st.session_state.conversation_history.append({
                "user": current_input,
                "assistant": response
            })

    # Display response
    if st.session_state.get("ai_response"):
        st.subheader("AI Agent Response")
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
            
def display_latency_metrics():
    """Display latency metrics in the UI"""
    st.sidebar.subheader("Performance Metrics")
    
    metrics = monitor.get_metrics()
    
    # Individual service metrics
    cols = st.sidebar.columns(2)
    with cols[0]:
        st.metric("Speech Recognition (STT)", 
                 f"{metrics['services']['STT']['last_latency']:.0f} ms",
                 help="Time taken to convert speech to text")
    
    with cols[1]:
        st.metric("Speech Synthesis (TTS)", 
                 f"{metrics['services']['TTS']['last_latency']:.0f} ms",
                 help="Time taken to convert text to speech")
    
    cols = st.sidebar.columns(2)
    with cols[0]:
        st.metric("LLM Processing", 
                 f"{metrics['services']['LLM']['last_latency']:.0f} ms",
                 help="Time taken for language model processing")
    
    with cols[1]:
        st.metric("Total Latency", 
                 f"{metrics['total_latency']:.0f} ms",
                 help="Total time for all operations")
    
    # Show historical data in expander
    with st.sidebar.expander("Detailed Metrics"):
        st.write("**Service Averages:**")
        for service, data in metrics['services'].items():
            if data['count'] > 0:
                avg = data['total_time'] / data['count']
                st.write(f"- {service}: {avg:.1f} ms (avg over {data['count']} calls)")
        
        st.write(f"\n**Average Total per Interaction:** {metrics['average_total_per_interaction']:.1f} ms")
        
def main():
    """Main application logic"""
    page_setup()

    # Initialize conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # PDF upload for RAG context
    uploaded_pdf = st.file_uploader("Upload PDF (optional for context)", type=["pdf"])
    if uploaded_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_pdf.read())
            st.session_state.pdf_path = f.name
        st.success(" PDF loaded for RAG context")
    elif st.session_state.get("pdf_path"):
        st.info(" Using previously uploaded PDF for context")

    # Audio recording interface
    audio_bytes = audio_recorder(
        text="🎙 How May I help you ?",
        key="audio_recorder"
    )
    
    # Process audio when recorded
    if audio_bytes:
        print("Audio Length is",len(audio_bytes)," bytes") 
        
        # Start parent trace for the entire interaction
        with langfuse.start_as_current_span(name="voice_assistant_interaction") as trace:
            interaction_start = time.perf_counter()
            
            # Clear previous results
            keys_to_keep = ['conversation_history', 'course_db', 'course_status', 'pdf_path']
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]

            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_bytes)
                temp_audio_path = f.name

            # Transcribe audio with Langfuse tracing
            with langfuse.start_as_current_span(name="speech_to_text", input={"audio_length": len(audio_bytes)}) as span:
                with st.spinner(" Processing your voice..."):
                    transcribed_text = transcribe_audio(temp_audio_path)
                    span.update(output={"transcribed_text": transcribed_text})

            # Clean up temp file
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                print(f" Failed to delete temp file: {e}")

            # Validate transcription
            if not transcribed_text or str(transcribed_text).strip() == "":
                st.warning("No speech detected or transcription failed")
                return

            # Store input
            st.session_state.user_text = transcribed_text.strip()

            # Process and display results
            display_results()
            
            # Calculate total interaction time
            interaction_end = time.perf_counter()
            total_time_ms = (interaction_end - interaction_start) * 1000
            monitor.total_latency = total_time_ms  # Update the total latency
            
            # Update trace with final metrics
            trace.update(metadata={
                "total_latency_ms": total_time_ms,
                "transcribed_text": transcribed_text,
                "response_length": len(st.session_state.get("ai_response", ""))
            })
        
    # Display conversation history
    with st.expander(" Conversation History"):
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
    
    # Display latency metrics
    display_latency_metrics()
    
    # Flush Langfuse events at the end
    langfuse.flush()

if __name__ == "__main__":
    main()