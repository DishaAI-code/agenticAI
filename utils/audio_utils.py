


# """
# audio_utils.py
# Purpose: Provides audio capabilities (STT/TTS) with latency monitoring
# """

# import os
# import base64
# import requests
# import tempfile
# from dotenv import load_dotenv
# from azure.cognitiveservices.speech import SpeechConfig, AudioConfig, SpeechRecognizer
# from utils.api_monitor import monitor

# # Load environment variables
# load_dotenv()

# @monitor.track("STT")
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
#         print(f"Transcription error: {str(e)}")
#         return None

# @monitor.track("TTS")
# def text_to_speech_elevenlabs(text: str):
#     """
#     Convert text to speech using ElevenLabs API
#     Args:
#         text: The text to convert to speech
#     Returns:
#         Base64 encoded audio data or None if failed
#     """
#     try:
#         xi_api_key = 'sk_12257cda71d5e07e53faae0ce502d978a2f99da590add799'
#         voice_id =  "JBFqnCBsd6RMkjVDRZzb"
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

#         response = requests.post(url, json=data, headers=headers)
        
#         if response.status_code == 200:
#             return base64.b64encode(response.content).decode("utf-8")
#         else:
#             print(f"ElevenLabs TTS failed: {response.text}")
#             return None
#     except Exception as e:
#         print(f"TTS Error: {str(e)}")
#         return None



import os
import base64
import requests
from dotenv import load_dotenv
from azure.cognitiveservices.speech import SpeechConfig, AudioConfig, SpeechRecognizer
from utils.api_monitor import monitor
from langfuse import get_client, observe

load_dotenv()
langfuse = get_client()


# @observe
# @monitor.track("STT")
def transcribe_audio(file_path):
    """Azure STT"""
    with langfuse.start_as_current_span(name="azure-stt") as span:
        try:
            with langfuse.start_as_current_span(name="audio_transcription", input={"file_path": file_path}) as span:
                speech_key = os.getenv("AZURE_SPEECH_KEY")
                service_region = os.getenv("AZURE_SPEECH_REGION")
                speech_config = SpeechConfig(subscription=speech_key, region=service_region)
                audio_config = AudioConfig(filename=file_path)
                recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
                result = recognizer.recognize_once()
                text = result.text if result.text and result.text.strip() else None
                span.update(output={"transcribed_text": text})
                return text
        except Exception as e:
            span.update(output=f"Error: {e}")
            return None

# def transcribe_audio(file_path):
#     """Transcribe audio via custom Azure-deployed STT API"""
#     with langfuse.start_as_current_span(name="custom-stt") as span:
#         try:
#             url = "http://98.70.101.220:8000/transcribe"

#             # Send audio file in request
#             with open(file_path, "rb") as f:
#                 files = {"file": f}
#                 print("Sending request to STT API...",files)
#                 print("length of audio file is",len(files))
#                 response = requests.post(url, files=files)
#             if response.status_code == 200:
#                 result = response.json()
#                 text = result.get("text") or result.get("transcription")  # adapt to API response format
#                 span.update(output={"transcribed_text": text})
#                 return text

#         except Exception as e:
#            print(f"STT Error: {str(e)}")
#            return None

# @observe
# @monitor.track("TTS")
# def text_to_speech_elevenlabs(text: str):
#     """ElevenLabs TTS"""
#     with langfuse.start_as_current_span(name="elevenlabs-tts") as span:
#         try:
#             xi_api_key = 'sk_12257cda71d5e07e53faae0ce502d978a2f99da590add799'
#             voice_id = "JBFqnCBsd6RMkjVDRZzb"
#             url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

#             headers = {
#                 "Accept": "audio/mpeg",
#                 "Content-Type": "application/json",
#                 "xi-api-key": xi_api_key
#             }
#             data = {
#                 "text": text,
#                 "model_id": "eleven_monolingual_v1",
#                 "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
#             }

#             response = requests.post(url, json=data, headers=headers)
#             if response.status_code == 200:
#                 audio = base64.b64encode(response.content).decode("utf-8")
#                 span.update(output={"audio_generated": True, "audio_length": len(audio)})
#                 return audio
#             else:
#                 span.update(output=f"Failed: {response.text}")
#                 return None
#         except Exception as e:
#             span.update(output=f"Error: {e}")
#             return None


def text_to_speech_elevenlabs(text: str):
    """Local TTS service (replacing ElevenLabs)"""
    with langfuse.start_as_current_span(name="local-tts") as span:
        try:
            url = "http://98.70.101.220:8001/tts"
            
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                "text": text
            }

            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                # Encode audio as base64 (same as ElevenLabs version)
                audio = base64.b64encode(response.content).decode("utf-8")
                span.update(output={"audio_generated": True, "audio_length": len(audio)})
                return audio
            else:
                span.update(output=f"Failed: {response.text}")
                return None
        except Exception as e:
            span.update(output=f"Error: {e}")
            return None