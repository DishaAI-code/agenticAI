# """
# # audio_utils.py

# Purpose:
# Provides audio capabilities: transcription and synthesis.

# Transcription (Speech-to-Text):
# - Uses Azure Speech SDK
# - Creates SpeechRecognizer with audio file and returns transcribed text.

# Text-to-Speech:
# - Uses ElevenLabs API
# - Constructs audio request with desired voice model and parameters.
# - Returns audio as base64-encoded string for playback in browser.
# """
# import os
# import base64
# import requests
# import tempfile
# from dotenv import load_dotenv
# from azure.cognitiveservices.speech import SpeechConfig, AudioConfig, SpeechRecognizer

# # Load environment variables from .env file
# load_dotenv()

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
#             print(f"ElevenLabs TTS failed: {response.text}")
#             return None
#     except Exception as e:
#         print(f"TTS Error: {str(e)}")
#         return None

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


"""
audio_utils.py
Purpose: Provides audio capabilities (STT/TTS) with latency monitoring
"""

import os
import base64
import requests
import tempfile
from dotenv import load_dotenv
from azure.cognitiveservices.speech import SpeechConfig, AudioConfig, SpeechRecognizer
from utils.api_monitor import monitor

# Load environment variables
load_dotenv()

@monitor.track("STT")
def transcribe_audio(file_path):
    """
    Transcribe audio file using Azure Speech-to-Text
    Args:
        file_path: Path to audio file
    Returns:
        Transcribed text or None if failed
    """
    try:
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        service_region = os.getenv("AZURE_SPEECH_REGION")
        speech_config = SpeechConfig(subscription=speech_key, region=service_region)
        audio_config = AudioConfig(filename=file_path)
        recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = recognizer.recognize_once()
        return result.text if result.text and result.text.strip() != "" else None
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return None

@monitor.track("TTS")
def text_to_speech_elevenlabs(text: str):
    """
    Convert text to speech using ElevenLabs API
    Args:
        text: The text to convert to speech
    Returns:
        Base64 encoded audio data or None if failed
    """
    try:
        xi_api_key = 'sk_f517a5ed8b9b1815c24e1e9cabe2dd1bd1e6844037597be7'
        voice_id =  "JBFqnCBsd6RMkjVDRZzb"
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": xi_api_key
        }

        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            return base64.b64encode(response.content).decode("utf-8")
        else:
            print(f"ElevenLabs TTS failed: {response.text}")
            return None
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return None