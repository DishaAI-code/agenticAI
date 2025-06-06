# import os
# import tempfile
# import requests
# import sounddevice as sd
# import numpy as np
# from scipy.io.wavfile import write
# from dotenv import load_dotenv
# from openai import OpenAI
# import azure.cognitiveservices.speech as speechsdk
# import webrtcvad

# # Load env
# load_dotenv()

# # Keys
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
# AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
# ELEVENLABS_VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"

# client = OpenAI(api_key=OPENAI_API_KEY)


# def transcribe_audio(audio_path):
#     with open(audio_path, "rb") as audio_file:
#         transcript = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=audio_file
#         )
#     return transcript.text


# def text_to_speech(text, voice_id=ELEVENLABS_VOICE_ID, api_key=ELEVENLABS_API_KEY):
#     url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
#     headers = {
#         "xi-api-key": api_key,
#         "Content-Type": "application/json"
#     }
#     data = {
#         "text": text,
#         "voice_settings": {"stability": 0.75, "similarity_boost": 0.75}
#     }
#     response = requests.post(url, headers=headers, json=data)
#     if response.status_code != 200:
#         raise Exception(f"Text-to-speech API error: {response.text}")
#     return response.content


# def record_audio_with_vad(duration=10, aggressiveness=2):
#     fs = 16000
#     vad = webrtcvad.Vad(aggressiveness)
#     frame_duration = 30
#     frame_size = int(fs * frame_duration / 1000)

#     recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
#     sd.wait()

#     audio_bytes = recording.flatten().tobytes()
#     frames = [audio_bytes[i:i + frame_size * 2] for i in range(0, len(audio_bytes), frame_size * 2)]
#     voiced = [f for f in frames if vad.is_speech(f, sample_rate=fs)]

#     if not voiced:
#         return None

#     audio_array = np.frombuffer(b"".join(voiced), dtype=np.int16)

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#         write(f.name, fs, audio_array.reshape(-1, 1))
#         return f.name


# def recognize_speech_azure(language="en-US"):
#     audio_path = record_audio_with_vad()
#     if not audio_path:
#         return "No speech detected."

#     speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
#     speech_config.speech_recognition_language = language

#     audio_input = speechsdk.audio.AudioConfig(filename=audio_path)
#     recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

#     result = recognizer.recognize_once()

#     if result.reason == speechsdk.ResultReason.RecognizedSpeech:
#         return result.text
#     elif result.reason == speechsdk.ResultReason.NoMatch:
#         return "No speech recognized."
#     elif result.reason == speechsdk.ResultReason.Canceled:
#         cancellation = result.cancellation_details
#         return f"Canceled: {cancellation.reason} - {cancellation.error_details}"

#     speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
#     speech_config.speech_recognition_language = language
#     audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
#     recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

#     result = recognizer.recognize_once()

#     if result.reason == speechsdk.ResultReason.RecognizedSpeech:
#         return result.text
#     elif result.reason == speechsdk.ResultReason.NoMatch:
#         return "No speech recognized."
#     elif result.reason == speechsdk.ResultReason.Canceled:
#         cancellation = result.cancellation_details
#         return f"Canceled: {cancellation.reason} - {cancellation.error_details}"


# updated code 


# audio_utils.py (updated)
import os
import tempfile
import requests
from dotenv import load_dotenv
from openai import OpenAI
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
ELEVENLABS_VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"

client = OpenAI(api_key=OPENAI_API_KEY)

def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

def text_to_speech(text, voice_id=ELEVENLABS_VOICE_ID, api_key=ELEVENLABS_API_KEY):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "voice_settings": {"stability": 0.75, "similarity_boost": 0.75}
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Text-to-speech API error: {response.text}")
    return response.content

def recognize_speech_azure(audio_path, language="en-US"):
    if not audio_path:
        return "No audio provided."

    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_recognition_language = language
    audio_input = speechsdk.audio.AudioConfig(filename=audio_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "No speech recognized."
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        return f"Canceled: {cancellation.reason} - {cancellation.error_details}"


# import os
# import tempfile
# import requests
# import azure.cognitiveservices.speech as speechsdk
# from dotenv import load_dotenv
# from openai import OpenAI

# # Load env
# load_dotenv()

# # Keys
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
# AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
# ELEVENLABS_VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"

# client = OpenAI(api_key=OPENAI_API_KEY)

# def recognize_speech_azure(audio_bytes, language="en-US"):
#     """Recognize speech from audio bytes using Azure Speech Service"""
#     # Save audio bytes to temporary file
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
#         tmp_file.write(audio_bytes)
#         tmp_path = tmp_file.name
    
#     try:
#         # Configure Azure Speech
#         speech_config = speechsdk.SpeechConfig(
#             subscription=AZURE_SPEECH_KEY, 
#             region=AZURE_SPEECH_REGION
#         )
#         speech_config.speech_recognition_language = language
#         audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)
        
#         # Create recognizer
#         recognizer = speechsdk.SpeechRecognizer(
#             speech_config=speech_config, 
#             audio_config=audio_config
#         )
        
#         # Recognize speech
#         result = recognizer.recognize_once()
        
#         # Process result
#         if result.reason == speechsdk.ResultReason.RecognizedSpeech:
#             return result.text
#         elif result.reason == speechsdk.ResultReason.NoMatch:
#             return "No speech recognized."
#         else:
#             return f"Error: {result.cancellation_details.error_details}"
    
#     except Exception as e:
#         return f"Recognition error: {str(e)}"
#     finally:
#         # Clean up temp file
#         if os.path.exists(tmp_path):
#             os.unlink(tmp_path)

# def text_to_speech(text, voice_id=ELEVENLABS_VOICE_ID, api_key=ELEVENLABS_API_KEY):
#     """Convert text to speech using ElevenLabs"""
#     url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
#     headers = {
#         "xi-api-key": api_key,
#         "Content-Type": "application/json"
#     }
#     data = {
#         "text": text,
#         "voice_settings": {"stability": 0.75, "similarity_boost": 0.75}
#     }
#     response = requests.post(url, headers=headers, json=data)
#     if response.status_code != 200:
#         raise Exception(f"Text-to-speech API error: {response.text}")
#     return response.content