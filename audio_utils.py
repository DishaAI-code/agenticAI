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

# audio_utils.py

# import os
# import requests
# from io import BytesIO
# import base64
# import azure.cognitiveservices.speech as speechsdk
# from dotenv import load_dotenv

# load_dotenv()  # Load .env for Azure and ElevenLabs keys

# def recognize_from_mic():
#     """Capture audio from microphone and transcribe using Azure Speech."""
#     try:
#         speech_config = speechsdk.SpeechConfig(
#             subscription=os.getenv("AZURE_SPEECH_KEY"),
#             region=os.getenv("AZURE_SPEECH_REGION")
#         )
#         speech_config.speech_recognition_language = "en-US"
#         recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
#         result = recognizer.recognize_once_async().get()
        
#         if result.reason == speechsdk.ResultReason.RecognizedSpeech:
#             return result.text
#         else:
#             return ""
#     except Exception as e:
#         return f"[Azure Speech Error] {str(e)}"

# def text_to_speech_elevenlabs(text: str):
#     """Convert text to speech using ElevenLabs API and return audio as base64."""
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
#             audio_bytes = BytesIO(response.content)
#             audio_base64 = base64.b64encode(audio_bytes.read()).decode("utf-8")
#             return audio_base64
#         else:
#             return None
#     except Exception as e:
#         return None

# audio_utils.py

import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

def transcribe_audio_file(audio_path: str) -> str:
    try:
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        service_region = os.getenv("AZURE_REGION", "eastus")

        if not speech_key:
            raise ValueError("Azure Speech Key not found in environment variables")

        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        result = recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return ""
        else:
            raise Exception("Speech recognition failed")

    except Exception as e:
        print(f"[Transcription Error]: {e}")
        return ""
