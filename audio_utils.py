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

import os
import numpy as np
import queue
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from streamlit_webrtc import WebRtcMode, webrtc_streamer

load_dotenv()

class AudioHandler:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("AZURE_SPEECH_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION")
        )
        self.speech_config.speech_recognition_language = "en-US"
        self.speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1500"
        )

    def process_audio(self, frame):
        self.audio_queue.put(frame.to_ndarray())
        return frame

def recognize_speech_azure():
    """Auto-detecting speech recognition"""
    handler = AudioHandler()
    
    webrtc_ctx = webrtc_streamer(
        key="speech-recognition",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioHandler,
        media_stream_constraints={
            "audio": {
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True
            },
            "video": False
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

    if webrtc_ctx.audio_processor:
        try:
            # Configure Azure recognizer
            audio_input = speechsdk.audio.PushAudioInputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=audio_input)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=handler.speech_config,
                audio_config=audio_config
            )
            
            # Stream audio to Azure
            while True:
                try:
                    audio_chunk = handler.audio_queue.get(timeout=5)
                    audio_input.write(np.frombuffer(audio_chunk, dtype=np.int16).tobytes())
                except queue.Empty:
                    break
            
            # Get final result
            result = recognizer.recognize_once_async().get()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            return "No speech detected"
            
        except Exception as e:
            return f"Error: {str(e)}"
    return None