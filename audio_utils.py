# import os
# import tempfile
# import requests
# import sounddevice as sd
# import numpy as np
# from scipy.io.wavfile import write
# from dotenv import load_dotenv
# from openai import OpenAI
# import webrtcvad

# # Load environment variables from .env
# load_dotenv()

# # API keys and default voice ID
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# ELEVENLABS_VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"  # Default: Roger voice

# # Initialize OpenAI client
# client = OpenAI(api_key=OPENAI_API_KEY)


# def transcribe_audio(audio_path):
#     """
#     Transcribes an audio file to text using OpenAI Whisper API.
#     """
#     with open(audio_path, "rb") as audio_file:
#         transcript = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=audio_file
#         )
#     return transcript.text


# def text_to_speech(text, voice_id=ELEVENLABS_VOICE_ID, api_key=ELEVENLABS_API_KEY):
#     """
#     Converts text to speech using ElevenLabs API and returns the audio content.
#     """
#     url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
#     headers = {
#         "xi-api-key": api_key,
#         "Content-Type": "application/json"
#     }
#     data = {
#         "text": text,
#         "voice_settings": {
#             "stability": 0.75,
#             "similarity_boost": 0.75
#         }
#     }
#     response = requests.post(url, headers=headers, json=data)
#     if response.status_code != 200:
#         raise Exception(f"Text-to-speech API error: {response.text}")
#     return response.content


# def record_audio_with_vad(duration=10, aggressiveness=2):
#     """
#     Records audio from the microphone using VAD (Voice Activity Detection)
#     to capture only voiced segments. Returns the path to the saved .wav file.
#     """
#     fs = 16000  # Sampling frequency
#     vad = webrtcvad.Vad(aggressiveness)
#     frame_duration = 30  # ms
#     frame_size = int(fs * frame_duration / 1000)
    
#     # Record raw audio for a fixed duration
#     recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
#     sd.wait()

#     # Extract bytes and split into frames
#     audio_bytes = recording.flatten().tobytes()
#     frames = [audio_bytes[i:i + frame_size * 2] for i in range(0, len(audio_bytes), frame_size * 2)]

#     # Use VAD to filter only voiced frames
#     voiced = [frame for frame in frames if vad.is_speech(frame, sample_rate=fs)]

#     if not voiced:
#         return None  # No speech detected

#     audio_array = np.frombuffer(b"".join(voiced), dtype=np.int16)

#     # Save to a temporary .wav file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#         write(f.name, fs, audio_array.reshape(-1, 1))
#         return f.name



import os
import tempfile
import requests
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from dotenv import load_dotenv
from openai import OpenAI
import webrtcvad

# Load environment variables from .env
load_dotenv()

# API keys and default voice ID
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"  # Default: Roger voice

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def transcribe_audio(audio_path):
    """
    Transcribes an audio file to text using OpenAI Whisper API.
    """
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text


def text_to_speech(text, voice_id=ELEVENLABS_VOICE_ID, api_key=ELEVENLABS_API_KEY):
    """
    Converts text to speech using ElevenLabs API and returns the audio content.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "voice_settings": {
            "stability": 0.75,
            "similarity_boost": 0.75
        }
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Text-to-speech API error: {response.text}")
    return response.content


def record_audio_with_vad(duration=10, aggressiveness=2):
    """
    Records audio from the microphone using VAD (Voice Activity Detection)
    to capture only voiced segments. Returns the path to the saved .wav file.
    """
    fs = 16000  # Sampling frequency
    vad = webrtcvad.Vad(aggressiveness)
    frame_duration = 30  # ms
    frame_size = int(fs * frame_duration / 1000)
    
    # Record raw audio for a fixed duration
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()

    # Extract bytes and split into frames
    audio_bytes = recording.flatten().tobytes()
    frames = [audio_bytes[i:i + frame_size * 2] for i in range(0, len(audio_bytes), frame_size * 2)]

    # Use VAD to filter only voiced frames
    voiced = [frame for frame in frames if vad.is_speech(frame, sample_rate=fs)]

    if not voiced:
        return None  # No speech detected

    audio_array = np.frombuffer(b"".join(voiced), dtype=np.int16)

    # Save to a temporary .wav file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        write(f.name, fs, audio_array.reshape(-1, 1))
        return f.name
