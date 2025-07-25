import azure.cognitiveservices.speech as speechsdk
import time


# Create a speech configuration
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)

# Create a recognizer
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

# Event handler
def on_recognized(args):
    if args.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Transcribed:", args.result.text)
    elif args.result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech detected - please speak louder")

# Connect event
speech_recognizer.recognized.connect(on_recognized)

# Start continuous recognition
print("Listening... Speak now (press Ctrl+C to stop)")
speech_recognizer.start_continuous_recognition()

# Keep the program running
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    speech_recognizer.stop_continuous_recognition()
    print("\nStopped transcription")
