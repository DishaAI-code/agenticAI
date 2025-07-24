# """
# 📁 sentiment_utils.py

# 🎯 Purpose:
# Analyzes user input text to extract **sentiment** and **intent** using OpenAI's GPT model.

# 🔍 Technical Flow:

# 1. 🔑 Environment Setup:
#    - Loads OpenAI API key securely from `.env` file using `dotenv`.

# 2. 🤖 Sentiment Classification:
#    - Sends system prompt instructing GPT to classify sentiment strictly as:
#      → "Positive", "Negative", or "Neutral"

# 3. 🧠 Intent Detection:
#    - Sends system prompt to GPT to classify user intent into a short 2–3 word label.
#      Examples: "Question", "Request", "Complaint", etc.

# 4. 🔁 Output:
#    - Returns a tuple: `(sentiment, intent)`
#    - If any failure occurs, returns `(None, None)`

# ✅ Used In:
# - `app.py` during result rendering to display emotional tone and intent.
# """

# from openai import OpenAI
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def analyze_sentiment_intent(text):
#     """
#     Analyze text for sentiment and intent using OpenAI
#     Args:
#         text: Input text to analyze
#     Returns:
#         Tuple of (sentiment, intent) or (None, None) if failed
#     """
#     try:
#         # Sentiment analysis
#         sentiment_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "Classify sentiment as Positive/Negative/Neutral only"},
#                 {"role": "user", "content": f"Text: {text}"}
#             ]
#         )
#         sentiment = sentiment_response.choices[0].message.content.strip()

#         # Intent analysis
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
#         print(f"Analysis error: {e}")
#         return None, None


# data - 24th--- testing this lower code final code is upper part code 
"""
sentiment_utils.py
Purpose: Analyze sentiment and intent with latency monitoring
"""

# from openai import OpenAI
# import os
# from dotenv import load_dotenv
# from utils.api_monitor import monitor

# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# @monitor.track("Sentiment")
# def analyze_sentiment_intent(text):
#     """
#     Analyze text for sentiment and intent using OpenAI
#     Args:
#         text: Input text to analyze
#     Returns:
#         Tuple of (sentiment, intent) or (None, None) if failed
#     """
#     try:
#         # Sentiment analysis
#         sentiment_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "Classify sentiment as Positive/Negative/Neutral only"},
#                 {"role": "user", "content": f"Text: {text}"}
#             ]
#         )
#         sentiment = sentiment_response.choices[0].message.content.strip()

#         # Intent analysis
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
#         print(f"Analysis error: {e}")
#         return None, None