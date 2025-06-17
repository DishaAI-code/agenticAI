from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_sentiment_intent(text):
    """
    Analyze text for sentiment and intent using OpenAI
    Args:
        text: Input text to analyze
    Returns:
        Tuple of (sentiment, intent) or (None, None) if failed
    """
    try:
        # Sentiment analysis
        sentiment_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Classify sentiment as Positive/Negative/Neutral only"},
                {"role": "user", "content": f"Text: {text}"}
            ]
        )
        sentiment = sentiment_response.choices[0].message.content.strip()

        # Intent analysis
        intent_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Classify intent in 2-3 words (e.g., 'Question', 'Request', 'Complaint')"},
                {"role": "user", "content": f"Text: {text}"}
            ]
        )
        intent = intent_response.choices[0].message.content.strip()

        return sentiment, intent
    except Exception as e:
        print(f"Analysis error: {e}")
        return None, None