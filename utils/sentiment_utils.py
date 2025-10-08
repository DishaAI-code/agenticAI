

# # # data - 24th--- testing this lower code final code is upper part code 
# # """
# # sentiment_utils.py
# # Purpose: Analyze sentiment and intent with latency monitoring
# # """

# # from openai import OpenAI
# # import os
# # from dotenv import load_dotenv
# # from utils.api_monitor import monitor

# # load_dotenv()
# # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # @monitor.track("Sentiment")
# # def analyze_sentiment_intent(text):
# #     """
# #     Analyze text for sentiment and intent using OpenAI
# #     Args:
# #         text: Input text to analyze
# #     Returns:
# #         Tuple of (sentiment, intent) or (None, None) if failed
# #     """
# #     try:
# #         # Sentiment analysis
# #         sentiment_response = client.chat.completions.create(
# #             model="gpt-3.5-turbo",
# #             messages=[
# #                 {"role": "system", "content": "Classify sentiment as Positive/Negative/Neutral only"},
# #                 {"role": "user", "content": f"Text: {text}"}
# #             ]
# #         )
# #         sentiment = sentiment_response.choices[0].message.content.strip()

# #         # Intent analysis
# #         intent_response = client.chat.completions.create(
# #             model="gpt-3.5-turbo",
# #             messages=[
# #                 {"role": "system", "content": "Classify intent in 2-3 words (e.g., 'Question', 'Request', 'Complaint')"},
# #                 {"role": "user", "content": f"Text: {text}"}
# #             ]
# #         )
# #         intent = intent_response.choices[0].message.content.strip()

# #         return sentiment, intent
# #     except Exception as e:
# #         print(f"Analysis error: {e}")
# #         return None, None


# """
# sentiment_utils.py
# Purpose: Analyze sentiment and intent with latency monitoring and Langfuse tracing
# """

# from openai import OpenAI
# import os
# from dotenv import load_dotenv
# from utils.api_monitor import monitor
# from langfuse import get_client, observe
# # from langfuse.decorators import langfuse_context

# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# langfuse = get_client()

# @observe
# @monitor.track("Sentiment")
# def analyze_sentiment_intent(text):
#     """
#     Analyze text for sentiment and intent using OpenAI with Langfuse tracing
#     Args:
#         text: Input text to analyze
#     Returns:
#         Tuple of (sentiment, intent) or (None, None) if failed
#     """
#     try:
#         # Sentiment analysis with Langfuse generation
#         # with langfuse.start_as_current_generation(
#         #     name="sentiment_analysis", 
#         #     model="gpt-3.5-turbo",
#         #     input=text,
#         #     metadata={"analysis_type": "sentiment"}
#         # ) as gen:
#         sentiment_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "Classify sentiment as Positive/Negative/Neutral only"},
#                 {"role": "user", "content": f"Text: {text}"}
#             ]
#         )
#         sentiment = sentiment_response.choices[0].message.content.strip()
#             # gen.update(output=sentiment)

#         # Intent analysis with Langfuse generation
#         # with langfuse.start_as_current_generation(
#         #     name="intent_analysis", 
#         #     model="gpt-3.5-turbo",
#         #     input=text,
#         #     metadata={"analysis_type": "intent"}
#         # ) as gen:
#         intent_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "Classify intent in 2-3 words (e.g., 'Question', 'Request', 'Complaint')"},
#                 {"role": "user", "content": f"Text: {text}"}
#                 ]
#         )
#         intent = intent_response.choices[0].message.content.strip()
#             # gen.update(output=intent)

#         return sentiment, intent
#     except Exception as e:
#         print(f"Analysis error: {e}")
#         return None, None