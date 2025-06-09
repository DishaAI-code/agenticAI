

# from openai import OpenAI
# import os
# from dotenv import load_dotenv

# load_dotenv()
# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# def moderate_text(text: str):
#     try:
#         response = client.moderations.create(
#             model="text-moderation-latest",
#             input=text
#         )
#         output = response.results[0]
#         flagged = output.flagged

#         # Convert category_scores and categories to dicts
#         category_scores = vars(output.category_scores)
#         categories = vars(output.categories)

#         violence_score = category_scores.get("violence", 0)

#         if flagged or violence_score > 0.2:
#             flagged_categories = [
#                 k for k, v in categories.items()
#                 if v or category_scores.get(k, 0) > 0.2
#             ]
#             return True, flagged_categories

#         return False, []
#     except Exception as e:
#         print("Moderation API error:", e)
#         return None, ["moderation_error"]


# moderation_utils.py

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def moderate_text(text: str):
    try:
        response = client.moderations.create(
            model="text-moderation-latest",
            input=text
        )
        output = response.results[0]
        flagged = output.flagged

        category_scores = vars(output.category_scores)
        categories = vars(output.categories)
        violence_score = category_scores.get("violence", 0)

        if flagged or violence_score > 0.2:
            flagged_categories = [
                k for k, v in categories.items()
                if v or category_scores.get(k, 0) > 0.2
            ]
            return True, flagged_categories

        return False, []

    except Exception as e:
        print("Moderation API error:", e)
        return None, ["moderation_error"]
