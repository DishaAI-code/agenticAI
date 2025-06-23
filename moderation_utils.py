
"""
📁 moderation_utils.py

🎯 Purpose:
Performs content moderation using OpenAI's Moderation API to ensure safe and appropriate user input
before further processing by the language model.

🔧 Technical Workflow:

1. 🔐 Environment Configuration:
   - Loads `OPENAI_API_KEY` from `.env` using `dotenv`.

2. ⚙️ Initialization:
   - Instantiates an OpenAI client using the API key.

3. 🛡️ Moderation Process:
   - Submits the input `text` to OpenAI’s moderation endpoint using the latest moderation model.
   - Evaluates the `flagged` boolean and `category_scores` (e.g., violence, hate, self-harm).

4. 🚨 Flag Detection:
   - If any category exceeds the safety threshold (e.g., `violence_score > 0.2`) or `flagged=True`,
     the input is considered unsafe.
   - Returns a list of flagged categories.

5. ❌ Fail-Safe:
   - If the moderation check fails due to API or runtime error, returns a fallback error reason.

✅ Returns:
- Tuple `(is_flagged: bool, categories: List[str])`
"""

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
