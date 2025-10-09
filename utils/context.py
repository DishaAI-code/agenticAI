import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
# Initialize conversation history
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

# Function to send message
def chat_with_memory(user_input):
    # Add user message to history
    conversation_history.append({"role": "user", "content": user_input})

    # Call OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # or gpt-4o, gpt-3.5-turbo
        messages=conversation_history,
        temperature=0.7
    )

    # Get assistant reply
    assistant_reply = response['choices'][0]['message']['content']

    # Add assistant reply to history
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply

# Example usage
while True:
    user_input = input("You: ")
    reply = chat_with_memory(user_input)
    print("Assistant:", reply)
