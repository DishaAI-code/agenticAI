"""
📁 conversational_memory.py

🎯 Purpose:
Manages conversational memory within a Streamlit session by storing, retrieving, and trimming
dialogue history between the user and assistant.

🔧 Technical Workflow:

1. 🧠 Session Initialization:
   - `init_conversational_memory()`: Checks if `conversation_history` exists in `st.session_state`.
     If not, it initializes it as an empty list.

2. 📝 Message Appending:
   - `append_to_conversation(role, content)`:
     - Appends messages to the session's memory.
     - Enforces valid roles (`user` or `assistant`) and non-empty content.
     - Maintains a rolling buffer by keeping only the last 10 interaction pairs (20 entries max).

3. 📤 Memory Retrieval:
   - `get_conversation_history()`:
     - Returns a cleaned version of the conversation history, filtering out empty messages.
     - Output is used to construct LLM prompts in context-aware queries.

✅ Key Benefits:
- Enables dynamic multi-turn conversations.
- Prevents session memory overload by pruning old messages.
- Fully integrated with Streamlit's state handling mechanism.
"""

import streamlit as st

def init_conversational_memory():
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
        print("✅ Initialized empty conversation history")

def append_to_conversation(role, content):
    if not content or str(content).strip() == "":
        print(f"⚠️ Skipping empty {role} message")
        return
    if role not in ["user", "assistant"]:
        raise ValueError("Role must be 'user' or 'assistant'")
    
    st.session_state.conversation_history.append({
        "role": role,
        "content": content.strip()
    })

    MAX_HISTORY = 10
    if len(st.session_state.conversation_history) > MAX_HISTORY * 2:
        st.session_state.conversation_history = st.session_state.conversation_history[-MAX_HISTORY * 2:]

def get_conversation_history():
    history = st.session_state.get("conversation_history", [])
    return [entry for entry in history if entry.get("content", "").strip() != ""]



