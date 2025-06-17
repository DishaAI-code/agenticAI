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



# conversational_memory.py

# import streamlit as st

# def init_conversational_memory():
#     if "conversation_history" not in st.session_state:
#         st.session_state.conversation_history = []
#         print("✅ Initialized empty conversation history")

# def append_to_conversation(role, content):
#     # Skip empty content
#     if not content or str(content).strip() == "":
#         print(f"⚠️ Skipping empty {role} message")
#         return
        
#     # Validate role
#     if role not in ["user", "assistant"]:
#         raise ValueError("Role must be 'user' or 'assistant'")
    
#     print(f"📩 Appending to history: {role} - {content[:50]}...")  # Log first 50 chars
#     st.session_state.conversation_history.append({
#         "role": role,
#         "content": content.strip()
#     })

#     # Trim history to last 10 exchanges (20 messages max)
#     MAX_HISTORY = 10
#     if len(st.session_state.conversation_history) > MAX_HISTORY * 2:
#         st.session_state.conversation_history = st.session_state.conversation_history[-MAX_HISTORY * 2:]
#         print(" Trimmed conversation history")

# def get_conversation_history():
#     # Return only non-empty entries
#     history = st.session_state.get("conversation_history", [])
#     return [entry for entry in history if entry.get("content", "").strip() != ""]