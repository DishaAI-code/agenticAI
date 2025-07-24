


# """
# 📁 rag_utils.py

# 🎯 Updated Purpose:
# Implements Retrieval-Augmented Generation (RAG) with:
# - PDF document processing
# - LPU course information integration
# - LLM response generation
# """

# import os
# import tempfile
# from dotenv import load_dotenv
# from openai import OpenAI
# from typing import Optional

# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# from conversational_memory import append_to_conversation, get_conversation_history

# load_dotenv()
# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# def format_history_for_openai(history):
#     messages = []
#     for turn in history:
#         if "role" in turn and "content" in turn:
#             messages.append({
#                 "role": turn["role"],
#                 "content": turn["content"]
#             })
#     return messages

# def process_text_with_llm(question: str, chat_history: Optional[list] = None) -> str:
#     """
#     Process text query with LLM using conversation history
#     Args:
#         question: User's question
#         chat_history: Optional conversation history
#     Returns:
#         Generated response
#     """
#     append_to_conversation("user", question)
    
#     history = chat_history or get_conversation_history()
#     formatted_history = format_history_for_openai(history)

#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=formatted_history + [{"role": "user", "content": question}],
#         temperature=0.7
#     )

#     answer = response.choices[0].message.content.strip()
#     append_to_conversation("assistant", answer)
#     return answer

# def process_pdf_and_ask(uploaded_pdf, question: str) -> str:
#     """
#     Process PDF and generate response using RAG
#     Args:
#         uploaded_pdf: Uploaded PDF file
#         question: User's question
#     Returns:
#         Generated response
#     """
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(uploaded_pdf.read())
#         tmp_pdf_path = tmp_file.name

#     loader = PyPDFLoader(tmp_pdf_path)
#     documents = loader.load()

#     tokenizer = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
#     splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
#     docs = splitter.split_documents(documents)

#     vectordb = FAISS.from_documents(docs, tokenizer)
#     results = vectordb.similarity_search(question, k=3)
#     context = "\n\n".join([doc.page_content for doc in results])

#     prompt = f"""Based on the following document excerpts, answer the question at the end as clearly and accurately as possible.\n\nDocument Chunks:\n{context}\n\nQuestion: {question}\nAnswer:"""

#     append_to_conversation("user", question)
#     formatted_history = format_history_for_openai(get_conversation_history())

#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=formatted_history + [{"role": "user", "content": prompt}],
#         temperature=0.7
#     )

#     answer = response.choices[0].message.content.strip()
#     append_to_conversation("assistant", answer)
#     return answer

# def generate_lpu_response(query: str, course_db) -> str:
#     """
#     Generate specialized response for LPU-related queries
#     Args:
#         query: User's query
#         course_db: Course database instance
#     Returns:
#         Generated response
#     """
#     # Check for B.Tech related queries
#     if any(kw in query.lower() for kw in ["btech", "engineering", "b.tech"]):
#         courses = course_db.search_courses("btech engineering", 3)
#         if courses:
#             return generate_btech_response(courses)
    
#     # General LPU query
#     return generate_general_response(query)

# def generate_btech_response(courses: list) -> str:
#     """
#     Generate response for B.Tech queries
#     Args:
#         courses: List of relevant courses
#     Returns:
#         Generated response
#     """
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {
#                 "role": "system",
#                 "content": f"""You're an LPU admission assistant. Respond to B.Tech queries with:
# - Friendly Indian English
# - Mention 2-3 key programs from: {courses}
# - Keep response under 20 words
# - End with a question"""
#             },
#             {"role": "user", "content": "What B.Tech programs does LPU offer?"}
#         ],
#         temperature=0.7
#     )
#     return response.choices[0].message.content

# def generate_general_response(query: str) -> str:
#     """
#     Generate response for general queries
#     Args:
#         query: User's query
#     Returns:
#         Generated response
#     """
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You're an LPU admission counselor. Respond in friendly Indian English (1-2 sentences)."},
#             {"role": "user", "content": query}
#         ],
#         temperature=0.7
#     )
#     return response.choices[0].message.content


"""
rag_utils.py
Purpose: RAG and LLM processing with latency monitoring
"""

import os
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from conversational_memory import append_to_conversation, get_conversation_history
from utils.api_monitor import monitor

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def format_history_for_openai(history):
    messages = []
    for turn in history:
        if "role" in turn and "content" in turn:
            messages.append({
                "role": turn["role"],
                "content": turn["content"]
            })
    return messages

@monitor.track("LLM")
def process_text_with_llm(question: str, chat_history: Optional[list] = None) -> str:
    """
    Process text query with LLM using conversation history
    Args:
        question: User's question
        chat_history: Optional conversation history
    Returns:
        Generated response
    """
    append_to_conversation("user", question)
    
    history = chat_history or get_conversation_history()
    formatted_history = format_history_for_openai(history)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=formatted_history + [{"role": "user", "content": question}],
        temperature=0.7
    )

    answer = response.choices[0].message.content.strip()
    append_to_conversation("assistant", answer)
    return answer

@monitor.track("RAG")
def process_pdf_and_ask(uploaded_pdf, question: str) -> str:
    """
    Process PDF and generate response using RAG
    Args:
        uploaded_pdf: Uploaded PDF file
        question: User's question
    Returns:
        Generated response
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_pdf_path = tmp_file.name

    loader = PyPDFLoader(tmp_pdf_path)
    documents = loader.load()

    tokenizer = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    vectordb = FAISS.from_documents(docs, tokenizer)
    results = vectordb.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""Based on the following document excerpts, answer the question at the end as clearly and accurately as possible.\n\nDocument Chunks:\n{context}\n\nQuestion: {question}\nAnswer:"""

    append_to_conversation("user", question)
    formatted_history = format_history_for_openai(get_conversation_history())

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=formatted_history + [{"role": "user", "content": prompt}],
        temperature=0.7
    )

    answer = response.choices[0].message.content.strip()
    append_to_conversation("assistant", answer)
    return answer

@monitor.track("LLM+LPU")
def generate_lpu_response(query: str, course_db) -> str:
    """
    Generate specialized response for LPU-related queries
    Args:
        query: User's query
        course_db: Course database instance
    Returns:
        Generated response
    """
    # Check for B.Tech related queries
    if any(kw in query.lower() for kw in ["btech", "engineering", "b.tech"]):
        courses = course_db.search_courses("btech engineering", 3)
        if courses:
            return generate_btech_response(courses)
    
    # General LPU query
    return generate_general_response(query)

def generate_btech_response(courses: list) -> str:
    """
    Generate response for B.Tech queries
    Args:
        courses: List of relevant courses
    Returns:
        Generated response
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"""You're an LPU admission assistant. Respond to B.Tech queries with:
- Friendly Indian English
- Mention 2-3 key programs from: {courses}
- Keep response under 20 words
- End with a question"""
            },
            {"role": "user", "content": "What B.Tech programs does LPU offer?"}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

def generate_general_response(query: str) -> str:
    """
    Generate response for general queries
    Args:
        query: User's query
    Returns:
        Generated response
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're an LPU admission counselor. Respond in friendly Indian English (1-2 sentences)."},
            {"role": "user", "content": query}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content