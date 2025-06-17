# import os
# import tempfile
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from openai import OpenAI
# from dotenv import load_dotenv
# from conversational_memory import append_to_conversation, get_conversation_history

# load_dotenv()
# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# def format_history_for_openai(history):
#     messages = []
#     for turn in history:
#         if "user" in turn:
#             messages.append({"role": "user", "content": turn["user"]})
#         if "assistant" in turn:
#             messages.append({"role": "assistant", "content": turn["assistant"]})
#     return messages

# def process_text_with_llm(question):
#     append_to_conversation("user", question)

#     history = get_conversation_history()
#     formatted_history = format_history_for_openai(history)

#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=formatted_history + [{"role": "user", "content": question}]
#     )

#     answer = response.choices[0].message.content.strip()
#     append_to_conversation("assistant", answer)
#     return answer

# def process_pdf_and_ask(uploaded_pdf, question):
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
#     history = get_conversation_history()
#     formatted_history = format_history_for_openai(history)

#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=formatted_history + [{"role": "user", "content": prompt}]
#     )

#     answer = response.choices[0].message.content.strip()
#     append_to_conversation("assistant", answer)
#     return answer



import os
import tempfile
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from conversational_memory import append_to_conversation, get_conversation_history

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

def process_text_with_llm(question, chat_history=None):
    append_to_conversation("user", question)
    
    history = chat_history or get_conversation_history()
    formatted_history = format_history_for_openai(history)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=formatted_history + [{"role": "user", "content": question}]
    )

    answer = response.choices[0].message.content.strip()
    append_to_conversation("assistant", answer)
    return answer

def process_pdf_and_ask(uploaded_pdf, question):
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
        messages=formatted_history + [{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip()
    append_to_conversation("assistant", answer)
    return answer
