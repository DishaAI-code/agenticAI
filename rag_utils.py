"""
📁 rag_utils.py

🎯 Purpose:
Implements Retrieval-Augmented Generation (RAG) using PDF documents for enhanced context-aware responses.

🔧 Technical Workflow:

1. 📄 PDF Processing:
   - Uses `PyPDFLoader` from LangChain to extract text from uploaded PDF files.
   - Splits documents into semantically meaningful chunks via `RecursiveCharacterTextSplitter`.

2. 🔍 Embedding Generation:
   - Converts text chunks into embeddings using HuggingFace's `all-MiniLM-L6-v2` model via `HuggingFaceEmbeddings`.

3. 🧠 Vector Indexing:
   - Stores document embeddings in a FAISS index to support fast vector similarity search.

4. 🧾 Prompt Construction:
   - Retrieves top-k most similar document chunks to the user query.
   - Constructs a system prompt including these chunks to give the LLM accurate and grounded context.

5. 🤖 LLM Querying:
   - Uses OpenAI's `gpt-3.5-turbo` to generate responses based on:
     - Full chat history (retrieved via `get_conversation_history()`).
     - Either plain question (in `process_text_with_llm()`) or document-grounded prompt (in `process_pdf_and_ask()`).
   - Appends both the user's question and AI response to the session memory using `append_to_conversation()`.

✅ Key Benefits:
- Adds factual grounding to chatbot answers using external documents.
- Maintains continuity and memory through conversational history.
- Keeps responses concise, contextual, and semantically aligned.
"""

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
