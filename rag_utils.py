import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def process_text_with_llm(question):
    """Process question with LLM when no PDF is available"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content.strip()

def process_pdf_and_ask(uploaded_pdf, question):
    """Process PDF and answer question using RAG"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_pdf_path = tmp_file.name

    loader = PyPDFLoader(tmp_pdf_path)
    documents = loader.load()
    
    tokenizer = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.split_documents(documents)

    vectordb = FAISS.from_documents(docs, tokenizer)
    results = vectordb.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""Based on the following document excerpts, answer the question at the end as clearly and accurately as possible.

Document Chunks:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()



# updated
# import os
# import tempfile
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from openai import OpenAI

# # 🟰 ADD THESE LINES TO HANDLE LOCAL .env
# from dotenv import load_dotenv
# load_dotenv()

# #loading api key from dotenv
# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# def process_pdf_and_ask(uploaded_pdf, question):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(uploaded_pdf.read())
#         tmp_pdf_path = tmp_file.name

#     loader = PyPDFLoader(tmp_pdf_path)
#     documents = loader.load()
#     return rag_answer_from_text(question, documents)


# def rag_answer_from_text(question, documents):
#     tokenizer = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=512,
#         chunk_overlap=50,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     docs = splitter.split_documents(documents)

#     vectordb = FAISS.from_documents(docs, tokenizer)
#     results = vectordb.similarity_search(question, k=3)

#     context = "\n\n".join([doc.page_content for doc in results])

#     prompt = f"""Based on the following document excerpts, answer the question at the end as clearly and accurately as possible.

# Document Chunks:
# {context}

# Question: {question}
# Answer:"""

#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     return response.choices[0].message.content.strip()
