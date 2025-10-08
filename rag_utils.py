
import os
import tempfile
import uuid
from typing import Optional
import requests
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from conversational_memory import append_to_conversation, get_conversation_history
from utils.api_monitor import monitor
from utils.get_ticket_status import get_ticket_status
from pinecone import Pinecone
from dotenv import load_dotenv

# ------------------------
# Load environment
# ------------------------
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "dishaai")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set in environment")

# ------------------------
# Clients
# ------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# ------------------------
# Ensure Pinecone index exists and has correct dimensions
# ------------------------
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # dimension of text-embedding-3-small

if pc.has_index(PINECONE_INDEX):
    idx_desc = pc.describe_index(PINECONE_INDEX)
    if idx_desc['dimension'] != EMBED_DIM:
        print(f"[INFO] Deleting Pinecone index '{PINECONE_INDEX}' due to dimension mismatch...")
        pc.delete_index(PINECONE_INDEX)
        pc.create_index_for_model(
            name=PINECONE_INDEX,
            cloud=PINECONE_ENV,
            region=PINECONE_REGION,
            embed={"model": EMBED_MODEL, "field_map": {"text": "chunk_text"}}
        )
else:
    pc.create_index_for_model(
        name=PINECONE_INDEX,
        cloud=PINECONE_ENV,
        region=PINECONE_REGION,
        embed={"model": EMBED_MODEL, "field_map": {"text": "chunk_text"}}
    )

index = pc.Index(PINECONE_INDEX)

# ------------------------
# Parameters
# ------------------------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 4
SIMILARITY_SCORE_THRESHOLD = 0.2

# ------------------------
# Helper functions
# ------------------------
def format_history_for_openai(history):
    messages = []
    for turn in history:
        if "role" in turn and "content" in turn:
            messages.append({"role": turn["role"], "content": turn["content"]})
    return messages

def _chunk_pdf_to_text_chunks(pdf_path: str) -> list:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for doc in documents:
        for chunk_text in splitter.split_text(doc.page_content):
            chunks.append({
                "text": chunk_text,
                "source": getattr(doc, "metadata", {}).get("source", "unknown")
            })
    return chunks

def _upsert_chunks_to_pinecone(chunks: list, file_id: str):
    texts = [chunk["text"] for chunk in chunks]
    embeddings_response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    vectors = [e.embedding for e in embeddings_response.data]

    to_upsert = []
    for idx, vector in enumerate(vectors):
        to_upsert.append({
            "id": f"{file_id}_{idx}",
            "values": vector,
            "metadata": {
                "text": chunks[idx]["text"],
                "source": chunks[idx].get("source", "unknown"),
                "file_id": file_id
            }
        })
    index.upsert(vectors=to_upsert)

def _query_pinecone_with_text(text: str, top_k: int = TOP_K) -> list:
    embedding_response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    query_vector = embedding_response.data[0].embedding

    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return response.get("matches", [])

# ------------------------
# LLM Processing
# ------------------------
@monitor.track("LLM")
def process_text_with_llm(question: str, chat_history: Optional[list] = None) -> str:
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

# ------------------------
# RAG Processing
# ------------------------
@monitor.track("RAG")
def process_pdf_and_ask(uploaded_pdf, question: str) -> str:
    if uploaded_pdf is None:
        return generate_general_response(question)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_pdf_path = tmp_file.name

    try:
        with open(tmp_pdf_path, "rb") as f:
            data = f.read()
            file_id = str(uuid.uuid5(uuid.NAMESPACE_URL, data.hex()))
    except Exception:
        file_id = str(uuid.uuid4())

    existing_hits = _query_pinecone_with_text(f"file_id:{file_id}", top_k=1)
    if not existing_hits:
        chunks = _chunk_pdf_to_text_chunks(tmp_pdf_path)
        _upsert_chunks_to_pinecone(chunks, file_id)

    hits = _query_pinecone_with_text(question, top_k=TOP_K)

    context = ""
    if hits:
        good_hits = [h for h in hits if (h.get("score") is not None and h.get("score") >= SIMILARITY_SCORE_THRESHOLD)]
        if good_hits:
            top_texts = []
            for h in good_hits[:TOP_K]:
                txt = h.get("metadata", {}).get("text", "")
                src = h.get("metadata", {}).get("source", "unknown")
                top_texts.append(f"Source: {src}\n{txt}")
            context = "\n\n".join(top_texts)

    if context:
        prompt = f"Based on the following document excerpts, answer the question as clearly and accurately as possible:\n\n{context}\n\nQuestion: {question}\nAnswer:"
        append_to_conversation("user", question)
        formatted_history = format_history_for_openai(get_conversation_history())
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=formatted_history + [{"role": "user", "content": prompt}],
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip()
        append_to_conversation("assistant", answer)
        return answer

    return generate_general_response(question)

# ------------------------
# LPU-specific response
# ------------------------
# @monitor.track("LLM+LPU")
# def generate_lpu_response(query: str, course_db) -> str:
#     if any(kw in query.lower() for kw in ["btech", "engineering", "b.tech"]):
#         courses = course_db.search_courses("btech engineering", 3)
#         if courses:
#             return generate_btech_response(courses)
        
#     return generate_general_response(query)

@monitor.track("LLM+LPU")
def generate_lpu_response(query: str, course_db) -> str:
    """
    Generate a response for LPU-related queries.
    Checks semantic relevance using Pinecone first, then falls back to course DB.
    """
    # ------------------------
    # Step 1: Semantic check in Pinecone
    # ------------------------
    # Query Pinecone for semantic similarity
    try:
        # Use default namespace
        hits = index.search(
            namespace="__default__",
            query={"inputs": {"text": query}},
            top_k=3,  # Get top 3 similar chunks
            fields=["chunk_text", "source"]
        )

        # Filter hits with a threshold score
        semantic_hits = [
            h for h in hits.get("matches", []) if h.get("score", 0) >= 0.2
        ]

        if semantic_hits:
            # If semantically related, construct context
            context_texts = [
                f"Source: {h['metadata'].get('source', 'unknown')}\n{h['metadata'].get('chunk_text','')}"
                for h in semantic_hits
            ]
            context = "\n\n".join(context_texts)

            # Build prompt with context
            prompt = f"Based on the following document excerpts, answer the question as clearly as possible:\n\n{context}\n\nQuestion: {query}\nAnswer:"

            # Call LLM
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            answer = response.choices[0].message.content.strip()
            return answer

    except Exception as e:
        print("Semantic search failed:", e)

    # ------------------------
    # Step 2: Keyword/course DB fallback
    # ------------------------
    lpu_keywords = ["btech", "engineering", "b.tech", "lpu", "lovely", "university", "admission", "course"]
    if any(kw in query.lower() for kw in lpu_keywords):
        courses = course_db.search_courses("btech engineering", 3)
        if courses:
            return generate_btech_response(courses)

    # ------------------------
    # Step 3: General LLM response
    # ------------------------
    return generate_general_response(query)


def generate_btech_response(courses: list) -> str:
    url = "http://98.70.101.220:11434/api/generate"
    payload = {"model": "lpu-assistant", "prompt": "Hello from outside!", "stream": False}
    response = requests.post(url, json=payload)
    try:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Sorry, no response")
    except Exception:
        return "Sorry, no response from LPU assistant"

# def generate_general_response(query: str) -> str:
#     if "ticket" in query.lower() and "status" in query.lower():
#         ticket_id = "ec89b530e1c74e44be2f1f5b569f6c79"
#         return get_ticket_status(ticket_id)

#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You're an LPU admission counselor. Respond in friendly Indian English (1-2 sentences)."},
#             {"role": "user", "content": query}
#         ],
#         temperature=0.7
#     )
#     return response.choices[0].message.content


def generate_general_response(query: str) -> str:
    """
    Generates a general response for any query.
    First tries semantic search in Pinecone, then falls back to standard LLM.
    """
    # ------------------------
    # Step 1: Ticket status shortcut
    # ------------------------
    if "ticket" in query.lower() and "status" in query.lower():
        ticket_id = "ec89b530e1c74e44be2f1f5b569f6c79"
        return get_ticket_status(ticket_id)

    # ------------------------
    # Step 2: Semantic search in Pinecone
    # ------------------------
    try:
        hits = _query_pinecone_with_text(query, top_k=3)

        # Filter by similarity threshold
        semantic_hits = [h for h in hits if h.get("score", 0) >= 0.2]

        if semantic_hits:
            context_texts = [
                f"Source: {h['metadata'].get('source', 'unknown')}\n{h['metadata'].get('text','')}"
                for h in semantic_hits
            ]
            context = "\n\n".join(context_texts)

            prompt = f"Based on the following relevant document excerpts, answer the question as clearly as possible:\n\n{context}\n\nQuestion: {query}\nAnswer:"

            response_obj = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response_obj.choices[0].message.content.strip()

    except Exception as e:
        print("Semantic search failed, using standard LLM:", e)

    # ------------------------
    # Step 3: Standard LLM fallback
    # ------------------------
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're an LPU admission counselor. Respond in friendly Indian English (1-2 sentences)."},
            {"role": "user", "content": query}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()
