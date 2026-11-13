"""
Khushboo - a simple RAG chatbot
Streamlit UI + Sentence Transformers + FAISS + Ollama (local LLM)
"""

import os

import faiss
import numpy as np
import ollama
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BOT_NAME = "Khushboo"
LLM_MODEL = "llama3.2"                 # the model you pulled with Ollama (local)
GROQ_MODEL = "llama-3.1-8b-instant"    # used only on the cloud when GROQ_API_KEY is set
EMBED_MODEL = "all-MiniLM-L6-v2"       # free Sentence Transformers model
CHUNK_SIZE = 800                       # characters per chunk
CHUNK_OVERLAP = 150                    # overlap so ideas are not cut in half
TOP_K = 3                              # how many chunks to retrieve

NOT_FOUND_MSG = "I could not find this information in the uploaded document."

SYSTEM_PROMPT = f"""You are {BOT_NAME}, a helpful assistant that answers questions ONLY using the document context provided.
Rules:
1. Use only the information in the CONTEXT. Do not use outside knowledge.
2. Do not invent, guess or assume anything that is not written in the CONTEXT.
3. If the answer is not in the CONTEXT, reply exactly: "{NOT_FOUND_MSG}"
4. Keep answers clear and concise."""


# ---------- Step 3: read and chunk the document ----------
def extract_text(uploaded_file) -> str:
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = " ".join(text.split())  # clean extra spaces/newlines
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


# ---------- Step 4: embeddings ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


# ---------- Step 5: FAISS vector store ----------
def build_index(chunks):
    embedder = load_embedder()
    vectors = embedder.encode(chunks, normalize_embeddings=True)
    vectors = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity (vectors are normalized)
    index.add(vectors)
    return index


def retrieve(question, index, chunks, k=TOP_K):
    embedder = load_embedder()
    q_vec = embedder.encode([question], normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype="float32")
    scores, ids = index.search(q_vec, min(k, len(chunks)))
    return [(chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


# ---------- Steps 6-7, 9: ask the LLM ----------
def get_groq_key():
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    return key or os.environ.get("GROQ_API_KEY")


def generate_answer(question, retrieved):
    context = "\n\n".join(f"[Chunk {n + 1}] {text}" for n, (text, _) in enumerate(retrieved))
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Cloud mode: a Groq key exists, so use the hosted model
    groq_key = get_groq_key()
    if groq_key:
        from groq import Groq
        client = Groq(api_key=groq_key)
        result = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=0
        )
        return result.choices[0].message.content

    # Local mode: use Ollama running on your Mac
    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"temperature": 0},  # 0 = less creative, fewer made-up answers
    )
    return response["message"]["content"]


# ---------- Step 1: Streamlit interface ----------
st.set_page_config(page_title=BOT_NAME, page_icon="🌸")
st.title(f"🌸 {BOT_NAME}")
st.caption("Ask questions about your document. I only answer from what it says.")

# Step 2: upload document
uploaded = st.sidebar.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded is not None:
    if st.session_state.get("doc_name") != uploaded.name:
        with st.spinner("Reading and indexing your document..."):
            text = extract_text(uploaded)
            if not text.strip():
                st.sidebar.error("No text could be extracted from this file.")
                st.stop()
            chunks = chunk_text(text)
            st.session_state.chunks = chunks
            st.session_state.index = build_index(chunks)
            st.session_state.doc_name = uploaded.name
            st.session_state.messages = []
    st.sidebar.success(f"{uploaded.name} ready ({len(st.session_state.chunks)} chunks)")
else:
    st.info("Upload a PDF or TXT file in the sidebar to begin.")

# show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Source text used"):
                for n, (src, score) in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {n}** (similarity {score:.2f})")
                    st.write(src)

# chat input
if uploaded is not None:
    question = st.chat_input(f"Ask {BOT_NAME} a question...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retrieved = retrieve(question, st.session_state.index, st.session_state.chunks)
                try:
                    answer = generate_answer(question, retrieved)
                except Exception as e:
                    answer = f"Could not get an answer. Details: {e}"
                    retrieved = []
            st.markdown(answer)
            # Step 8: show the source text
            if retrieved:
                with st.expander("Source text used"):
                    for n, (src, score) in enumerate(retrieved, 1):
                        st.markdown(f"**Chunk {n}** (similarity {score:.2f})")
                        st.write(src)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": retrieved}
        )