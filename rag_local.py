import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai


# ---------- Setup ----------

def configure_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")
    genai.configure(api_key=api_key)


# ---------- Loading local store ----------

def load_local_store(store_dir: str):
    store_path = Path(store_dir)
    index_path = store_path / "local_index.faiss"
    emb_path = store_path / "chunk_embeddings.npy"
    meta_path = store_path / "chunks.json"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Did not find FAISS index / chunks in {store_dir}. "
            "Run ingestion/ingest_local.py first."
        )

    index = faiss.read_index(str(index_path))
    embeddings = np.load(emb_path)
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, embeddings, metadata


# ---------- Embedding + search ----------

def embed_query(query: str) -> np.ndarray:
    model_name = "text-embedding-004"  # still fine for now, can later switch to gemini-embedding-001
    resp = genai.embed_content(
        model=model_name,
        content=query,
    )
    emb = np.array(resp["embedding"], dtype="float32")
    return emb


def search_chunks(
    index: faiss.Index,
    query_emb: np.ndarray,
    metadata: List[Dict],
    k: int = 5,
) -> List[Dict]:
    """
    Return top-k chunk metadata dicts with distances.
    """
    query_emb = np.expand_dims(query_emb, axis=0)  # shape (1, dim)
    distances, indices = index.search(query_emb, k)
    indices = indices[0]
    distances = distances[0]

    results = []
    for dist, idx in zip(distances, indices):
        if idx == -1:
            continue
        meta = metadata[int(idx)].copy()
        meta["distance"] = float(dist)
        results.append(meta)
    return results


# ---------- RAG answer generation ----------

def build_rag_prompt(question: str, chunks: List[Dict]) -> List[Dict]:
    """
    Build a messages list for Gemini with context + question.
    """
    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"[doc_id={c['doc_id']}, chunk_id={c['chunk_id']}]\n{c['chunk_text']}\n"
        )
    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant that must answer using ONLY the provided context.\n"
        "If the context is insufficient, say you are not sure.\n"
        "Always cite which doc_id and chunk_id you used."
    )

    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly and concisely."
    )

    messages = [
        {"role": "user", "parts": [system_prompt + "\n\n" + user_prompt]}
    ]
    return messages


def generate_answer(question: str, chunks: List[Dict]) -> str:
    # UPDATED MODEL NAME HERE 👇
    model = genai.GenerativeModel("gemini-2.5-flash")

    messages = build_rag_prompt(question, chunks)
    resp = model.generate_content(messages)

    return resp.text


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Query local RAG index.")
    parser.add_argument(
        "--store",
        type=str,
        default="local_store",
        help="Folder with local_index.faiss, chunks.json, etc.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask over the documents.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve.",
    )
    args = parser.parse_args()

    configure_gemini()
    index, embeddings, metadata = load_local_store(args.store)

    print(f"\nEmbedding query: {args.question!r}")
    q_emb = embed_query(args.question)

    print(f"Searching top-{args.k} chunks...")
    top_chunks = search_chunks(index, q_emb, metadata, k=args.k)

    print("\nRetrieved chunks:")
    for c in top_chunks:
        print(f"- doc_id={c['doc_id']}, chunk_id={c['chunk_id']}, distance={c['distance']:.4f}")

    print("\nGenerating answer with Gemini...")
    answer = generate_answer(args.question, top_chunks)

    print("\n===== ANSWER =====\n")
    print(answer)
    print("\n==================\n")


if __name__ == "__main__":
    main()
