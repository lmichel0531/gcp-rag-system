import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import pdfplumber
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai


# ---------- Helpers for loading docs ----------

def load_text_from_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    pages_text = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    return "\n".join(pages_text)


def load_text_from_txt(path: Path) -> str:
    """Load text from a plain .txt file."""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(input_dir: Path) -> List[Tuple[str, str]]:
    """
    Load all .pdf and .txt files in the directory.
    Returns a list of (doc_id, full_text).
    """
    docs = []
    for path in sorted(input_dir.glob("**/*")):
        if path.suffix.lower() == ".pdf":
            text = load_text_from_pdf(path)
        elif path.suffix.lower() == ".txt":
            text = load_text_from_txt(path)
        else:
            continue  # skip other file types

        doc_id = path.stem  # filename without extension
        if text.strip():
            docs.append((doc_id, text))
            print(f"Loaded document: {path.name} (chars={len(text)})")
        else:
            print(f"Warning: {path.name} contained no text.")
    return docs


# ---------- Chunking ----------

def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> List[str]:
    """
    Simple character-based chunking with overlap.
    This is good enough for a first prototype.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------- Embeddings with Gemini ----------

def configure_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")
    genai.configure(api_key=api_key)


def embed_chunks(chunks: List[str]) -> np.ndarray:
    """
    Get embeddings for a list of chunks using Gemini.
    Returns a numpy array of shape (num_chunks, dim).
    """
    model_name = "text-embedding-004"  # current Gemini embedding model
    embeddings = []

    print(f"Embedding {len(chunks)} chunks with {model_name}...")

    # For now, embed one-by-one for simplicity
    for i, chunk in enumerate(chunks):
        resp = genai.embed_content(
            model=model_name,
            content=chunk,
        )
        emb = np.array(resp["embedding"], dtype="float32")
        embeddings.append(emb)

        if (i + 1) % 10 == 0:
            print(f"  Embedded {i + 1}/{len(chunks)} chunks")

    return np.vstack(embeddings)


# ---------- FAISS index building ----------

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS L2 index from embeddings.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index


# ---------- Main pipeline ----------

def main(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    configure_gemini()

    # 1) Load documents
    docs = load_documents(input_path)
    if not docs:
        print(f"No .pdf or .txt files found in {input_dir}")
        return

    # 2) Chunk documents and collect metadata
    all_chunks: List[str] = []
    chunk_metadata: List[Dict] = []

    for doc_id, text in docs:
        chunks = chunk_text(text)
        print(f"  Document {doc_id}: {len(chunks)} chunks")

        for chunk in chunks:
            chunk_id = f"{doc_id}_chunk_{len(all_chunks)}"
            all_chunks.append(chunk)
            chunk_metadata.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_text": chunk,
                }
            )

    print(f"Total chunks: {len(all_chunks)}")

    # 3) Embed all chunks
    embeddings = embed_chunks(all_chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    # 4) Build FAISS index
    index = build_faiss_index(embeddings)

    # 5) Save everything to disk
    index_path = output_path / "local_index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")

    emb_path = output_path / "chunk_embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings to {emb_path}")

    meta_path = output_path / "chunks.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved chunk metadata to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local RAG ingestion pipeline.")
    parser.add_argument("--input", type=str, default="docs", help="Folder with PDFs/TXT")
    parser.add_argument(
        "--output", type=str, default="local_store", help="Folder to save index + metadata"
    )
    args = parser.parse_args()
    main(args.input, args.output)
