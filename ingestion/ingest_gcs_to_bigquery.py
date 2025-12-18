import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pdfplumber
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

import google.generativeai as genai
from google.cloud import storage
from google.cloud import bigquery


# ---------------------------
# Config / setup
# ---------------------------

DEFAULT_EMBED_MODEL = "text-embedding-004"  # can swap later via --embed_model


def configure_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found. Put it in your .env file.")
    genai.configure(api_key=api_key)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


# ---------------------------
# GCS loading
# ---------------------------

def list_gcs_objects(bucket_name: str, prefix: str = "") -> List[str]:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket, prefix=prefix)
    return [b.name for b in blobs if not b.name.endswith("/")]


def download_blob_to_temp(bucket_name: str, blob_name: str, temp_dir: Path) -> Path:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    temp_dir.mkdir(parents=True, exist_ok=True)
    local_path = temp_dir / Path(blob_name).name
    blob.download_to_filename(str(local_path))
    return local_path


def load_text_from_pdf(path: Path) -> str:
    pages_text = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    return "\n".join(pages_text)


def load_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text(local_path: Path) -> str:
    suf = local_path.suffix.lower()
    if suf == ".pdf":
        return load_text_from_pdf(local_path)
    if suf == ".txt":
        return load_text_from_txt(local_path)
    return ""


# ---------------------------
# Chunking
# ---------------------------

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    step = max(1, chunk_size - overlap)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


# ---------------------------
# Embeddings (Gemini)
# ---------------------------

@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
def embed_text(text: str, embed_model: str) -> List[float]:
    # Returns a Python list of floats
    resp = genai.embed_content(model=embed_model, content=text)
    return resp["embedding"]


# ---------------------------
# BigQuery IO
# ---------------------------

def bq_client(project_id: Optional[str]) -> bigquery.Client:
    return bigquery.Client(project=project_id) if project_id else bigquery.Client()


def ensure_dataset_table_exists_note():
    # Created tables already. If user changes names, they'd need to create them.
    pass


def insert_rows_bq(
    client: bigquery.Client,
    table_fq: str,
    rows: List[Dict],
) -> None:
    """
    Uses streaming inserts. For bigger loads later,can switch to load jobs.
    """
    errors = client.insert_rows_json(table_fq, rows)
    if errors:
        # errors is a list of row-level errors
        raise RuntimeError(f"BigQuery insert_rows_json errors: {errors}")


# ---------------------------
# Main ingestion
# ---------------------------

def main(
    bucket: str,
    prefix: str,
    dataset: str,
    table: str,
    project_id: Optional[str],
    chunk_size: int,
    overlap: int,
    top_n_files: Optional[int],
    embed_model: str,
    batch_size: int,
):
    configure_gemini()
    bq = bq_client(project_id)

    project = bq.project
    table_fq = f"{project}.{dataset}.{table}"

    print(f"Using BigQuery table: {table_fq}")
    print(f"Listing GCS objects in gs://{bucket}/{prefix} ...")

    object_names = list_gcs_objects(bucket, prefix=prefix)
    object_names = [o for o in object_names if o.lower().endswith((".pdf", ".txt"))]

    if not object_names:
        print("No .pdf or .txt files found in that bucket/prefix.")
        return

    if top_n_files is not None:
        object_names = object_names[:top_n_files]

    print(f"Found {len(object_names)} ingestible files.")

    temp_dir = Path(".tmp_gcs_downloads")
    total_chunks = 0
    pending_rows: List[Dict] = []

    for idx, blob_name in enumerate(object_names, start=1):
        print(f"\n[{idx}/{len(object_names)}] Downloading: {blob_name}")
        local_path = download_blob_to_temp(bucket, blob_name, temp_dir)

        doc_id = Path(blob_name).stem
        text = extract_text(local_path)
        if not text.strip():
            print(f"  Skipping (no extractable text): {blob_name}")
            continue

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"  Extracted chars={len(text)} -> chunks={len(chunks)}")

        for j, chunk in enumerate(chunks):
            # deterministic-ish chunk_id so reruns are stable
            chunk_hash = sha1(f"{doc_id}:{j}:{chunk[:200]}")
            chunk_id = f"{doc_id}_{j}_{chunk_hash[:10]}"

            emb = embed_text(chunk, embed_model)

            row = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_text": chunk,
                "metadata": json.dumps({
                    "gcs_bucket": bucket,
                    "gcs_path": blob_name,
                    "chunk_index": j,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "embed_model": embed_model,
                }),
                "embedding": emb,  # ARRAY<FLOAT64> in BigQuery
            }
            pending_rows.append(row)
            total_chunks += 1

            if len(pending_rows) >= batch_size:
                print(f"  Inserting batch of {len(pending_rows)} rows into BigQuery...")
                insert_rows_bq(bq, table_fq, pending_rows)
                pending_rows = []

        # cleanup local temp file
        try:
            local_path.unlink(missing_ok=True)
        except Exception:
            pass

    if pending_rows:
        print(f"\nInserting final batch of {len(pending_rows)} rows into BigQuery...")
        insert_rows_bq(bq, table_fq, pending_rows)

    print(f"\n✅ Done. Total chunks inserted: {total_chunks}")
    print("Next: we’ll test BigQuery vector-like retrieval (Phase 2.3) and then build the Cloud Run API (Phase 3).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs/TXTs from GCS into BigQuery for RAG.")
    parser.add_argument("--bucket", required=True, help="GCS bucket name (no gs://)")
    parser.add_argument("--prefix", default="", help="Optional prefix/path inside bucket")
    parser.add_argument("--dataset", default="rag_demo_dataset", help="BigQuery dataset")
    parser.add_argument("--table", default="document_chunks", help="BigQuery table")
    parser.add_argument("--project_id", default=None, help="Override GCP project id (optional)")
    parser.add_argument("--chunk_size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--top_n_files", type=int, default=None, help="For testing (e.g., 1 or 2)")
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--batch_size", type=int, default=25, help="Streaming insert batch size")
    args = parser.parse_args()

    main(
        bucket=args.bucket,
        prefix=args.prefix,
        dataset=args.dataset,
        table=args.table,
        project_id=args.project_id,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_n_files=args.top_n_files,
        embed_model=args.embed_model,
        batch_size=args.batch_size,
    )
