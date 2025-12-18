import os
import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone


import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai
from google.cloud import bigquery
import json
from cachetools import TTLCache


# -------------------------
# Config
# -------------------------

DATASET = os.getenv("BQ_DATASET", "rag_demo_dataset")
TABLE = os.getenv("BQ_TABLE", "document_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
CACHE = TTLCache(maxsize=256, ttl=300)

def configure_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found. Set it in env or .env for local runs.")
    genai.configure(api_key=api_key)


# -------------------------
# Request/Response models
# -------------------------

class QueryRequest(BaseModel):
    question: str
    k: int = 5
    include_context: bool = False  # Optional "B" mode for debugging


class Source(BaseModel):
    doc_id: str
    chunk_id: str
    cosine_sim: float
    metadata: Optional[Any] = None
    chunk_text: Optional[str] = None  # only if include_context=true


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    latency_ms: int


# -------------------------
# Core: embed -> retrieve -> generate
# -------------------------

def embed_query(question: str) -> List[float]:
    resp = genai.embed_content(model=EMBED_MODEL, content=question)
    return resp["embedding"]


def retrieve_top_k(
    client: bigquery.Client,
    question_embedding: List[float],
    k: int,
) -> List[Dict[str, Any]]:
    table_fq = f"{client.project}.{DATASET}.{TABLE}"

    sql = f"""
    WITH
      q AS (SELECT @q_emb AS q_emb),
      scored AS (
        SELECT
          chunk_id,
          doc_id,
          chunk_text,
          metadata,
          (
            (SELECT SUM(e * q)
             FROM UNNEST(embedding) e WITH OFFSET i
             JOIN UNNEST((SELECT q_emb FROM q)) q WITH OFFSET j
             ON i = j
            )
            /
            (
              SQRT((SELECT SUM(e*e) FROM UNNEST(embedding) e))
              *
              SQRT((SELECT SUM(q*q) FROM UNNEST((SELECT q_emb FROM q)) q))
            )
          ) AS cosine_sim
        FROM `{table_fq}`
      )
    SELECT chunk_id, doc_id, chunk_text, metadata, cosine_sim
    FROM scored
    ORDER BY cosine_sim DESC
    LIMIT @k
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("q_emb", "FLOAT64", question_embedding),
            bigquery.ScalarQueryParameter("k", "INT64", k),
        ]
    )
    rows = client.query(sql, job_config=job_config).result()
    return [dict(r) for r in rows]


def generate_answer(question: str, retrieved: List[Dict[str, Any]]) -> str:
    # Build grounded context
    context_blocks = []
    for r in retrieved:
        context_blocks.append(
            f"[doc_id={r['doc_id']}, chunk_id={r['chunk_id']}]\n{r['chunk_text']}\n"
        )
    context = "\n\n".join(context_blocks)

    system = (
        "You are a helpful assistant. Answer using ONLY the provided context.\n"
        "If the context is insufficient, say you are not sure.\n"
        "Cite sources by including doc_id and chunk_id in your answer."
    )

    prompt = f"{system}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    model = genai.GenerativeModel(CHAT_MODEL)
    resp = model.generate_content(prompt)
    return resp.text


# -------------------------
# FastAPI app
# -------------------------
def log_query(
    client: bigquery.Client,
    question: str,
    retrieved: List[Dict[str, Any]],
    latency_ms: int,
) -> None:
    """
    Log each query to BigQuery for monitoring and analytics.
    """
    table_fq = f"{client.project}.{DATASET}.rag_query_logs"

    rows = [{
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "top_chunk_ids": [r["chunk_id"] for r in retrieved],
        "top_doc_ids": [r["doc_id"] for r in retrieved],
        "latency_ms": latency_ms,
    }]

    errors = client.insert_rows_json(table_fq, rows)
    if errors:
        # Don't fail the request if logging fails
        print("BigQuery log insert errors:", errors)

app = FastAPI(title="Cloud RAG API (Gemini + BigQuery)")

# Initialize clients once
configure_gemini()
bq_client = bigquery.Client()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):

    cache_key = json.dumps(
        {"question": req.question, "k": req.k, "include_context": req.include_context},
        sort_keys=True
    )
    cached = CACHE.get(cache_key)
    if cached is not None:
        return cached
    



    t0 = time.time()

    q_emb = embed_query(req.question)
    retrieved = retrieve_top_k(bq_client, q_emb, req.k)
    answer = generate_answer(req.question, retrieved)

    sources: List[Source] = []
    for r in retrieved:
        src = Source(
            doc_id=r["doc_id"],
            chunk_id=r["chunk_id"],
            cosine_sim=float(r["cosine_sim"]),
            metadata=r.get("metadata"),
            chunk_text=r["chunk_text"] if req.include_context else None,
        )
        sources.append(src)

    latency_ms = int((time.time() - t0) * 1000)
    log_query(bq_client, req.question, retrieved, latency_ms)
    resp = QueryResponse(answer=answer, sources=sources, latency_ms=latency_ms)
    resp_dict = resp.model_dump()
    CACHE[cache_key] = resp_dict
    return resp_dict


