import os
import argparse
from typing import List, Dict

import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import bigquery


def configure_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")
    genai.configure(api_key=api_key)


def embed_query(question: str, embed_model: str = "text-embedding-004") -> List[float]:
    resp = genai.embed_content(model=embed_model, content=question)
    return resp["embedding"]


def main():
    parser = argparse.ArgumentParser(description="Retrieve top-k chunks from BigQuery for a question.")
    parser.add_argument("--question", required=True, type=str)
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--dataset", default="rag_demo_dataset", type=str)
    parser.add_argument("--table", default="document_chunks", type=str)
    args = parser.parse_args()

    configure_gemini()
    client = bigquery.Client()

    query_emb = embed_query(args.question)
    table_fq = f"{client.project}.{args.dataset}.{args.table}"

    # Cosine similarity = dot(a,b) / (||a||*||b||)
    # We compute it in SQL by:
    #  - UNNEST embeddings with offsets
    #  - join on offset to multiply corresponding dims
    #  - compute norms
    sql = f"""
    WITH
      q AS (
        SELECT @q_emb AS q_emb
      ),
      scored AS (
        SELECT
          chunk_id,
          doc_id,
          chunk_text,
          metadata,
          (
            -- cosine similarity
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
    SELECT
      chunk_id, doc_id, cosine_sim, chunk_text, metadata
    FROM scored
    ORDER BY cosine_sim DESC
    LIMIT @k
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("q_emb", "FLOAT64", query_emb),
            bigquery.ScalarQueryParameter("k", "INT64", args.k),
        ]
    )

    print(f"\nQuestion: {args.question}\n")
    results = client.query(sql, job_config=job_config).result()

    print(f"Top-{args.k} chunks:\n")
    for row in results:
        print(f"- doc_id={row.doc_id}, chunk_id={row.chunk_id}, cosine_sim={row.cosine_sim:.4f}")
        preview = row.chunk_text[:250].replace("\n", " ")
        print(f"  preview: {preview}...")
        print()

if __name__ == "__main__":
    main()
