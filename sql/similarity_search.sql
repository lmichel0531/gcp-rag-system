-- BigQuery cosine similarity over ARRAY<FLOAT64> embeddings
-- Expects query embedding as @q_emb (ARRAY<FLOAT64>) and @k (INT64)

WITH
  q AS (SELECT @q_emb AS q_emb),
  scored AS (
    SELECT
      chunk_id,
      doc_id,
      chunk_text,
      metadata,
      (
        (SELECT SUM(e * qv)
         FROM UNNEST(embedding) e WITH OFFSET i
         JOIN UNNEST((SELECT q_emb FROM q)) qv WITH OFFSET j
         ON i = j
        )
        /
        (
          SQRT((SELECT SUM(e*e) FROM UNNEST(embedding) e))
          *
          SQRT((SELECT SUM(qv*qv) FROM UNNEST((SELECT q_emb FROM q)) qv))
        )
      ) AS cosine_sim
    FROM `rag_demo_dataset.document_chunks`
  )
SELECT *
FROM scored
ORDER BY cosine_sim DESC
LIMIT @k;
