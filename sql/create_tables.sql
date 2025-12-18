CREATE TABLE IF NOT EXISTS `rag_demo_dataset.document_chunks` (
  chunk_id STRING NOT NULL,
  doc_id STRING NOT NULL,
  chunk_text STRING,
  metadata JSON,
  embedding ARRAY<FLOAT64>
);

CREATE TABLE IF NOT EXISTS `rag_demo_dataset.rag_query_logs` (
  ts TIMESTAMP,
  question STRING,
  top_chunk_ids ARRAY<STRING>,
  top_doc_ids ARRAY<STRING>,
  latency_ms INT64
);
