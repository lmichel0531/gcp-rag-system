# Cloud-Native RAG System on Google Cloud (Gemini + BigQuery + Cloud Run)

Live demo:
- UI: https://rag-ui-406448352886.us-central1.run.app
- API: https://rag-api-406448352886.us-central1.run.app

## What this is
A production-style Retrieval-Augmented Generation (RAG) system:
- Documents ingested from Google Cloud Storage
- Embedded with Gemini
- Stored in BigQuery for semantic retrieval
- Served via a FastAPI API on Cloud Run
- Demo UI via Streamlit on Cloud Run

## Architecture
(Coming soon — see `diagrams/`)

## Repository layout
- `app/` FastAPI service (Cloud Run)
- `ingestion/` ingestion scripts (GCS -> embeddings -> BigQuery)
- `ui/` Streamlit frontend (Cloud Run)
- `sql/` BigQuery DDL and retrieval queries
- `docs/` small non-sensitive sample documents
- `diagrams/` architecture images

## Key features
- Semantic retrieval from BigQuery using cosine similarity over embeddings
- Grounded answers with citations (doc_id + chunk_id)
- Query/latency logging to BigQuery
- TTL caching for repeated queries (per Cloud Run instance)
- Secrets managed via Secret Manager
- Dockerized deployments (Cloud Run)

## Local run (API)
1) Create `.env` with `GEMINI_API_KEY=...`
2) Install dependencies:
   - `pip install -r requirements.txt`
3) Run:
   - `uvicorn app.main:app --reload --port 8000`

## Design Decisions
- Why BigQuery for vector storage? 

BigQuery provides a scalable, managed store for embeddings and metadata while allowing cosine similarity
to be computed directly in SQL. This avoids introducing additional vector databases while keeping retrieval
fully cloud-native and auditable.

- Why Cloud Run?  

Cloud Run enables stateless, autoscaling services with minimal operational overhead. Both the FastAPI API
and Streamlit UI are containerized and deployed independently, mirroring real world microservice patterns.

- Why cache at the API layer? 

A TTL-based in memory cache reduces latency for repeated queries and lowers embedding and generation costs.
This mirrors common production patterns before introducing shared caches like Redis.

- Why log queries to BigQuery? 

Query and latency logging enables monitoring, debugging, and future analysis of usage patterns and system
performance without coupling logging to application logic.

## Future Improvements
- Add authentication (IAP or OAuth) for protected deployments.
- Introduce a shared cache (e.g., Redis) for cross instance reuse.
- Support hybrid retrieval (keyword + vector search).
- Add document-level permissions and multi-tenant support.
- Add evaluation pipelines for retrieval and generation quality.

## API usage
```bash
curl -s -X POST "https://rag-api-406448352886.us-central1.run.app/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?","k":5}'
