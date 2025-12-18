import os
import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "https://rag-api-406448352886.us-central1.run.app")

st.set_page_config(page_title="Cloud RAG Demo", layout="wide")
st.title("Cloud RAG Demo (Gemini + BigQuery)")
st.caption(f"API: {API_URL}")

question = st.text_input("Ask a question", value="What is this document about?")
k = st.slider("Top-k chunks", 1, 10, 5)
include_context = st.checkbox("Include retrieved context (debug)", value=False)

col1, col2 = st.columns([1, 1])
with col1:
    ask = st.button("Ask", type="primary")
with col2:
    st.write("")

if ask:
    with st.spinner("Querying the RAG API..."):
        r = requests.post(
            f"{API_URL}/query",
            json={"question": question, "k": k, "include_context": include_context},
            timeout=60,
        )
    if r.status_code != 200:
        st.error(f"Error {r.status_code}: {r.text}")
    else:
        data = r.json()

        st.subheader("Answer")
        st.write(data.get("answer", ""))

        st.subheader("Sources")
        for s in data.get("sources", []):
            st.markdown(
                f"- **doc_id**: `{s['doc_id']}` | **chunk_id**: `{s['chunk_id']}` | **cosine_sim**: `{s['cosine_sim']:.4f}`"
            )
            md = s.get("metadata")
            if md:
                st.code(md, language="json")
            if include_context and s.get("chunk_text"):
                st.text_area("chunk_text", s["chunk_text"], height=150)

        st.caption(f"Latency: {data.get('latency_ms')} ms")
