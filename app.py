"""Streamlit UI:  streamlit run app.py

Ask a question -> see structured results, the retrieved chunks, and the exact
PDF page with the value highlighted (so every number can be verified at a glance).
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from specx.config import Settings
from specx.pipeline import Pipeline
from specx.render import render_page

st.set_page_config(page_title="Service Manual Spec Extractor", page_icon="🔧", layout="wide")

EXAMPLES = [
    "Torque for brake caliper bolts",
    "Brake fluid capacity",
    "Rear axle lubricant fill capacity",
    "Part number for shock absorber lower bolt",
    "Brake fluid specification",
    "Front brake disc minimum thickness",
    "Lower ball joint nut torque",
]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    pdf_path = st.text_input("PDF path", os.getenv("SPECX_PDF", "data/service-manual.pdf"))
    embed = st.selectbox("Embeddings", ["sentence-transformers", "tfidf", "openai"],
                         index=["sentence-transformers", "tfidf", "openai"].index(os.getenv("SPECX_EMBED_BACKEND", "sentence-transformers")))
    provider = st.selectbox("Extractor", ["auto", "anthropic", "openai", "none (rules only)"], index=0)
    mode = st.radio("Mode", ["auto", "llm", "rules"], horizontal=True,
                    help="auto = LLM if an API key is configured, otherwise deterministic table parsing")
    top_k = st.slider("Chunks to retrieve (k)", 3, 12, 6)
    strict = st.checkbox("Strict: drop values not found in the source text", value=False)


@st.cache_resource(show_spinner="Indexing the manual (first run only)...")
def get_pipeline(pdf: str, embed_backend: str, llm_provider: str) -> Pipeline:
    s = Settings()
    s.pdf_path, s.embed_backend = pdf, embed_backend
    s.llm_provider = "none" if llm_provider.startswith("none") else llm_provider
    pipe = Pipeline(s)
    pipe.ensure_ready()
    return pipe


st.title("🔧 Service Manual Spec Extractor")
st.caption("Hybrid retrieval (dense + BM25) over a structure-aware index of the manual, then structured extraction.")

if not os.path.exists(pdf_path):
    st.error(f"PDF not found: `{pdf_path}`. Put the manual at that path or change it in the sidebar.")
    st.stop()

pipe = get_pipeline(pdf_path, embed, provider)
pipe.s.top_k = top_k

if "q" not in st.session_state:
    st.session_state.q = EXAMPLES[0]
cols = st.columns(len(EXAMPLES))
for c, ex in zip(cols, EXAMPLES):
    if c.button(ex, width="stretch"):
        st.session_state.q = ex
question = st.text_input("Ask for a specification", key="q")

if question:
    res = pipe.query(question, mode=mode, strict=strict)
    for w in res.warnings:
        st.warning(w)
    st.markdown(f"**Intent:** `{res.intent}` &nbsp; **Extractor:** `{res.mode}` {('(' + res.llm_name + ')') if res.llm_name else ''}")

    if not res.records:
        st.info("No matching specification found in the retrieved sections.")
    else:
        df = pd.DataFrame([r.to_flat_dict() for r in res.records])
        st.dataframe(df, width="stretch", hide_index=True)
        c1, c2, _ = st.columns([1, 1, 4])
        c1.download_button("Download CSV", res.to_csv(), "specs.csv", "text/csv")
        c2.download_button("Download JSON", res.to_json(), "specs.json", "application/json")

        st.subheader("Verify against the PDF")
        labels = [f"{r.component[:60]} - {r.value or 'n/a'} {r.unit or ''} (p.{r.page})" for r in res.records]
        pick = st.selectbox("Result", range(len(labels)), format_func=lambda i: labels[i])
        r = res.records[pick]
        vals = [r.value or ""] + [q.value for q in r.alt_values]
        left, right = st.columns([1, 1])
        with left:
            st.json(json.loads(r.model_dump_json(exclude_none=True)))
        with right:
            if r.page:
                st.image(render_page(pdf_path, r.page, r.component, vals), caption=f"PDF page {r.page}")

    with st.expander(f"Retrieved chunks ({len(res.hits)})"):
        for h in res.hits:
            st.markdown(f"**{h.chunk.id}** · p.{h.chunk.page_start} · `{h.chunk.chunk_type}` · score {h.score:.4f} "
                        f"(dense rank {h.dense_rank}, BM25 rank {h.sparse_rank})")
            st.code(h.chunk.text, language="markdown")
