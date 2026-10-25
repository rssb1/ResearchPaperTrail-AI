"""
ResearchPaperTrail AI — MVP
---------------------------
Upload research paper PDFs, ask grounded questions with page-level citations,
and generate a structured comparison table across papers.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pdf_processor import PDFExtractionError, build_chunks, extract_pages
from embeddings import EmbeddingError, embed_chunks, retrieve_top_chunks
from llm import LLMError, NO_ANSWER_PHRASE, answer_question, compare_paper

load_dotenv()

TOP_K = 6
MIN_SIMILARITY = 0.25
COMPARISON_QUERY = (
    "research question, objective, methodology, methods, datasets used, "
    "evaluation metrics, results, limitations and future work"
)

st.set_page_config(
    page_title="ResearchPaperTrail AI",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "papers" not in st.session_state:
    st.session_state.papers = {}  # title -> {"pages": int, "chunks": int}
if "chunks" not in st.session_state:
    st.session_state.chunks = []  # flat list of pdf_processor.Chunk, embedded
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"question", "answer", "sources"}
if "comparison_rows" not in st.session_state:
    st.session_state.comparison_rows = None

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .source-box {
        background-color: rgba(127,127,127,0.08);
        border-left: 3px solid #6c63ff;
        padding: 0.6rem 0.9rem;
        border-radius: 4px;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
    }
    .citation-tag {
        font-weight: 600;
        color: #6c63ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — upload & manage papers
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Papers")

    if not os.environ.get("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY is not set.\n\n"
            "Create a `.env` file (see `.env.example`) with your key, then restart the app.",
            icon="⚠️",
        )

    uploaded_files = st.file_uploader(
        "Upload research paper PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_clicked = st.button("Process papers", type="primary", use_container_width=True)

    if process_clicked:
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            progress = st.progress(0.0, text="Starting...")
            total = len(uploaded_files)
            for i, f in enumerate(uploaded_files):
                title = f.name
                if title in st.session_state.papers:
                    progress.progress((i + 1) / total, text=f"Skipping {title} (already processed)")
                    continue
                progress.progress((i + 1) / total, text=f"Reading {title}...")
                try:
                    file_bytes = f.read()
                    pages = extract_pages(file_bytes, title)
                    new_chunks = build_chunks(title, pages)
                    if not new_chunks:
                        st.warning(f"'{title}' produced no usable text chunks — skipped.")
                        continue
                    progress.progress((i + 1) / total, text=f"Embedding {title}...")
                    embed_chunks(new_chunks)
                    st.session_state.chunks.extend(new_chunks)
                    st.session_state.papers[title] = {
                        "pages": len(pages),
                        "chunks": len(new_chunks),
                    }
                except PDFExtractionError as e:
                    st.error(f"Could not process '{title}': {e}")
                except EmbeddingError as e:
                    st.error(f"Embedding failed for '{title}': {e}")
                except Exception as e:
                    st.error(f"Unexpected error processing '{title}': {e}")
            progress.empty()
            st.session_state.comparison_rows = None  # invalidate stale comparison
            st.rerun()

    st.divider()

    if st.session_state.papers:
        st.caption(f"{len(st.session_state.papers)} paper(s) loaded")
        for title, info in st.session_state.papers.items():
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(f"{info['pages']} pages · {info['chunks']} chunks")
        if st.button("Clear all papers", use_container_width=True):
            st.session_state.papers = {}
            st.session_state.chunks = []
            st.session_state.chat_history = []
            st.session_state.comparison_rows = None
            st.rerun()
    else:
        st.caption("No papers uploaded yet.")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("📄 ResearchPaperTrail AI")
st.caption("Ask grounded questions across your papers, with citations — or compare them side by side.")

tab_chat, tab_compare = st.tabs(["💬 Chat", "📊 Compare Papers"])

# --- Chat tab --------------------------------------------------------------
with tab_chat:
    if not st.session_state.papers:
        st.info("Upload and process at least one PDF in the sidebar to start asking questions.")
    else:
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant"):
                st.markdown(turn["answer"])
                if turn["sources"]:
                    with st.expander(f"Show {len(turn['sources'])} supporting passage(s)"):
                        for chunk, score in turn["sources"]:
                            st.markdown(
                                f'<div class="source-box">'
                                f'<span class="citation-tag">{chunk.paper_title} — p.{chunk.page_number}</span>'
                                f' (similarity {score:.2f})<br>{chunk.text}'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

        question = st.chat_input("Ask a question about your uploaded papers...")
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant passages..."):
                    try:
                        top_chunks = retrieve_top_chunks(
                            question, st.session_state.chunks, top_k=TOP_K, min_score=MIN_SIMILARITY
                        )
                    except EmbeddingError as e:
                        st.error(str(e))
                        top_chunks = None

                if top_chunks is not None:
                    with st.spinner("Generating answer..."):
                        try:
                            answer = answer_question(question, top_chunks)
                        except LLMError as e:
                            answer = f"⚠️ {e}"
                            top_chunks = []

                    st.markdown(answer)
                    if top_chunks and answer.strip() != NO_ANSWER_PHRASE:
                        with st.expander(f"Show {len(top_chunks)} supporting passage(s)"):
                            for chunk, score in top_chunks:
                                st.markdown(
                                    f'<div class="source-box">'
                                    f'<span class="citation-tag">{chunk.paper_title} — p.{chunk.page_number}</span>'
                                    f' (similarity {score:.2f})<br>{chunk.text}'
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                    st.session_state.chat_history.append(
                        {"question": question, "answer": answer, "sources": top_chunks}
                    )

# --- Compare tab -------------------------------------------------------------
with tab_compare:
    if not st.session_state.papers:
        st.info("Upload and process at least one PDF in the sidebar to compare papers.")
    else:
        st.write(
            "Generates a structured table — research question, methods, datasets, "
            "evaluation metrics, and limitations — for every uploaded paper. "
            "Fields not found in the paper are marked **Not specified**."
        )
        if st.button("🔍 Compare Papers", type="primary"):
            rows = []
            errors = []
            progress = st.progress(0.0, text="Starting comparison...")
            titles = list(st.session_state.papers.keys())
            for i, title in enumerate(titles):
                progress.progress((i + 1) / len(titles), text=f"Analyzing {title}...")
                paper_chunks = [c for c in st.session_state.chunks if c.paper_title == title]
                try:
                    top_chunks = retrieve_top_chunks(COMPARISON_QUERY, paper_chunks, top_k=10, min_score=MIN_SIMILARITY)
                    row = compare_paper(title, top_chunks)
                    rows.append(row)
                except (EmbeddingError, LLMError) as e:
                    errors.append(f"{title}: {e}")
            progress.empty()
            st.session_state.comparison_rows = rows
            for err in errors:
                st.error(err)

        if st.session_state.comparison_rows:
            df = pd.DataFrame(st.session_state.comparison_rows)
            df = df.rename(
                columns={
                    "paper_title": "Paper",
                    "research_question": "Research Question",
                    "methods": "Methods",
                    "datasets": "Datasets",
                    "evaluation_metrics": "Evaluation Metrics",
                    "limitations": "Limitations",
                }
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="paper_comparison.csv",
                mime="text/csv",
            )
