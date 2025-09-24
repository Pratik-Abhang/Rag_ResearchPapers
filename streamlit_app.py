import os
import time
from typing import Tuple, Optional, List

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import NodeWithScore

import chromadb


PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "papers")
DATA_DIR = os.getenv("DATA_DIR", "Data")
DEFAULT_LLM = os.getenv("OLLAMA_LLM", "llama3.1")
DEFAULT_EMBED = os.getenv("OLLAMA_EMBED", "nomic-embed-text")


def configure_models(llm_model: str, embed_model: str, request_timeout: float = 120.0) -> None:
    Settings.llm = Ollama(model=llm_model, request_timeout=request_timeout)
    Settings.embed_model = OllamaEmbedding(model_name=embed_model)


def get_chroma_collection(persist_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(collection_name)
    return collection


def ensure_index(collection_name: str, persist_dir: str, data_dir: str) -> VectorStoreIndex:
    collection = get_chroma_collection(persist_dir, collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # If collection is empty, (re)index
    count = 0
    try:
        count = collection.count()
    except Exception:
        count = 0

    if count == 0:
        documents = SimpleDirectoryReader(data_dir).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=None,
            vector_store=vector_store,
            show_progress=True,
        )
        # Chroma persists automatically via PersistentClient
        return index

    # Reload from existing vector store
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index


def format_sources(nodes: List[NodeWithScore]) -> List[str]:
    items = []
    for i, node in enumerate(nodes, 1):
        meta = node.node.metadata or {}
        file_path = meta.get("file_path") or meta.get("filename") or meta.get("source") or "(unknown)"
        score = f"{node.score:.3f}" if node.score is not None else "-"
        items.append(f"{i}. {os.path.basename(file_path)} (score: {score})")
    return items


def save_uploaded_pdfs(files: List[UploadedFile], data_dir: str) -> List[str]:
    os.makedirs(data_dir, exist_ok=True)
    saved_paths: List[str] = []
    for f in files:
        name = os.path.basename(f.name)
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(data_dir, name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        saved_paths.append(path)
    return saved_paths


def main() -> None:
    st.set_page_config(page_title="RAG (LlamaIndex + Ollama + Chroma)", page_icon="📚", layout="wide")
    st.title("📚 RAG over PDFs (LlamaIndex + Ollama + Chroma)")

    with st.sidebar:
        st.header("Settings")
        llm_model = st.text_input("LLM model (Ollama)", value=DEFAULT_LLM, help="Run `ollama pull <model>` if not present.")
        embed_model = st.text_input("Embedding model (Ollama)", value=DEFAULT_EMBED, help="Run `ollama pull <model>` if not present.")
        top_k = st.slider("Top K", min_value=1, max_value=10, value=3, step=1)
        rebuild = st.button("Rebuild index from PDFs")

    # Configure models
    configure_models(llm_model, embed_model)

    # Build / load index
    status_placeholder = st.empty()
    try:
        if rebuild:
            # Force clear by recreating collection
            client = chromadb.PersistentClient(path=PERSIST_DIR)
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            status_placeholder.info("Rebuilding index from PDFs...")
        index = ensure_index(COLLECTION_NAME, PERSIST_DIR, DATA_DIR)
        status_placeholder.success("Index ready.")
    except Exception as e:
        status_placeholder.error(f"Failed to prepare index: {e}")
        st.stop()

    # Upload PDFs and auto-index
    st.subheader("Upload PDFs")
    uploaded = st.file_uploader("Add PDFs to your knowledge base", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        if st.button("Upload and index"):
            saved_paths = save_uploaded_pdfs(uploaded, DATA_DIR)
            if not saved_paths:
                st.warning("No valid PDF files uploaded.")
            else:
                with st.spinner("Indexing uploaded PDFs..."):
                    try:
                        # Load only newly saved PDFs and insert into existing index/vector store
                        new_docs = SimpleDirectoryReader(input_files=saved_paths).load_data()
                        index.insert(new_docs)
                        st.success(f"Indexed {len(new_docs)} new document(s).")
                    except Exception as e:
                        st.error(f"Failed to index uploaded PDFs: {e}")

    # Query UI
    st.subheader("Ask a question about your PDFs")
    question = st.text_input("Your question", value="Summarize the main ideas across these PDFs.")
    ask = st.button("Ask")

    if ask and question.strip():
        with st.spinner("Thinking..."):
            try:
                engine = index.as_query_engine(similarity_top_k=top_k)
                response = engine.query(question)
            except Exception as e:
                st.error(f"Query failed: {e}")
                return

        st.markdown("### Answer")
        st.write(str(response))

        # Sources
        if hasattr(response, "source_nodes") and response.source_nodes:
            st.markdown("### Sources")
            for line in format_sources(response.source_nodes):
                st.write(line)
        else:
            st.caption("No sources returned.")

    st.divider()
    st.caption(
        "Models served by Ollama. Embeddings and vectors persisted in Chroma at `./chroma_store`."
    )


if __name__ == "__main__":
    main()
