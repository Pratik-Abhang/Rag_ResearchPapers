# RAG Research Papers (LlamaIndex + Ollama + Chroma)

Minimal local RAG over PDFs using:
- LlamaIndex (LLM + embeddings via Ollama)
- Chroma (persistent vector store)
- Streamlit (simple UI)

## Prerequisites
- Python 3.12+ (3.13 tested)
- [Ollama](https://ollama.com) running locally
- Pull models once:
  - `ollama pull llama3.1`
  - `ollama pull nomic-embed-text`

## Setup
```bash
pip install -r requirements.txt
```

## Run the UI
```bash
streamlit run streamlit_app.py
```

## Usage
- Put PDFs in `Data/` (or upload from the UI)
- First run builds the index into `./chroma_store`
- Next runs load from Chroma; use sidebar to rebuild if needed

## Project Structure
```
.
├─ Data/                 # place PDFs here (empty in repo)
├─ chroma_store/         # local vector store (gitignored)
├─ streamlit_app.py      # UI app
├─ agent.ipynb           # notebook (optional)
├─ requirements.txt      # Python deps
├─ .streamlit/config.toml# theme config (dark)
├─ .gitignore            # repo hygiene
└─ README.md
```

## Notes
- No secrets or large data are checked in.
- The `Data/` folder is empty by default (`.gitkeep` present) – add your PDFs locally.
