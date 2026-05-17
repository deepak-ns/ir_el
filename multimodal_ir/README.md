# Multimodal PDF Search Engine

A full-document information retrieval system that ingests PDFs, indexes them by page, and enables searchable text and figure-based retrieval using:

- Sentence-BERT embeddings for page text
- CLIP `openai/clip-vit-base-patch32` for figure image encoding and cross-modal retrieval
- BM25 exact token matching across pages
- FAISS vector search for both text and figure embeddings
- Streamlit UI for search, PDF upload, and figure-based retrieval
- FastAPI backend for REST search and ingestion

## Features

- Upload PDFs and index every page as a searchable unit
- Search by keyword/phrase and return ranked PDF results with best matching page previews
- Upload a figure image to retrieve papers with visually similar extracted figures
- Rebuild indexes from the corpus with text-only, figure-only, or combined modes
- Serve figure and page image previews through the API

## Repository Layout

- `api/` — FastAPI application and search API endpoints
- `ui/` — Streamlit frontend for search, figure retrieval, and PDF ingestion
- `scripts/` — CLI helpers for running the app and building indexes
- `src/` — core modules for config, database, embedding, retrieval, fusion, ingestion, and evaluation
- `configs/config.yaml` — central configuration for data paths, models, retrieval params, and ports
- `requirements.txt` — Python dependency list

## Installation

1. Create a Python virtual environment and activate it.

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Verify your working directory is the repository root:

```bash
cd multimodal_ir
```

## Running the System

```bash
## Load the demo data first and build index
python scripts/build_corpus.py --demo
python scripts/build_index.py
```

### Start the app and UI workflow

1. Run `python scripts/run.py`
2. Open `http://localhost:8501`
3. Use the **Upload PDF** tab to add a text-based PDF to the index
4. Use the **Text search** tab to query indexed content
5. Use the **Figure image upload** tab to search by visual similarity

### Search behavior

- Text search uses fused SBERT + BM25 ranking
- Figure search encodes uploaded images and retrieves similar extracted figures
- The UI displays the best-matching page and score breakdown for each result

## License

This repository is provided as-is for experimentation and research.
