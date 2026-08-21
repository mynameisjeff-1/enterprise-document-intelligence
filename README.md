# Enterprise Document Intelligence & RAG Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

An enterprise-grade platform for document layout parsing, structured data extraction (tables, formulas, metadata), semantic chunking, and high-performance Retrieval-Augmented Generation (RAG) powered by Qdrant vector search.

---

## Architecture Overview

```
+------------------+     Document Ingestion     +-----------------------------------------+
| Document Source  | -------------------------> | Docling Layout & Structure Parser       |
| (PDF, Docx, HTML)|                            | (Extracts Tables, Metadata & Formats)  |
+------------------+                            +-----------------------------------------+
                                                                     |
                                                                     v
+------------------+     Vector Indexing        +-----------------------------------------+
| Qdrant Vector DB | <------------------------- | Semantic Window Chunker & Embedder      |
| (Sub-10ms Search)|                            | (Overlap Windows & Metadata Injection)  |
+------------------+                            +-----------------------------------------+
         ^
         |
         | Vector Search / RAG Context
         v
+-----------------------------------------------------------------------------------------+
|                      FastAPI Document Intelligence & RAG API                            |
|             (/api/v1/ingest | /api/v1/extract | /api/v1/search | /api/v1/rag/query)        |
+-----------------------------------------------------------------------------------------+
```

---

## Core Features

- **Advanced Document Structure Parsing**: Ingests complex PDFs, Microsoft Word files, PowerPoint presentations, and HTML documents to extract structured layout trees, tables, and mathematical formulas.
- **Semantic Window Chunking**: Context-aware chunking pipeline with configurable token limits and overlap windows to preserve document continuity.
- **Qdrant Vector Database Integration**: High-density vector indexing and similarity search with metadata filtering.
- **FastAPI RAG Orchestration**: Clean REST interface serving document ingestion (`/ingest`), layout extraction (`/extract`), semantic vector search (`/search`), and RAG query execution (`/rag/query`).
- **Production Infrastructure**: Built-in Docker and Docker Compose environment for instant orchestration of Docling processing containers and Qdrant DB.

---

## Technology Stack

- **Core Framework**: Python 3.10+, Docling Core
- **API Engine**: FastAPI, Pydantic v2, Uvicorn
- **Vector Database**: Qdrant Vector Database
- **Text & Math Processing**: RapidOCR, PyMuPDF, Pandoc
- **Infrastructure**: Docker, Docker Compose

---

## Quickstart & Setup

### 1. Installation
```bash
git clone https://github.com/mynameisjeff-1/enterprise-document-intelligence.git
cd enterprise-document-intelligence

cp .env.example .env
```

### 2. Run with Docker Compose (Recommended)
```bash
cd examples/enterprise_rag_platform
docker-compose up --build
```
Access the interactive OpenAPI interface at `http://localhost:8000/docs`.

---

## API Documentation & Usage

### 1. Ingest & Index Document
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Quarterly Financial Analysis 2026",
       "content": "Q4 operating revenue reached $48.2 million driven by enterprise cloud AI adoption across North America.",
       "document_type": "pdf"
     }'
```

### 2. Extract Document Layout & Structure
```bash
curl -X POST "http://localhost:8000/api/v1/extract" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Architecture Overview",
       "content": "# Core Engine\n| Service | Port |\n| Docling | 8000 |"
     }'
```

### 3. Vector Similarity Search
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "quarterly revenue breakdown", "top_k": 3}'
```

### 4. Execute RAG Query
```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What was the operating revenue in Q4?", "top_k": 3}'
```

---

## Automated Tests

Run the platform integration test suite:
```bash
/home/hamza/ide/portfolio-projects/venv/bin/python -m unittest tests/test_enterprise_rag_platform.py
```

---
