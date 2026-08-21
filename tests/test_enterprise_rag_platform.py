"""
Unit and Integration Tests for Enterprise Document Intelligence & RAG Platform
"""

import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/enterprise_rag_platform")))

from vector_store import QdrantVectorStoreManager
from fastapi.testclient import TestClient
from server import app

class TestEnterpriseDocumentIntelligence(unittest.TestCase):

    def setUp(self):
        self.store = QdrantVectorStoreManager(collection_name="test_collection")
        self.client = TestClient(app)

    def test_semantic_chunking_and_indexing(self):
        sample_doc = "Docling provides deep layout analysis for enterprise PDF documents. " * 50
        res = self.store.index_document("doc_001", "Docling Architectural Guide", sample_doc)
        
        self.assertEqual(res["document_id"], "doc_001")
        self.assertGreater(res["chunks_indexed"], 0)
        
        stats = self.store.get_stats()
        self.assertEqual(stats["total_chunks"], res["chunks_indexed"])

    def test_vector_similarity_search(self):
        sample_doc = "Enterprise RAG pipeline uses Qdrant vector database for sub-10ms retrieval."
        self.store.index_document("doc_002", "RAG Pipeline Spec", sample_doc)
        
        matches = self.store.search_similarity("vector database retrieval", top_k=2)
        self.assertGreater(len(matches), 0)
        self.assertIn("chunk_id", matches[0])
        self.assertIn("score", matches[0])

    def test_fastapi_rag_endpoints(self):
        # Health check
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "healthy")

        # Document ingestion endpoint
        ingest_res = self.client.post("/api/v1/ingest", json={
            "title": "Financial Report 2026",
            "content": "Q4 revenue increased by 24% year-over-year reaching $12.4 million across AI product lines.",
            "document_type": "pdf"
        })
        self.assertEqual(ingest_res.status_code, 201)
        self.assertIn("document_id", ingest_res.json())

        # Vector search endpoint
        search_res = self.client.post("/api/v1/search", json={
            "query": "revenue growth percentage",
            "top_k": 3
        })
        self.assertEqual(search_res.status_code, 200)
        self.assertGreaterEqual(search_res.json()["results_count"], 1)

        # RAG query execution endpoint
        rag_res = self.client.post("/api/v1/rag/query", json={
            "question": "What was the Q4 revenue?",
            "top_k": 2
        })
        self.assertEqual(rag_res.status_code, 200)
        self.assertIn("answer", rag_res.json())

if __name__ == "__main__":
    unittest.main()
