import pytest
import io
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


COLLECTION_NAME = "documents"


@pytest.fixture
def sample_pdf_bytes():
    """Create a minimal valid PDF in memory without needing a real file."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50), "This is a test document about machine learning and AI systems."
    )
    page.insert_text((50, 80), "Neural networks are used to solve complex problems.")
    page.insert_text((50, 110), "Deep learning has revolutionized computer vision.")
    buffer = io.BytesIO()
    doc.save(buffer)
    doc.close()
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def sample_chunks():
    return [
        "This is a test document about machine learning and AI systems.",
        "Neural networks are used to solve complex problems.",
        "Deep learning has revolutionized computer vision.",
    ]


@pytest.fixture
def sample_vectors():
    """384-dimensional zero vectors matching all-MiniLM-L6-v2 output size."""
    return [[0.0] * 384 for _ in range(3)]


@pytest.fixture
def sample_search_results():
    return [
        {
            "chunk_text": "This is a test document about machine learning and AI systems.",
            "document_name": "test.pdf",
            "chunk_index": 0,
            "score": 0.92,
        },
        {
            "chunk_text": "Neural networks are used to solve complex problems.",
            "document_name": "test.pdf",
            "chunk_index": 1,
            "score": 0.87,
        },
    ]


# ─── Health Check ─────────────────────────────────────────────────────────────


class TestHealthCheck:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_healthy_status(self):
        response = client.get("/health")
        assert response.json() == {"status": "healthy"}


# ─── POST /documents ──────────────────────────────────────────────────────────


class TestUploadDocument:
    def test_valid_pdf_returns_201(
        self, sample_pdf_bytes, sample_chunks, sample_vectors
    ):
        with patch(
            "app.main.process_pdf", return_value=(sample_chunks, "test.pdf")
        ), patch("app.main.embed_batch", return_value=sample_vectors), patch(
            "app.main.upsert_vectors", return_value=None
        ):

            response = client.post(
                "/documents",
                files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            )
        assert response.status_code == 201

    def test_returns_document_name(
        self, sample_pdf_bytes, sample_chunks, sample_vectors
    ):
        with patch(
            "app.main.process_pdf", return_value=(sample_chunks, "test.pdf")
        ), patch("app.main.embed_batch", return_value=sample_vectors), patch(
            "app.main.upsert_vectors", return_value=None
        ):

            response = client.post(
                "/documents",
                files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            )
        assert response.json()["document_name"] == "test.pdf"

    def test_returns_correct_chunks_stored_count(
        self, sample_pdf_bytes, sample_chunks, sample_vectors
    ):
        with patch(
            "app.main.process_pdf", return_value=(sample_chunks, "test.pdf")
        ), patch("app.main.embed_batch", return_value=sample_vectors), patch(
            "app.main.upsert_vectors", return_value=None
        ):

            response = client.post(
                "/documents",
                files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            )
        assert response.json()["chunks_stored"] == len(sample_chunks)

    def test_returns_message_field(
        self, sample_pdf_bytes, sample_chunks, sample_vectors
    ):
        with patch(
            "app.main.process_pdf", return_value=(sample_chunks, "test.pdf")
        ), patch("app.main.embed_batch", return_value=sample_vectors), patch(
            "app.main.upsert_vectors", return_value=None
        ):

            response = client.post(
                "/documents",
                files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            )
        assert "message" in response.json()

    def test_message_contains_document_name(
        self, sample_pdf_bytes, sample_chunks, sample_vectors
    ):
        with patch(
            "app.main.process_pdf", return_value=(sample_chunks, "test.pdf")
        ), patch("app.main.embed_batch", return_value=sample_vectors), patch(
            "app.main.upsert_vectors", return_value=None
        ):

            response = client.post(
                "/documents",
                files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            )
        assert "test.pdf" in response.json()["message"]

    def test_non_pdf_returns_400(self, sample_pdf_bytes):
        response = client.post(
            "/documents",
            files={"file": ("test.txt", b"just some text", "text/plain")},
        )
        assert response.status_code == 400

    def test_non_pdf_returns_correct_error_message(self):
        response = client.post(
            "/documents",
            files={"file": ("test.txt", b"just some text", "text/plain")},
        )
        assert response.json()["detail"] == "Only PDF files are accepted"

    def test_jpeg_returns_400(self):
        response = client.post(
            "/documents",
            files={"file": ("photo.jpg", b"fake image bytes", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_empty_pdf_returns_422(self, sample_pdf_bytes):
        with patch("app.main.process_pdf", return_value=([], "empty.pdf")):
            response = client.post(
                "/documents",
                files={"file": ("empty.pdf", sample_pdf_bytes, "application/pdf")},
            )
        assert response.status_code == 422

    def test_empty_pdf_returns_correct_error_message(self, sample_pdf_bytes):
        with patch("app.main.process_pdf", return_value=([], "empty.pdf")):
            response = client.post(
                "/documents",
                files={"file": ("empty.pdf", sample_pdf_bytes, "application/pdf")},
            )
        assert "text" in response.json()["detail"].lower()

    def test_embed_batch_called_with_chunks(
        self, sample_pdf_bytes, sample_chunks, sample_vectors
    ):
        with patch(
            "app.main.process_pdf", return_value=(sample_chunks, "test.pdf")
        ), patch(
            "app.main.embed_batch", return_value=sample_vectors
        ) as mock_embed, patch(
            "app.main.upsert_vectors", return_value=None
        ):

            client.post(
                "/documents",
                files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            )
        mock_embed.assert_called_once_with(sample_chunks)

    def test_upsert_vectors_called_with_correct_args(
        self, sample_pdf_bytes, sample_chunks, sample_vectors
    ):
        with patch(
            "app.main.process_pdf", return_value=(sample_chunks, "test.pdf")
        ), patch("app.main.embed_batch", return_value=sample_vectors), patch(
            "app.main.upsert_vectors", return_value=None
        ) as mock_upsert:

            client.post(
                "/documents",
                files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            )
        mock_upsert.assert_called_once_with(
            collection_name="documents",
            vectors=sample_vectors,
            chunks=sample_chunks,
            document_name="test.pdf",
        )

    def test_missing_file_returns_422(self):
        response = client.post("/documents")
        assert response.status_code == 422


# ─── POST /query ──────────────────────────────────────────────────────────────


class TestQueryDocuments:
    def test_valid_query_returns_200(self, sample_vectors, sample_search_results):
        with patch("app.main.embed_text", return_value=sample_vectors[0]), patch(
            "app.main.search_vectors", return_value=sample_search_results
        ), patch(
            "app.main.generate_answer",
            return_value="Machine learning is a field of AI.",
        ):

            response = client.post(
                "/query",
                json={"question": "What is machine learning?"},
            )
        assert response.status_code == 200

    def test_returns_question_field(self, sample_vectors, sample_search_results):
        with patch("app.main.embed_text", return_value=sample_vectors[0]), patch(
            "app.main.search_vectors", return_value=sample_search_results
        ), patch(
            "app.main.generate_answer",
            return_value="Machine learning is a field of AI.",
        ):

            response = client.post(
                "/query",
                json={"question": "What is machine learning?"},
            )
        assert response.json()["question"] == "What is machine learning?"

    def test_returns_answer_field(self, sample_vectors, sample_search_results):
        with patch("app.main.embed_text", return_value=sample_vectors[0]), patch(
            "app.main.search_vectors", return_value=sample_search_results
        ), patch(
            "app.main.generate_answer",
            return_value="Machine learning is a field of AI.",
        ):

            response = client.post(
                "/query",
                json={"question": "What is machine learning?"},
            )
        assert response.json()["answer"] == "Machine learning is a field of AI."

    def test_empty_question_returns_400(self):
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 400

    def test_empty_question_returns_correct_error(self):
        response = client.post("/query", json={"question": ""})
        assert "empty" in response.json()["detail"].lower()

    def test_whitespace_only_question_returns_400(self):
        response = client.post("/query", json={"question": "    "})
        assert response.status_code == 400

    def test_missing_question_returns_422(self):
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_no_results_found_returns_404(self, sample_vectors):
        with patch("app.main.embed_text", return_value=sample_vectors[0]), patch(
            "app.main.search_vectors", return_value=[]
        ):

            response = client.post(
                "/query",
                json={"question": "What is quantum computing?"},
            )
        assert response.status_code == 404

    def test_no_results_returns_correct_error(self, sample_vectors):
        with patch("app.main.embed_text", return_value=sample_vectors[0]), patch(
            "app.main.search_vectors", return_value=[]
        ):

            response = client.post(
                "/query",
                json={"question": "What is quantum computing?"},
            )
        assert "upload" in response.json()["detail"].lower()

    def test_document_name_filter_passed_to_search(
        self, sample_vectors, sample_search_results
    ):
        with patch("app.main.embed_text", return_value=sample_vectors[0]), patch(
            "app.main.search_vectors", return_value=sample_search_results
        ) as mock_search, patch(
            "app.main.generate_answer", return_value="Some answer."
        ):

            client.post(
                "/query",
                json={
                    "question": "What is machine learning?",
                    "document_name": "test.pdf",
                },
            )
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["document_name"] == "test.pdf"

    def test_default_top_k_is_5(self, sample_vectors, sample_search_results):
        with patch("app.main.embed_text", return_value=sample_vectors[0]), patch(
            "app.main.search_vectors", return_value=sample_search_results
        ) as mock_search, patch(
            "app.main.generate_answer", return_value="Some answer."
        ):

            client.post(
                "/query",
                json={"question": "What is machine learning?"},
            )
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["limit"] == 5

    def test_custom_top_k_is_respected(self, sample_vectors, sample_search_results):
        with patch("app.main.embed_text", return_value=sample_vectors[0]), patch(
            "app.main.search_vectors", return_value=sample_search_results
        ) as mock_search, patch(
            "app.main.generate_answer", return_value="Some answer."
        ):

            client.post(
                "/query",
                json={"question": "What is machine learning?", "top_k": 3},
            )
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["limit"] == 3

    def test_embed_text_called_with_question(
        self, sample_vectors, sample_search_results
    ):
        with patch(
            "app.main.embed_text", return_value=sample_vectors[0]
        ) as mock_embed, patch(
            "app.main.search_vectors", return_value=sample_search_results
        ), patch(
            "app.main.generate_answer", return_value="Some answer."
        ):

            client.post(
                "/query",
                json={"question": "What is machine learning?"},
            )
        mock_embed.assert_called_once_with("What is machine learning?")

    def test_generate_answer_called_with_context_chunks(
        self, sample_vectors, sample_search_results
    ):
        with patch("app.main.embed_text", return_value=sample_vectors[0]), patch(
            "app.main.search_vectors", return_value=sample_search_results
        ), patch(
            "app.main.generate_answer", return_value="Some answer."
        ) as mock_generate:

            client.post(
                "/query",
                json={"question": "What is machine learning?"},
            )

        expected_chunks = [r["chunk_text"] for r in sample_search_results]
        mock_generate.assert_called_once_with(
            "What is machine learning?", expected_chunks
        )


# ─── Service Unit Tests ───────────────────────────────────────────────────────


class TestEmbeddingService:
    def test_embed_text_returns_list(self):
        from app.services.embeddings import embed_text

        result = embed_text("hello world")
        assert isinstance(result, list)

    def test_embed_text_returns_384_dimensions(self):
        from app.services.embeddings import embed_text

        result = embed_text("hello world")
        assert len(result) == 384

    def test_embed_text_returns_floats(self):
        from app.services.embeddings import embed_text

        result = embed_text("hello world")
        assert all(isinstance(v, float) for v in result)

    def test_embed_batch_returns_list_of_lists(self):
        from app.services.embeddings import embed_batch

        result = embed_batch(["hello", "world"])
        assert isinstance(result, list)
        assert all(isinstance(v, list) for v in result)

    def test_embed_batch_returns_correct_count(self):
        from app.services.embeddings import embed_batch

        texts = ["hello", "world", "machine learning"]
        result = embed_batch(texts)
        assert len(result) == len(texts)

    def test_embed_batch_each_vector_is_384_dimensions(self):
        from app.services.embeddings import embed_batch

        result = embed_batch(["hello", "world"])
        assert all(len(v) == 384 for v in result)

    def test_similar_texts_produce_similar_vectors(self):
        from app.services.embeddings import embed_text
        import numpy as np

        v1 = np.array(embed_text("machine learning"))
        v2 = np.array(embed_text("deep learning"))
        v3 = np.array(embed_text("banana smoothie recipe"))

        # cosine similarity
        sim_related = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        sim_unrelated = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))

        assert sim_related > sim_unrelated


class TestDocumentProcessor:
    def test_extract_text_returns_string(self, sample_pdf_bytes):
        from app.services.document_processor import extract_text_from_pdf

        result = extract_text_from_pdf(sample_pdf_bytes)
        assert isinstance(result, str)

    def test_extract_text_is_not_empty(self, sample_pdf_bytes):
        from app.services.document_processor import extract_text_from_pdf

        result = extract_text_from_pdf(sample_pdf_bytes)
        assert len(result) > 0

    def test_extract_text_contains_expected_content(self, sample_pdf_bytes):
        from app.services.document_processor import extract_text_from_pdf

        result = extract_text_from_pdf(sample_pdf_bytes)
        assert "machine learning" in result.lower()

    def test_chunk_text_returns_list(self):
        from app.services.document_processor import chunk_text

        result = chunk_text("This is some sample text for chunking.")
        assert isinstance(result, list)

    def test_chunk_text_produces_at_least_one_chunk(self):
        from app.services.document_processor import chunk_text

        result = chunk_text("This is some sample text.")
        assert len(result) >= 1

    def test_chunk_text_respects_chunk_size(self):
        from app.services.document_processor import chunk_text

        text = "a" * 2000
        chunks = chunk_text(text, chunk_size=500, overlap=0)
        assert all(len(c) <= 500 for c in chunks)

    def test_chunk_text_empty_string_returns_empty_list(self):
        from app.services.document_processor import chunk_text

        result = chunk_text("")
        assert result == []

    def test_chunk_text_overlap_means_chunks_share_content(self):
        from app.services.document_processor import chunk_text

        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        # With overlap, consecutive chunks share content
        # so total characters across all chunks > len(text)
        total_chars = sum(len(c) for c in chunks)
        assert total_chars > len(text)


class TestLLMService:
    def test_build_prompt_contains_question(self):
        from app.services.llm import build_prompt

        prompt = build_prompt("What is AI?", ["AI is artificial intelligence."])
        assert "What is AI?" in prompt

    def test_build_prompt_contains_context(self):
        from app.services.llm import build_prompt

        prompt = build_prompt("What is AI?", ["AI is artificial intelligence."])
        assert "AI is artificial intelligence." in prompt

    def test_build_prompt_contains_all_chunks(self):
        from app.services.llm import build_prompt

        chunks = ["Chunk one content.", "Chunk two content.", "Chunk three content."]
        prompt = build_prompt("Question?", chunks)
        assert all(chunk in prompt for chunk in chunks)

    def test_generate_answer_calls_ollama(self):
        from app.services.llm import generate_answer

        mock_response = MagicMock()
        mock_response.json = lambda: {"response": "This is the answer."}
        mock_response.raise_for_status = lambda: None

        with patch(
            "app.services.llm.httpx.post", return_value=mock_response
        ) as mock_post:
            result = generate_answer("What is AI?", ["AI is artificial intelligence."])

        mock_post.assert_called_once()
        assert result == "This is the answer."

    def test_generate_answer_returns_string(self):
        from app.services.llm import generate_answer

        mock_response = MagicMock()
        mock_response.json = lambda: {"response": "This is the answer."}
        mock_response.raise_for_status = lambda: None

        with patch("app.services.llm.httpx.post", return_value=mock_response):
            result = generate_answer("What is AI?", ["AI is artificial intelligence."])

        assert isinstance(result, str)


# ─── GET /documents ───────────────────────────────────────────────────────────


class TestGetAllDocuments:
    def test_returns_200(self):
        with patch("app.main.list_documents", return_value=[]):
            response = client.get("/documents")
        assert response.status_code == 200

    def test_returns_result_field(self):
        with patch("app.main.list_documents", return_value=[]):
            response = client.get("/documents")
        assert "result" in response.json()

    def test_returns_documents_field(self):
        with patch("app.main.list_documents", return_value=[]):
            response = client.get("/documents")
        assert "documents" in response.json()

    def test_returns_zero_result_when_empty(self):
        with patch("app.main.list_documents", return_value=[]):
            response = client.get("/documents")
        assert response.json()["result"] == 0

    def test_returns_empty_list_when_no_documents(self):
        with patch("app.main.list_documents", return_value=[]):
            response = client.get("/documents")
        assert response.json()["documents"] == []

    def test_returns_correct_result_count(self):
        mock_docs = [
            {"document_name": "doc1.pdf", "chunk_count": 10},
            {"document_name": "doc2.pdf", "chunk_count": 8},
        ]
        with patch("app.main.list_documents", return_value=mock_docs):
            response = client.get("/documents")
        assert response.json()["result"] == 2

    def test_returns_correct_document_names(self):
        mock_docs = [
            {"document_name": "doc1.pdf", "chunk_count": 10},
            {"document_name": "doc2.pdf", "chunk_count": 8},
        ]
        with patch("app.main.list_documents", return_value=mock_docs):
            response = client.get("/documents")
        names = [d["document_name"] for d in response.json()["documents"]]
        assert "doc1.pdf" in names
        assert "doc2.pdf" in names

    def test_documents_contain_chunk_count(self):
        mock_docs = [{"document_name": "doc1.pdf", "chunk_count": 10}]
        with patch("app.main.list_documents", return_value=mock_docs):
            response = client.get("/documents")
        assert "chunk_count" in response.json()["documents"][0]

    def test_result_matches_documents_list_length(self):
        mock_docs = [
            {"document_name": "doc1.pdf", "chunk_count": 10},
            {"document_name": "doc2.pdf", "chunk_count": 8},
            {"document_name": "doc3.pdf", "chunk_count": 5},
        ]
        with patch("app.main.list_documents", return_value=mock_docs):
            response = client.get("/documents")
        data = response.json()
        assert data["result"] == len(data["documents"])

    def test_documents_is_a_list(self):
        with patch("app.main.list_documents", return_value=[]):
            response = client.get("/documents")
        assert isinstance(response.json()["documents"], list)


# ─── DELETE /documents/{document_name} ───────────────────────────────────────


class TestDeleteDocument:
    def test_existing_document_returns_200(self):
        with patch("app.main.delete_document", return_value=10):
            response = client.delete("/documents/test.pdf")
        assert response.status_code == 200

    def test_returns_document_name(self):
        with patch("app.main.delete_document", return_value=10):
            response = client.delete("/documents/test.pdf")
        assert response.json()["document_name"] == "test.pdf"

    def test_returns_chunks_deleted_count(self):
        with patch("app.main.delete_document", return_value=10):
            response = client.delete("/documents/test.pdf")
        assert response.json()["chunks_deleted"] == 10

    def test_returns_message_field(self):
        with patch("app.main.delete_document", return_value=10):
            response = client.delete("/documents/test.pdf")
        assert "message" in response.json()

    def test_message_contains_document_name(self):
        with patch("app.main.delete_document", return_value=10):
            response = client.delete("/documents/test.pdf")
        assert "test.pdf" in response.json()["message"]

    def test_message_contains_chunks_deleted_count(self):
        with patch("app.main.delete_document", return_value=10):
            response = client.delete("/documents/test.pdf")
        assert "10" in response.json()["message"]

    def test_nonexistent_document_returns_404(self):
        with patch("app.main.delete_document", return_value=0):
            response = client.delete("/documents/nonexistent.pdf")
        assert response.status_code == 404

    def test_nonexistent_document_returns_correct_error(self):
        with patch("app.main.delete_document", return_value=0):
            response = client.delete("/documents/nonexistent.pdf")
        assert "not found" in response.json()["detail"].lower()

    def test_error_message_contains_document_name(self):
        with patch("app.main.delete_document", return_value=0):
            response = client.delete("/documents/missing.pdf")
        assert "missing.pdf" in response.json()["detail"]

    def test_delete_document_called_with_correct_name(self):
        with patch("app.main.delete_document", return_value=5) as mock_delete:
            client.delete("/documents/report.pdf")
        mock_delete.assert_called_once_with(COLLECTION_NAME, "report.pdf")

    def test_delete_called_with_collection_name(self):
        with patch("app.main.delete_document", return_value=5) as mock_delete:
            client.delete("/documents/report.pdf")
        args = mock_delete.call_args
        assert args[0][0] == COLLECTION_NAME
