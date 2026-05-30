from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import uuid

import os

client = QdrantClient(
    host=os.getenv("QDRANT_HOST"),
    port=int(os.getenv("QDRANT_PORT")),
)

# Must match the output size of all-MiniLM-L6-v2
VECTOR_SIZE = 384


def create_collection(collection_name: str) -> None:
    """Create a Qdrant collection if it doesn't already exist.
    Skips creation silently if the collection is already there —
    safe to call on every app startup."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE, distance=Distance.COSINE  # best for text similarity
            ),
        )


def upsert_vectors(
    collection_name: str,
    vectors: list[list[float]],
    chunks: list[str],
    document_name: str,
) -> None:
    """Store a list of chunk embeddings with their metadata.

    Each point stores:
    - vector:         the embedding of the chunk
    - chunk_text:     the original text — returned when this chunk is retrieved
    - document_name:  which document this chunk came from
    - chunk_index:    position of this chunk within the document
    """
    points = [
        PointStruct(
            id=str(uuid.uuid4()),  # unique ID per chunk
            vector=vector,
            payload={
                "chunk_text": chunk,
                "document_name": document_name,
                "chunk_index": i,
            },
        )
        for i, (vector, chunk) in enumerate(zip(vectors, chunks))
    ]
    client.upsert(collection_name=collection_name, points=points)


def search_vectors(
    collection_name: str,
    query_vector: list[float],
    limit: int = 5,
    document_name: str | None = None,
) -> list[dict]:
    """Search for the most similar chunks to the query vector.

    If document_name is provided, search is scoped to that document only.
    Returns a list of dicts containing chunk text, source document,
    chunk index, and similarity score.
    """
    # Optionally filter by document name
    query_filter = None
    if document_name:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_name", match=MatchValue(value=document_name)
                )
            ]
        )

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
    )

    return [
        {
            "chunk_text": result.payload["chunk_text"],
            "document_name": result.payload["document_name"],
            "chunk_index": result.payload["chunk_index"],
            "score": result.score,
        }
        for result in results.points
    ]


def list_documents(collection_name: str) -> list[dict]:
    """Return a list of all unique documents stored in the collection.

    Scrolls through all points in the collection and extracts unique
    document names along with their chunk counts and first upload metadata.
    scroll() is used instead of search() because we're not querying by
    similarity — we just want all stored records.
    """
    all_points = []
    offset = None

    # Qdrant paginates large collections — loop until all points are fetched
    while True:
        results, next_offset = client.scroll(
            collection_name=collection_name,
            limit=100,  # fetch 100 points at a time
            offset=offset,
            with_payload=True,
            with_vectors=False,  # we don't need the vectors — just metadata
        )
        all_points.extend(results)

        if next_offset is None:
            break
        offset = next_offset

    # Group points by document name and count chunks per document
    documents = {}
    for point in all_points:
        name = point.payload.get("document_name")
        if name not in documents:
            documents[name] = {"document_name": name, "chunk_count": 0}
        documents[name]["chunk_count"] += 1

    return list(documents.values())


def delete_document(collection_name: str, document_name: str) -> int:
    """Delete all points belonging to a specific document.

    Uses a payload filter to find and delete only the points where
    document_name matches — leaving all other documents untouched.
    Returns the number of points deleted.
    """
    # First count how many chunks exist for this document
    # so we can return a meaningful count and detect if it existed at all
    existing = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="document_name", match=MatchValue(value=document_name)
                )
            ]
        ),
        with_payload=False,
        with_vectors=False,
        limit=10000,
    )
    points_to_delete = existing[0]

    if not points_to_delete:
        return 0  # document not found — caller handles the 404

    # Delete all points matching this document name
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_name", match=MatchValue(value=document_name)
                )
            ]
        ),
    )

    return len(points_to_delete)
