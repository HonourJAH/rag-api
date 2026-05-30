from sentence_transformers import SentenceTransformer

# Load once at startup — not inside the function
# all-MiniLM-L6-v2 produces 384-dimensional vectors
# small, fast, and good enough for most RAG use cases
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str) -> list[float]:
    """Convert a string into a 384-dimensional embedding vector."""
    vector = model.encode(text)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Convert a list of strings into embedding vectors in one pass.
    Faster than calling embed_text in a loop because the model
    processes all texts in parallel as a single batch."""
    vectors = model.encode(texts)
    return vectors.tolist()
