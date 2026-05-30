import pymupdf  # PyMuPDF — imported as pymupdf in newer versions
from fastapi import UploadFile
import re

CHUNK_SIZE = 500  # target characters per chunk
CHUNK_OVERLAP = 100  # characters shared between adjacent chunks


def clean_text(text: str) -> str:
    # Remove hyphenated line breaks
    text = re.sub(r"-\n", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes.
    Opens the PDF from memory without saving to disk."""
    with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
        pages = [page.get_text() for page in pdf]
        texts = "\n".join(pages)
        text = clean_text(texts)
        return text


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping chunks of approximately chunk_size characters.

    The overlap ensures that sentences or ideas split at a boundary
    still appear fully in at least one chunk — preventing context loss
    at the edges.

    Example with chunk_size=20, overlap=5:
      "The cat sat on the mat in the sun"
      Chunk 1: "The cat sat on the m"
      Chunk 2: "n the mat in the sun"  ← shares 5 chars with chunk 1
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        # Only keep chunks with meaningful content
        if chunk:
            chunks.append(chunk)

        # Move forward by chunk_size minus overlap
        # so the next chunk starts overlap characters before this one ended
        start += chunk_size - overlap

    return chunks


async def process_pdf(file: UploadFile) -> tuple[list[str], str]:
    """Full pipeline: read uploaded PDF → extract text → chunk it.

    Returns:
        chunks:        list of text chunks ready for embedding
        document_name: the original filename for metadata storage
    """
    contents = await file.read()
    text = extract_text_from_pdf(contents)
    chunks = chunk_text(text)
    document_name = file.filename

    return chunks, document_name
