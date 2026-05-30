from fastapi import FastAPI, UploadFile, HTTPException, status
from contextlib import asynccontextmanager

from app.services.document_processor import process_pdf
from app.services.embeddings import embed_text, embed_batch
from app.services.vector_store import (
    create_collection,
    upsert_vectors,
    search_vectors,
    list_documents,
    delete_document,
)
from app.services.llm import generate_answer

from app.schema import (
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
    GetAllDocumentsResponse,
    DeleteDocumentResponse,
)

COLLECTION_NAME = "documents"


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_collection(COLLECTION_NAME)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile) -> DocumentUploadResponse:
    """Upload a PDF, chunk it, embed the chunks, and store in Qdrant.

    Flow:
    1. Validate file type
    2. Extract text and chunk it
    3. Embed all chunks in one batch
    4. Store vectors + metadata in Qdrant
    5. Return confirmation
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted",
        )

    # 1. Extract text and chunk
    chunks, document_name = await process_pdf(file)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Could not extract any text from the PDF",
        )

    # 2. Embed all chunks in one batch — faster than one at a time
    vectors = embed_batch(chunks)

    # 3. Store in Qdrant
    upsert_vectors(
        collection_name=COLLECTION_NAME,
        vectors=vectors,
        chunks=chunks,
        document_name=document_name,
    )

    return DocumentUploadResponse(
        document_name=document_name,
        chunks_stored=len(chunks),
        message=f"Successfully processed and stored {len(chunks)} chunks from {document_name}",
    )


@app.get("/documents")
async def get_all_documents() -> GetAllDocumentsResponse:
    """List all documents that have been uploaded with their chunk counts."""
    documents = list_documents(COLLECTION_NAME)
    return GetAllDocumentsResponse(
        result=len(documents),
        documents=documents,
    )


@app.delete("/documents/{document_name}", status_code=status.HTTP_200_OK)
async def remove_document(document_name: str) -> DeleteDocumentResponse:
    """Delete a document and all its chunks from Qdrant."""
    chunks_deleted = delete_document(COLLECTION_NAME, document_name)

    if chunks_deleted == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_name}' not found",
        )

    return DeleteDocumentResponse(
        document_name=document_name,
        chunks_deleted=chunks_deleted,
        message=f"Successfully deleted {chunks_deleted} chunks from '{document_name}'",
    )


@app.post("/query")
async def query_documents(request: QueryRequest):
    """Search relevant chunks and generate an answer via Ollama.

    Flow:
    1. Embed the question
    2. Search Qdrant for the most similar chunks
    3. Pass chunks as context to Ollama
    4. Return the answer and the source chunks used
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty"
        )

    # 1. Embed the question using the same model used for documents
    query_vector = embed_text(request.question)

    # 2. Retrieve the most relevant chunks
    results = search_vectors(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=request.top_k,
        document_name=request.document_name,
    )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant documents found. Upload a document first.",
        )

    # 3. Extract just the text from results to pass to the LLM
    context_chunks = [r["chunk_text"] for r in results]

    # 4. Generate answer with Ollama
    answer = generate_answer(request.question, context_chunks)

    return QueryResponse(
        question=request.question,
        answer=answer,
        # sources=results,  # include sources so caller knows what was used
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
