from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    document_name: str
    chunks_stored: int
    message: str


class DocumentSummary(BaseModel):
    document_name: str
    chunk_count: int


class GetAllDocumentsResponse(BaseModel):
    result: int
    documents: list[DocumentSummary]


class DeleteDocumentResponse(BaseModel):
    document_name: str
    chunks_deleted: int
    message: str


class QueryRequest(BaseModel):
    question: str
    document_name: str | None = None
    top_k: int = 5


class QueryResponse(BaseModel):
    question: str
    answer: str
