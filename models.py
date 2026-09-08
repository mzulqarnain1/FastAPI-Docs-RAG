from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    doc_name: str
    chunk_idx: int
    content: str
    cosine_sim: float
    rerank_score: float | None = None


class AskRequest(BaseModel):
    question: str
    k: int = 5


class AskResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
