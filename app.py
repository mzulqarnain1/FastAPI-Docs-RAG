from contextlib import asynccontextmanager

from fastapi import FastAPI

import rag
from models import AskRequest, AskResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.reranker.predict([("warmup", "warmup")])
    yield

app = FastAPI(title="FastAPI-Docs-RAG", lifespan=lifespan)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    answer, chunks = rag.answer(req.question, k=req.k)
    return AskResponse(answer=answer, sources=chunks)


@app.get("/healthz")
def healthz():
    return {"ok": True}
