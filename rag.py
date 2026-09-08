import os, sys

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

from models import RetrievedChunk

load_dotenv()
client = OpenAI()
EMBED_MODEL = os.environ["OPENAI_EMBED_MODEL"]
CHAT_MODEL = os.environ["OPENAI_CHAT_MODEL"]
DB = os.environ["DATABASE_URL"]

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

PROMPT = """Answer the question using ONLY the context below.
    Cite sources inline as [n], matching the numbered context blocks.
    If the context does not contain the answer, say "I don't know based on the provided documents." Do not answer from your own knowledge.

    Context:
    {context}

    Question: {question}
"""


def retrieve(question: str, k: int = 5) -> list[RetrievedChunk]:
    vec = str(
        client.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding
    )
    with psycopg.connect(DB) as conn:
        rows = conn.execute(
            """SELECT doc_name, chunk_idx, content,
                      1 - (embedding <=> %s::vector) AS cosine_sim
               FROM chunks
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (vec, vec, k),
        ).fetchall()
    return [
        RetrievedChunk(doc_name=d, chunk_idx=i, content=c, cosine_sim=s)
        for d, i, c, s in rows
    ]


def retrieve_and_rerank(
    question: str, k_retrieve: int = 20, k_final: int = 5
) -> list[RetrievedChunk]:
    candidates = retrieve(question, k_retrieve)
    scores = reranker.predict([(question, ch.content) for ch in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
    return [
        ch.model_copy(update={"rerank_score": float(s)}) for s, ch in ranked[:k_final]
    ]


def answer(question: str, k: int = 5) -> tuple[str, list[RetrievedChunk]]:
    chunks = retrieve_and_rerank(question, k_retrieve=20, k_final=k)
    context = "\n\n".join(
        f"[{i}] ({ch.doc_name} chunk {ch.chunk_idx})\n{ch.content}"
        for i, ch in enumerate(chunks, 1)
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(context=context, question=question),
            }
        ],
    )
    return resp.choices[0].message.content, chunks


if __name__ == "__main__":
    ans, rows = answer(sys.argv[1])
    print(ans, "\n\n--- retrieved (rerank order) ---")
    for doc, idx, _, sim, rr in rows:
        print(f"rerank {rr:+.2f}  cosine {sim:.3f}  {doc} #{idx}")
