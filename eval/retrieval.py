import json

from rag import retrieve, retrieve_and_rerank


def score(rows_docs: list[str], expected: list[str]) -> tuple[int, float]:
    hit, rr = 0, 0.0
    for rank, doc in enumerate(rows_docs, 1):
        if doc in expected:
            hit = 1
            rr = 1.0 / rank
            break
    return hit, rr

gold = [json.loads(l) for l in open("gold.jsonl") if l.strip()]
answerable = [g for g in gold if g["expected_docs"]]

for name, fn in [
    ("cosine only ", lambda q: [r[0] for r in retrieve(q, 5)]),
    ("reranked    ", lambda q: [r[0] for r in retrieve_and_rerank(q, 20, 5)]),
]:
    hits, mrrs = zip(*(score(fn(g["q"]), g["expected_docs"]) for g in answerable))
    print(f"{name} hit@5={sum(hits)/len(hits):.2f}  MRR={sum(mrrs)/len(mrrs):.2f}  (n={len(hits)})")
