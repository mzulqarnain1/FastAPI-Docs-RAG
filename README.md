# FastAPI-Docs-RAG

A small, self-contained RAG (Retrieval-Augmented Generation) system that answers questions about FastAPI using its official documentation as the only knowledge source — with citations, refusal on out-of-corpus questions, and an evaluation harness that measures retrieval quality, answer faithfulness, and abstention separately.

Built as a hands-on learning project: every component (chunking, vector search, reranking, evaluation) is implemented explicitly rather than hidden behind framework abstractions, so each design decision is inspectable and measurable.

## Architecture

```
Ingestion (ingest.py)

  docs/*.md
    | split on markdown headers, prepend breadcrumb
    v
  chunks
    | OpenAI text-embedding-3-small (1536d)
    v
  PostgreSQL + pgvector  --  HNSW index, cosine distance

Query (rag.py / app.py)

  question
    | embed with the same model
    v
  pgvector top-20  --  bi-encoder: cheap, wide net
    | cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
    v
  top-5 chunks  --  narrow judge: reads (query, chunk) jointly
    | context-only prompt
    v
  answer + [n] citations  --  fixed refusal phrase when unsupported
```

Key design decisions:

- **Structure-aware chunking.** Documents are split on markdown headers (not fixed character counts), and each chunk is prefixed with its header breadcrumb (e.g. `[Dependencies > What is "Dependency Injection"]`) before embedding. This bakes the section's topic into the chunk's vector, so chunks remain findable and understandable in isolation.
- **Two-stage retrieval (bi-encoder → cross-encoder).** pgvector's cosine search retrieves 20 candidates cheaply from precomputed embeddings; a locally-run cross-encoder then reads each (query, chunk) pair jointly and reorders them. The wide stage only needs recall; the narrow stage provides precision.
- **Raw SQL retrieval.** Vector search is a plain `ORDER BY embedding <=> query LIMIT k` query against pgvector — no vector-store wrapper — so the retrieval behavior is fully visible and tunable.
- **Grounded generation.** The prompt restricts the model to the retrieved context, requires inline `[n]` citations resolving to the returned sources, and mandates a fixed refusal phrase when the context doesn't contain the answer.
- **Typed pipeline boundaries.** Retrieval results are Pydantic `RetrievedChunk` models (`models.py`) carrying provenance from both stages (cosine similarity + rerank score); the same model doubles as the API response schema.

## Evaluation

The harness (`eval/`) scores the two failure modes of RAG independently — retrieval missing the right material vs. generation misusing it — plus abstention behavior:

| Script | Question it answers | Metric |
|---|---|---|
| `eval/retrieval.py` | Did we find the right document? | hit@5, MRR — cosine-only vs. reranked |
| `eval/answers.py` | Is the answer grounded in the retrieved context? | Faithfulness via LLM-as-judge (claim decomposition → per-claim support check) |
| `eval/abstention.py` | Does the system refuse out-of-corpus questions? | Abstention rate on questions with no valid source |

The gold set (`eval/gold.jsonl`) labels expected **documents** rather than chunk IDs, so labels survive re-chunking experiments. It includes questions phrased in user vocabulary that never matches the docs' terminology (the cases that separate retrieval configurations), and deliberately unanswerable questions.

**Results on the current gold set:**

| Configuration | hit@5 | MRR |
|---|---|---|
| Cosine similarity only | 1.00 | 0.80 |
| + cross-encoder reranking | 1.00 | **1.00** |

Reranking left recall unchanged but promoted the correct document to rank 1 on every question — the expected signature of a reranker (same net, better ordering). Mean answer faithfulness: **1.00**. Abstention: **2/2** out-of-corpus questions correctly refused.

*Honest caveat: the gold set is small (single-digit questions per category). At this size the harness reliably catches regressions and large effects, not small deltas. It runs in seconds, so it's executed on every pipeline change.*

## Running it

Requirements: Docker, Python 3.12+, an OpenAI API key.

```bash
# 1. Configure
cp .env.example .env        # set OPENAI_API_KEY, OPENAI_EMBED_MODEL, OPENAI_CHAT_MODEL

# 2. Database
docker compose up -d db
psql postgresql://rag:rag@localhost:5433/rag -f schema.sql

# 3. Ingest the corpus (chunks, embeds, and stores docs/*.md)
pip install -r requirements.txt
python ingest.py

# 4a. Ask from the CLI
python rag.py "How do I share database connection logic across multiple endpoints?"

# 4b. Or run the API
docker compose up --build
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "How do I make a query parameter required?"}' | jq
```

The Dockerfile bakes the cross-encoder weights into the image at build time, so containers start without downloading from Hugging Face and can run offline (aside from OpenAI calls). The model is warmed at startup via the FastAPI lifespan hook, so no request pays the load cost.

```bash
# Evaluation
python eval/retrieval.py
python eval/answers.py
python eval/abstention.py
```

## Project layout

```
├── app.py                  # FastAPI service: POST /ask -> answer + cited sources, GET /healthz
├── rag.py                  # retrieval (pgvector SQL), reranking, grounded generation; CLI entrypoint
├── ingest.py               # load docs/ -> header-aware chunking + breadcrumbs -> embed -> store
├── models.py               # RetrievedChunk (Pydantic): typed retrieval results + API schema
├── schema.sql              # pgvector extension, chunks table, HNSW index
├── eval/
│   ├── gold.jsonl          # labeled questions (expected docs; [] = should abstain)
│   ├── retrieval.py        # hit@5 + MRR, cosine vs reranked
│   ├── answers.py          # faithfulness (LLM-as-judge, claim decomposition)
│   └── abstention.py       # refusal check on unanswerable questions
├── docs/                   # corpus: 10 pages of FastAPI official documentation (markdown)
├── embedding_example.py    # standalone demo: embeddings + cosine similarity from first principles
├── Dockerfile              # bakes reranker weights at build time
└── docker-compose.yml      # Postgres (pgvector) + app
```

## Limitations & future work

- **Score-threshold pruning.** Rerank scores show a large gap between relevant and merely-topical chunks (~2.4 logits on inspection). Pruning chunks below a margin from the top score — instead of a fixed k=5 — would cut irrelevant context tokens sent to the LLM.
- **Reranker-based abstention.** The top rerank score is a cheap signal that the corpus likely contains no answer; refusing *before* generation when it falls below a calibrated threshold would complement the prompt-level refusal.
- **Harder abstention cases.** Current unanswerable questions are out-of-corpus by topic. The sharper test is questions whose topic is covered but whose specific answer isn't — where topical similarity maximally tempts the model to answer from pretraining.
- **Larger gold set.** Expand to the point where the harness discriminates smaller quality deltas; cross-check the hand-rolled faithfulness metric against RAGAs' implementation.
- **Ingestion is deliberately offline.** Corpus updates are a manual `python ingest.py` run; a demo doesn't need an ingestion API.