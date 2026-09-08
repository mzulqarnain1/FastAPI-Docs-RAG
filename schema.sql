CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id         bigserial PRIMARY KEY,
    doc_name   text NOT NULL,
    chunk_idx  int  NOT NULL,
    content    text NOT NULL,
    embedding  vector(1536) NOT NULL
);
