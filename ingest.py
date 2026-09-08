import os, glob

import psycopg
from dotenv import load_dotenv

from openai import OpenAI
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

load_dotenv()
client = OpenAI()
EMBED_MODEL = os.environ["OPENAI_EMBED_MODEL"]
DB = os.environ["DATABASE_URL"]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
)

sub_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=0, separators=["\n```", "\n\n", "\n", " "]
)


def chunk_file(text: str) -> list[str]:
    out = []
    for sec in md_splitter.split_text(text):
        breadcrumb = " > ".join(v for k, v in sorted(sec.metadata.items()))
        for piece in sub_splitter.split_text(sec.page_content):
            out.append(f"[{breadcrumb}]\n{piece}" if breadcrumb else piece)
    return out


def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main():
    with psycopg.connect(DB) as conn:
        conn.execute("TRUNCATE chunks")
        for path in sorted(glob.glob("docs/*.md")):
            name = os.path.basename(path)
            chunks = chunk_file(open(path).read())
            for start in range(0, len(chunks), 100):  # batch the API calls
                batch = chunks[start : start + 100]
                for i, (c, v) in enumerate(zip(batch, embed(batch)), start=start):
                    conn.execute(
                        "INSERT INTO chunks (doc_name, chunk_idx, content, embedding) "
                        "VALUES (%s, %s, %s, %s::vector)",
                        (name, i, c, str(v)),
                    )
            print(f"{name}: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
