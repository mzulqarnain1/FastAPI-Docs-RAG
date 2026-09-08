import json

from rag import answer, client, CHAT_MODEL

JUDGE = """Given a context and an answer, do two things:
1. Break the answer into its atomic factual claims.
2. For each claim, decide if it is supported by the context alone (yes/no).
Respond with ONLY a JSON object: {{"claims": [{{"claim": "...", "supported": true}}]}}

Context:
{context}

Answer:
{answer}"""

def faithfulness(question: str) -> float | None:
    ans, rows = answer(question)
    context = "\n\n".join(r[2] for r in rows)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": JUDGE.format(context=context, answer=ans)}],
        response_format={"type": "json_object"},
    )
    claims = json.loads(resp.choices[0].message.content)["claims"]
    return sum(c["supported"] for c in claims) / len(claims) if claims else None

gold = [json.loads(l) for l in open("gold.jsonl") if l.strip()]
scores = []
for g in gold:
    if not g["expected_docs"]:
        continue  # abstention questions get checked separately
    f = faithfulness(g["q"])
    scores.append(f)
    print(f"{f:.2f}  {g['q']}")
print(f"\nmean faithfulness: {sum(scores)/len(scores):.2f}")