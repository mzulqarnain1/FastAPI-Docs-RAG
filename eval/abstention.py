import json, sys

sys.path.append(".")

from rag import answer

REFUSAL_MARKER = "i don't know based on the provided documents"

gold = [json.loads(l) for l in open("gold.jsonl") if l.strip()]
unanswerable = [g for g in gold if not g["expected_docs"]]

correct = 0
for g in unanswerable:
    ans, _ = answer(g["q"])
    abstained = REFUSAL_MARKER in ans.lower()
    correct += abstained
    print(f"{'ABSTAIN ✓' if abstained else 'ANSWERED ✗'}  {g['q']}")
print(f"\nabstention rate: {correct}/{len(unanswerable)}")