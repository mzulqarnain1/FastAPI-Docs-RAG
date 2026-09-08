# scratch_embed.py
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

texts = [
    "How do I declare a query parameter?",
    "Function arguments that aren't part of the path are interpreted as query parameters.",
    "The pasta at that restaurant was badly overcooked.",
]
r = OpenAI().embeddings.create(model="text-embedding-3-small", input=texts)
v = [np.array(d.embedding) for d in r.data]
print("related  :", v[0] @ v[1])
print("unrelated:", v[0] @ v[2])
