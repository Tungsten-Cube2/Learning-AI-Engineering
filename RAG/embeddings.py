from pathlib import Path
import ollama, chromadb

p = Path("test_data.txt")
a = p.read_text()
print(a)
response = ollama.embed(
  model='qwen3-embedding:0.6b',
  input=a
)
vector = response['embeddings'][0]
print(vector)