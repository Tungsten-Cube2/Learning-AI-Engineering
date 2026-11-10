#You will need to store vectors. Chromadb makes it easier
from pathlib import Path
import ollama, chromadb
import os

doc = "test_data.txt"
query = "Can you summarize this transcript of a video?"
system_instructions = "You are a summarizer of video transcripts. Your best method of displaying summaries is by heavily utilizing bullet points. When summarizing, you point out advice given in the video transcript."
embed_model = "qwen3-embedding:0.6b"
response_model = "qwen2.5:7b"

def main(doc: str, embed_model: str, response_model: str, system_instructions: str, query: str):
    client = chromadb.Client()
    collection = client.create_collection(name="docs")
    p = Path(doc)
    a = p.read_text()

    response = ollama.embed(model=embed_model, input=a)['embeddings']
    print(response)
    collection.add(
        ids=["doc1"],
        embeddings=response,
        documents=[a]
    )

    query_embedding = ollama.embeddings(model=embed_model, prompt=query)["embedding"]

    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    retrieved_texts = [doc for doc in results["documents"][0]]

    print(results)

    context = "\n".join(retrieved_texts)
    prompt = f"Answer using the following context:{context}\n\nQuestion: {query}"
    resp = ollama.chat(model=response_model, messages=[{"role": "system", "content": system_instructions}, {"role": "user", "content": prompt}])
    print(resp["message"]["content"])

os.system('cls' if os.name == 'nt' else 'clear')
while True:
    print(f"""Document: {doc}
Query: {query}
System Instructions: {system_instructions}
Embedding Model: {embed_model}
Language Model: {response_model}""")
    execute = input("Do you want to run (y/n)? ")
    if execute == "y" or execute == "Y":
        main(doc=doc, embed_model=embed_model, response_model=response_model, system_instructions=system_instructions, query=query)
    elif execute.find("change ") == 0:
        if execute.find("change query to ") == 0:
            query = execute.split("change query to ")[1]
        elif execute.find("change doc to ") == 0:
            doc = execute.split("change doc to ")[1]
        elif execute.find("change system to ") == 0:
            system_instructions = execute.split("change system to ")[1]
        elif execute.find("change embed to ") == 0:
            embed_model = execute.split("change embed to ")[1]
        elif execute.find("change language to ") == 0:
            response_model = execute.split("change language to ")[1]
        os.system('cls' if os.name == 'nt' else 'clear')
    elif execute == "n" or execute == "N":
        exit()
