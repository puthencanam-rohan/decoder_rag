import os
import json
import time
import ollama
import numpy as np
from numpy.linalg import norm


def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            # print(line)
            line = line.strip()

            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append(" ".join(buffer))
                buffer = []

            # print(buffer)

        if len(buffer):
            paragraphs.append(" ".join(buffer))
        return paragraphs


def get_embeddings(filename, modelname, chunks):
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings

    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]

    save_embeddings(filename, embeddings)
    return embeddings


def save_embeddings(filename, embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")

    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)


def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False

    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)


def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)


def main():
    EMBEDDING_MODEL = "nomic-embed-text:latest"
    # INFERENCE_MODEL = "Llama3.2-3B:latest"
    INFERENCE_MODEL = "Mistral7BI"
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
    based on snippets of text provided in context. Answer only using the context provided
    , being as concise as possible. If you're unsure, just say that you don't know.
    Context: 
    """
    filename = "peter_pan.txt"
    paragraphs = parse_file(filename)
    start = time.perf_counter()
    # embeddings = get_embeddings(filename, EMBEDDING_MODEL, paragraphs)
    # embeddings = get_embeddings(filename, INFERENCE_MODEL, paragraphs[5:90])
    embeddings = get_embeddings(filename, INFERENCE_MODEL, paragraphs)
    print(
        f"Calculating embeddings using {INFERENCE_MODEL}, took: {time.perf_counter() - start:.2f}s"
    )
    # print(paragraphs[:10])
    print(len(embeddings))

    prompt = input(">>: ")
    # prompt_embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=prompt)[
    prompt_embedding = ollama.embeddings(model=INFERENCE_MODEL, prompt=prompt)[
        "embedding"
    ]

    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]
    for item in most_similar_chunks:
        print(item[0], paragraphs[item[1]])

    start = time.perf_counter()
    response = ollama.chat(
        model=INFERENCE_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + " ".join(paragraphs[item[1]] for item in most_similar_chunks),
            },
            {"role": "user", "content": prompt},
        ],
    )
    print(
        f"Computed response from {INFERENCE_MODEL} in {time.perf_counter()-start: .2f}s"
    )
    print(response["message"]["content"])


if __name__ == "__main__":
    main()
