import os
import json
import time
import ollama
import numpy as np
from numpy.linalg import norm
from langchain_ollama import OllamaEmbeddings

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


def create_vector_store(
    collection_name: str, modelname: str, chroma_db_file: str
) -> Chroma:
    """
    This function creates a ChromaDB vector store.
    Returns a client.
    """
    embeddings = OllamaEmbeddings(model=modelname)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_db_file,  # Where to save data locally, remove if not necessary
    )

    return vector_store


def parse_text_file_and_save_embeddings(filename, vector_store):
    """
    This function parses a text file and saves embeddings in a file.
    DO NOT USE FOR PRODUCTION. Use chroma-db or another vector store.
    """
    with open(filename, encoding="utf-8-sig") as f:
        lines = []
        for no, line in enumerate(f.readlines()):
            line = line.strip()

            if line:
                lines.append(line)
            elif lines and not line:
                document = Document(
                    page_content=" ".join(lines), metadata={"line_no": no}
                )
                vector_store.add_documents(documents=[document], ids=[str(no)])
                lines = []


def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        # FIXME:
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
    CHROMA_DB_FILE = "./embeddings/chroma_langchain_db"
    COLLECTION_NAME = "peter_pan"
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
    based on snippets of text provided in context. Answer only using the context provided
    , being as concise as possible. If you're unsure, just say that you don't know.
    Context: 
    """
    vstore = None
    filename = "peter_pan.txt"
    if not os.path.exists(CHROMA_DB_FILE):
        vstore = create_vector_store(COLLECTION_NAME, EMBEDDING_MODEL, CHROMA_DB_FILE)
        start = time.perf_counter()
        parse_text_file_and_save_embeddings(filename, vstore)
        print(
            f"Calculating embeddings using {EMBEDDING_MODEL}, took: {time.perf_counter() - start:.2f}s"
        )
    else:
        print(f"Using existing vector database for {COLLECTION_NAME}")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        vstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            create_collection_if_not_exists=False,
            persist_directory=CHROMA_DB_FILE,
        )

    prompt = input("")
    # prompt = "Who is the main villain of this story?"
    results = vstore.similarity_search(query=prompt, k=10)
    lines = []
    for doc in results:
        lines.append(doc.page_content)
        print(f"* {doc.page_content} [{doc.metadata}]")

    start = time.perf_counter()
    response = ollama.chat(
        model=INFERENCE_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT + " ".join(lines),
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
