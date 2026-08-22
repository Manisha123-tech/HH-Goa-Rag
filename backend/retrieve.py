import json
import faiss
from sentence_transformers import SentenceTransformer

INDEX_FILE = "data/faiss_index.bin"
DOCUMENTS_FILE = "data/documents.json"

TOP_K = 5

# Lazy-loaded resources
model = None
index = None
documents = None


def load_resources():
    global model, index, documents

    # Load only when retrieval is actually used
    if model is None:
        print("Loading embedding model...")

        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        print("Embedding model loaded.")

    if index is None:
        print("Loading FAISS index...")

        index = faiss.read_index(INDEX_FILE)

        print(f"FAISS index loaded with {index.ntotal} vectors.")

    if documents is None:
        print("Loading document metadata...")

        with open(
            DOCUMENTS_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            documents = json.load(file)

        print(f"Loaded {len(documents)} chunks.")


def retrieve(query, top_k=TOP_K):

    # Load resources only when a query arrives
    load_resources()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        document = documents[idx]

        results.append({
            "id": document["id"],
            "text": document["text"],
            "score": float(score),
            "parent_id": document["parent_id"],
            "chunk_index": document["chunk_index"],
            "strategy": document["chunking_strategy"]
        })

    return results


if __name__ == "__main__":

    print("\nHindi RAG Retrieval Test")
    print("Type 'exit' to stop.\n")

    while True:

        query = input(
            "Ask a question: "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = retrieve(query)

        print("\nTop Retrieved Context:\n")

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(f"--- Result {rank} ---")
            print(
                f"Score: {result['score']:.4f}"
            )
            print(
                f"Strategy: {result['strategy']}"
            )
            print(
                f"Text: {result['text']}\n"
            )