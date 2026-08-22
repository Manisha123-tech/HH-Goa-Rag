import json
import faiss
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/chunks.jsonl"
INDEX_FILE = "data/faiss_index.bin"
DOCUMENTS_FILE = "data/documents.json"

print("Loading chunks...")

documents = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        documents.append(json.loads(line))

print(f"Loaded {len(documents)} chunks.")

print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

texts = [doc["text"] for doc in documents]

print("\nCreating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(f"\nEmbedding shape: {embeddings.shape}")

dimension = embeddings.shape[1]

print(f"Vector dimension: {dimension}")

print("\nCreating FAISS index...")

# Inner Product works as cosine similarity
# because embeddings are already normalized
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"Vectors added to index: {index.ntotal}")

print("\nSaving FAISS index...")

faiss.write_index(index, INDEX_FILE)

print("Saving chunk metadata...")

with open(DOCUMENTS_FILE, "w", encoding="utf-8") as file:
    json.dump(documents, file, ensure_ascii=False, indent=2)

print("\nEmbedding process completed successfully!")

print(f"FAISS index saved to: {INDEX_FILE}")
print(f"Documents saved to: {DOCUMENTS_FILE}")