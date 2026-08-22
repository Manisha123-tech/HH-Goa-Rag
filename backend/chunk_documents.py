import json
import os
import re

INPUT_FILE = "data/passages.jsonl"
OUTPUT_FILE = "data/chunks.jsonl"

# Chunk size settings
SHORT_THRESHOLD = 500
MEDIUM_CHUNK_SIZE = 800
LONG_CHUNK_SIZE = 1200
OVERLAP = 200


def split_sentences(text):
    """
    Split Hindi/English text into sentences.
    Supports Hindi danda (।), . ? and !
    """
    sentences = re.split(r'(?<=[।.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def create_sentence_chunks(text, chunk_size, overlap_chars):
    """
    Create sentence-aware chunks with character overlap.
    """
    sentences = split_sentences(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += (" " if current_chunk else "") + sentence

        else:
            if current_chunk:
                chunks.append(current_chunk)

            # Add overlap from previous chunk
            overlap_text = current_chunk[-overlap_chars:]

            current_chunk = overlap_text + " " + sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def adaptive_chunk(text):
    """
    Choose chunking strategy based on passage length.
    """

    text_length = len(text)

    # Strategy 1: Short passages remain unchanged
    if text_length <= SHORT_THRESHOLD:
        return [text], "whole_passage"

    # Strategy 2: Medium passages
    elif text_length <= 2000:
        chunks = create_sentence_chunks(
            text,
            MEDIUM_CHUNK_SIZE,
            OVERLAP
        )
        return chunks, "sentence_overlap"

    # Strategy 3: Long passages
    else:
        chunks = create_sentence_chunks(
            text,
            LONG_CHUNK_SIZE,
            OVERLAP
        )
        return chunks, "large_sentence_overlap"


print("Loading passages...")

documents = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        documents.append(json.loads(line))

print(f"Loaded {len(documents)} passages.")

os.makedirs("data", exist_ok=True)

total_chunks = 0
strategy_counts = {
    "whole_passage": 0,
    "sentence_overlap": 0,
    "large_sentence_overlap": 0
}

print("\nCreating adaptive chunks...\n")

with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

    for document in documents:

        parent_id = document["id"]
        text = document["text"].strip()

        chunks, strategy = adaptive_chunk(text)

        strategy_counts[strategy] += 1

        for chunk_index, chunk_text in enumerate(chunks):

            chunk_document = {
                "id": f"doc_{parent_id}_chunk_{chunk_index}",
                "parent_id": parent_id,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "chunking_strategy": strategy,
                "original_length": len(text)
            }

            output.write(
                json.dumps(
                    chunk_document,
                    ensure_ascii=False
                ) + "\n"
            )

            total_chunks += 1

        if parent_id % 500 == 0:
            print(
                f"Processed document {parent_id} | "
                f"Total chunks: {total_chunks}"
            )


print("\nChunking completed successfully!")

print(f"Original passages: {len(documents)}")
print(f"Total chunks created: {total_chunks}")

print("\nChunking strategies used:")

for strategy, count in strategy_counts.items():
    print(f"{strategy}: {count}")

print(f"\nOutput saved to: {OUTPUT_FILE}")