from datasets import load_dataset
import json
import os

DATA_URL = (
    "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/"
    "resolve/main/validation/hinval.parquet"
)

OUTPUT_FILE = "data/passages.jsonl"

MAX_EXAMPLES = 500

print("Loading MSMARCO-XI Hindi validation dataset...")

dataset = load_dataset(
    "parquet",
    data_files={"validation": DATA_URL},
    split="validation",
    streaming=True
)

print("Dataset stream ready.")
print(f"Processing first {MAX_EXAMPLES} examples...\n")

# Create data folder if it does not exist
os.makedirs("data", exist_ok=True)

unique_passages = set()
processed_examples = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    for example in dataset:

        processed_examples += 1

        passages = example["passages"]["Translated_passages"]

        for passage in passages:

            # Clean the passage
            passage = passage.strip()

            # Skip empty or duplicate passages
            if passage and passage not in unique_passages:

                unique_passages.add(passage)

                document = {
                    "id": len(unique_passages),
                    "text": passage
                }

                file.write(
                    json.dumps(document, ensure_ascii=False) + "\n"
                )

        if processed_examples >= MAX_EXAMPLES:
            break

        if processed_examples % 50 == 0:
            print(
                f"Processed {processed_examples} examples | "
                f"Unique passages: {len(unique_passages)}"
            )

print("\nData preparation completed!")
print(f"Examples processed: {processed_examples}")
print(f"Unique passages saved: {len(unique_passages)}")
print(f"Output saved to: {OUTPUT_FILE}")