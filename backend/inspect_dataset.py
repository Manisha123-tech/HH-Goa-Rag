from datasets import load_dataset

DATA_URL = (
    "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/"
    "resolve/main/validation/hinval.parquet"
)

print("Loading MSMARCO-XI Hindi validation data...")
print("Using streaming mode...\n")

dataset = load_dataset(
    "parquet",
    data_files={"validation": DATA_URL},
    split="validation",
    streaming=True
)

print("Stream created successfully!")
print("Reading one example...\n")

example = next(iter(dataset))

print("Example received!\n")

print("FIELDS:")
print(example.keys())

print("\nHINDI QUERY:")
print(example["query"])

print("\nHINDI ANSWER:")
print(example["Answer"])

print("\nENGLISH QUERY:")
print(example["Eng_Query"])

print("\nENGLISH ANSWER:")
print(example["Eng_Answer"])

print("\nNUMBER OF PASSAGES:")
print(len(example["passages"]["Translated_passages"]))

print("\nFIRST HINDI PASSAGE:")
print(example["passages"]["Translated_passages"][0])

print("\nFIRST ENGLISH PASSAGE:")
print(example["passages"]["English_passages"][0])