from transcribe import transcribe_audio
from retrieve import retrieve


def voice_retrieve(audio_path):
    print("\nStep 1: Transcribing audio...")

    transcription = transcribe_audio(audio_path)

    query = transcription["transcript"]

    print(f"\nTranscript: {query}")

    print("\nStep 2: Searching relevant context...")

    results = retrieve(query, top_k=5)

    return {
        "query": query,
        "language": transcription["language_code"],
        "results": results
    }


if __name__ == "__main__":

    audio_path = input(
        "Enter the path to your audio file: "
    ).strip()

    try:
        output = voice_retrieve(audio_path)

        print("\n" + "=" * 50)
        print("VOICE RAG RETRIEVAL RESULT")
        print("=" * 50)

        print(f"\nTranscribed Query: {output['query']}")
        print(f"Language: {output['language']}")

        print("\nTop Retrieved Context:\n")

        for rank, result in enumerate(
            output["results"],
            start=1
        ):
            print(f"--- Result {rank} ---")
            print(f"Score: {result['score']:.4f}")
            print(f"Text: {result['text']}\n")

    except Exception as error:
        print(f"\nError: {error}")
        