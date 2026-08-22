import os
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

api_key = os.getenv("SARVAM_API_KEY")

if not api_key:
    raise ValueError(
        "SARVAM_API_KEY not found. Check your .env file."
    )

client = SarvamAI(
    api_subscription_key=api_key
)


def transcribe_audio(audio_path):
    """
    Transcribe an audio file using Sarvam Saaras v3.
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    try:
        with open(audio_path, "rb") as audio_file:

            response = client.speech_to_text.transcribe(
                file=audio_file,
                model="saaras:v3",
                mode="transcribe",
                language_code="hi-IN"
            )

        return {
            "transcript": response.transcript,
            "language_code": response.language_code
        }

    except Exception as e:
        raise RuntimeError(
            f"Speech-to-text failed: {str(e)}"
        )


if __name__ == "__main__":

    audio_path = input(
        "Enter the path to your audio file: "
    ).strip()

    result = transcribe_audio(audio_path)

    print("\nTranscription successful!")
    print(f"Language: {result['language_code']}")
    print(f"Transcript: {result['transcript']}")