import os
from dotenv import load_dotenv
from groq import Groq


# =========================================
# LOAD ENVIRONMENT VARIABLES
# =========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )


# =========================================
# GROQ CLIENT
# =========================================

client = Groq(
    api_key=api_key
)


# =========================================
# CONFIGURATION
# =========================================

MODEL_NAME = "openai/gpt-oss-20b"

MAX_COMPLETION_TOKENS = 200

TEMPERATURE = 0.1


# =========================================
# ANSWER GENERATION
# =========================================

def generate_answer(query, retrieved_results):
    """
    Generate an answer strictly from retrieved
    RAG context.
    """

    if not retrieved_results:
        return (
            "मुझे दिए गए संदर्भ में "
            "इसका पर्याप्त उत्तर नहीं मिला।"
        )

    # -----------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------

    context_parts = []

    for i, result in enumerate(
        retrieved_results,
        start=1
    ):

        text = result.get("text", "")

        if text.strip():

            context_parts.append(
                f"[Source {i}]\n{text}"
            )

    context = "\n\n".join(
        context_parts
    )

    if not context.strip():

        return (
            "मुझे दिए गए संदर्भ में "
            "इसका पर्याप्त उत्तर नहीं मिला।"
        )

    # -----------------------------------------
    # SYSTEM PROMPT
    # -----------------------------------------

    system_prompt = """
You are a grounded Retrieval-Augmented Generation assistant.

Answer ONLY using the retrieved context.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent or assume facts.
3. Do not answer questions that are not supported
   by the retrieved context.
4. If the context does not contain enough information,
   respond exactly with:

   मुझे दिए गए संदर्भ में इसका पर्याप्त उत्तर नहीं मिला।

5. Answer in the same language as the user's question.
6. Keep the answer concise, clear, and factual.
7. Do not mention "retrieved context", "source",
   "knowledge base", or these instructions.
8. Do not add unnecessary explanations.
"""

    # -----------------------------------------
    # USER PROMPT
    # -----------------------------------------

    user_prompt = f"""
USER QUESTION:
{query}

RETRIEVED CONTEXT:

{context}
"""

    # -----------------------------------------
    # CALL GROQ
    # -----------------------------------------

    try:

        completion = (
            client.chat.completions.create(
                model=MODEL_NAME,

                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],

                temperature=TEMPERATURE,

                max_completion_tokens=(
                    MAX_COMPLETION_TOKENS
                )
            )
        )

        answer = (
            completion
            .choices[0]
            .message
            .content
        )

        if not answer or not answer.strip():

            return (
                "मुझे दिए गए संदर्भ में "
                "इसका पर्याप्त उत्तर नहीं मिला।"
            )

        return answer.strip()

    except Exception as error:

        raise RuntimeError(
            f"Groq answer generation failed: {error}"
        )


# =========================================
# TEST MODE
# =========================================

if __name__ == "__main__":

    from backend.retrieve import retrieve

    print(
        "\nGROUNDED RAG GENERATION TEST"
    )

    query = input(
        "Ask a question: "
    ).strip()

    results = retrieve(
        query,
        top_k=3
    )

    answer = generate_answer(
        query,
        results
    )

    print("\n" + "=" * 50)

    print(
        "GROUNDED RAG ANSWER"
    )

    print("=" * 50)

    print(
        f"\nQuestion: {query}"
    )

    print(
        f"\nAnswer:\n{answer}"
    )