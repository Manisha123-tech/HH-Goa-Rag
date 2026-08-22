import time
import re
from typing import Any
from pydantic import BaseModel, Field

from backend.retrieve import retrieve
from backend.generate_answer import generate_answer

# =========================================
# CONFIGURATION
# =========================================

TOP_K = 3
MIN_RELEVANCE_SCORE = 0.45

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 0.5


# =========================================
# STRUCTURED INPUT
# =========================================

class RAGRequest(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=500
    )


# =========================================
# STRUCTURED OUTPUT
# =========================================

class RAGResponse(BaseModel):
    success: bool
    query: str
    answer: str | None = None

    sources: list[dict[str, Any]] = Field(
        default_factory=list
    )

    error: str | None = None
    blocked: bool = False

    retrieval_latency_ms: float = 0
    generation_latency_ms: float = 0
    total_latency_ms: float = 0


# =========================================
# INPUT GUARDRAIL
# =========================================

def validate_query(query: str):

    query = query.strip()

    if len(query) < 2:
        return False, "Query is too short."

    if not any(char.isalpha() for char in query):
        return False, "Please provide a meaningful question."

    if re.fullmatch(r"(.)\1{4,}", query):
        return False, "Please provide a meaningful question."

    return True, None


# =========================================
# RETRIEVAL WITH RETRIES
# =========================================

def retrieve_with_retry(query: str):

    last_error = None

    for attempt in range(MAX_RETRIES + 1):

        try:
            return retrieve(
                query,
                top_k=TOP_K
            )

        except Exception as error:

            last_error = error

            if attempt < MAX_RETRIES:

                print(
                    f"Retrieval failed. "
                    f"Retrying... ({attempt + 1}/{MAX_RETRIES})"
                )

                time.sleep(
                    RETRY_DELAY_SECONDS * (attempt + 1)
                )

    raise RuntimeError(
        f"Retrieval failed after retries: {last_error}"
    )


# =========================================
# GENERATION WITH RETRIES
# =========================================

def generate_with_retry(query: str, results: list):

    last_error = None

    for attempt in range(MAX_RETRIES + 1):

        try:
            return generate_answer(
                query,
                results
            )

        except Exception as error:

            last_error = error

            if attempt < MAX_RETRIES:

                print(
                    f"Generation failed. "
                    f"Retrying... ({attempt + 1}/{MAX_RETRIES})"
                )

                time.sleep(
                    RETRY_DELAY_SECONDS * (attempt + 1)
                )

    raise RuntimeError(
        f"Answer generation failed after retries: {last_error}"
    )


# =========================================
# LANGUAGE-SAFE TOKENIZATION
# =========================================

def tokenize(text: str):

    # Works better for Hindi and English because
    # we split by whitespace and punctuation
    tokens = re.split(
        r"[\s,.;:!?(){}\[\]\"'।]+",
        text.lower()
    )

    return {
        token.strip()
        for token in tokens
        if len(token.strip()) >= 2
    }


# =========================================
# RELEVANCE GUARDRAIL
# =========================================

def is_relevant_enough(query: str, results: list):

    if not results:
        return False

    best_score = results[0].get(
        "score",
        0
    )

    # First-stage semantic similarity check
    if best_score < MIN_RELEVANCE_SCORE:
        return False

    # ---------------------------------
    # QUERY TOKENIZATION
    # ---------------------------------

    query_words = tokenize(query)

    # Common English question words
    stop_words = {
        "what",
        "is",
        "who",
        "the",
        "a",
        "an",
        "are",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "how",
        "why",
        "when",
        "where"
    }

    query_words = query_words - stop_words

    # ---------------------------------
    # HINDI QUESTION WORDS
    # ---------------------------------

    hindi_stop_words = {
        "क्या",
        "है",
        "कौन",
        "का",
        "की",
        "के",
        "में",
        "और",
        "को",
        "से",
        "पर",
        "यह",
        "वह",
        "कैसे",
        "क्यों",
        "कब",
        "कहाँ"
    }

    query_words = (
        query_words
        - hindi_stop_words
    )

    # ---------------------------------
    # CONTEXT TOKENIZATION
    # ---------------------------------

    context = " ".join(
        result.get("text", "")
        for result in results
    )

    context_words = tokenize(context)

    overlap = query_words.intersection(
        context_words
    )

    # ---------------------------------
    # DEBUG INFORMATION
    # ---------------------------------

    print(
        f"Meaningful Query Words: "
        f"{query_words}"
    )

    print(
        f"Word Overlap: "
        f"{overlap}"
    )

    # If query contains meaningful words,
    # at least one should connect to context.
    if query_words and not overlap:
        return False

    return True


# =========================================
# MAIN RAG HARNESS
# =========================================

def run_rag(query: str):

    total_start = time.perf_counter()

    try:

        # ---------------------------------
        # 1. INPUT VALIDATION
        # ---------------------------------

        request = RAGRequest(
            query=query
        )

        valid, error = validate_query(
            request.query
        )

        if not valid:

            return RAGResponse(
                success=False,
                blocked=True,
                query=request.query,
                error=error,
                total_latency_ms=(
                    time.perf_counter()
                    - total_start
                ) * 1000
            )

        # ---------------------------------
        # 2. RETRIEVAL
        # ---------------------------------

        retrieval_start = time.perf_counter()

        results = retrieve_with_retry(
            request.query
        )

        retrieval_latency = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        # ---------------------------------
        # 3. NO RESULTS GUARDRAIL
        # ---------------------------------

        if not results:

            return RAGResponse(
                success=False,
                blocked=True,
                query=request.query,
                error=(
                    "No relevant context was found "
                    "in the knowledge base."
                ),
                retrieval_latency_ms=retrieval_latency,
                total_latency_ms=(
                    time.perf_counter()
                    - total_start
                ) * 1000
            )

        # ---------------------------------
        # 4. DEBUG RETRIEVAL SCORES
        # ---------------------------------

        best_score = results[0].get(
            "score",
            0
        )

        print("\nDEBUG RETRIEVAL SCORES:")

        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"Result {i}: "
                f"{result.get('score', 0):.4f}"
            )

        print(
            f"\nBest Score: "
            f"{best_score:.4f}"
        )

        print(
            f"Minimum Required Score: "
            f"{MIN_RELEVANCE_SCORE:.4f}"
        )

        # ---------------------------------
        # 5. RELEVANCE GUARDRAIL
        # ---------------------------------

        if not is_relevant_enough(
            request.query,
            results
        ):

            return RAGResponse(
                success=False,
                blocked=True,
                query=request.query,
                error=(
                    "I cannot answer this reliably because "
                    "the knowledge base does not contain "
                    "sufficiently relevant information."
                ),
                sources=results,
                retrieval_latency_ms=retrieval_latency,
                total_latency_ms=(
                    time.perf_counter()
                    - total_start
                ) * 1000
            )

        # ---------------------------------
        # 6. ANSWER GENERATION
        # ---------------------------------

        generation_start = time.perf_counter()

        answer = generate_with_retry(
            request.query,
            results
        )

        generation_latency = (
            time.perf_counter()
            - generation_start
        ) * 1000

        # ---------------------------------
        # 7. TOTAL LATENCY
        # ---------------------------------

        total_latency = (
            time.perf_counter()
            - total_start
        ) * 1000

        return RAGResponse(
            success=True,
            query=request.query,
            answer=answer,
            sources=results,
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=generation_latency,
            total_latency_ms=total_latency
        )

    except Exception as error:

        total_latency = (
            time.perf_counter()
            - total_start
        ) * 1000

        return RAGResponse(
            success=False,
            blocked=False,
            query=query,
            error=str(error),
            total_latency_ms=total_latency
        )


# =========================================
# TEST MODE
# =========================================

if __name__ == "__main__":

    print("\nOPTIMIZED RAG PIPELINE TEST")
    print("Type 'exit' to stop.\n")

    while True:

        query = input(
            "Ask a question: "
        ).strip()

        if query.lower() == "exit":
            break

        response = run_rag(query)

        print("\n" + "=" * 60)

        if response.success:

            print("RAG RESPONSE")

            print(
                f"\nQuestion: "
                f"{response.query}"
            )

            print(
                f"\nAnswer:\n"
                f"{response.answer}"
            )

        elif response.blocked:

            print("REQUEST BLOCKED")

            print(
                f"\nReason: "
                f"{response.error}"
            )

        else:

            print("PIPELINE FAILED")

            print(
                f"\nError: "
                f"{response.error}"
            )

        print("\nLATENCY METRICS")

        print(
            f"Retrieval: "
            f"{response.retrieval_latency_ms:.2f} ms"
        )

        print(
            f"Generation: "
            f"{response.generation_latency_ms:.2f} ms"
        )

        print(
            f"Total: "
            f"{response.total_latency_ms:.2f} ms"
        )

        print("=" * 60 + "\n")