import time
import numpy as np

from rag_pipeline import run_rag


# =========================================
# TEST QUERIES
# =========================================

TEST_QUERIES = [

    "कॉर्पोरेशन क्या है?",

    "राज्य क्या है?",

    "राष्ट्र क्या है?",

    "सरकारी निगम क्या है?",

    "कंपनी क्या है?",

    "निगम कैसे काम करता है?",

    "शेयरधारक क्या होते हैं?",

    "संघीय ठेकेदार क्या है?",

    "एएएफपी क्या है?",

    "राज्य और राष्ट्र में क्या अंतर है?",

    "कंपनी का क्या अर्थ है?",

    "निगम के सदस्य कौन होते हैं?",

    "सार्वजनिक निगम क्या है?",

    "निजी निगम क्या है?",

    "कॉर्पोरेशन और कंपनी में क्या अंतर है?",

    "कानून में निगम को कैसे मान्यता दी जाती है?",

    "सरकारी स्वामित्व वाला निगम क्या है?",

    "शेयर क्या होते हैं?",

    "राजनीतिक समुदाय क्या है?",

    "राष्ट्र-राज्य क्या है?",

    "राज्य की परिभाषा क्या है?",

    "कंपनी कैसे निगमित होती है?",

    "निगम की शक्तियां क्या होती हैं?",

    "निगम के दायित्व क्या होते हैं?",

    "सार्वजनिक स्टॉक क्या है?",

    "निजी स्टॉक क्या है?",

    "निगम का अस्तित्व किससे स्वतंत्र होता है?",

    "सरकार के स्वामित्व वाली कंपनी क्या है?",

    "एकल कानूनी इकाई क्या होती है?",

    "शेयरधारक कंपनी को कैसे नियंत्रित करते हैं?"
]


# =========================================
# PERCENTILE FUNCTION
# =========================================

def percentile(values, p):

    if not values:
        return 0

    return float(
        np.percentile(
            values,
            p
        )
    )


# =========================================
# RUN BENCHMARK
# =========================================

def run_benchmark():

    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []

    successful_queries = 0
    failed_queries = 0

    print("\n" + "=" * 65)
    print("HH GOA 2026 - RAG LATENCY BENCHMARK")
    print("=" * 65)

    print(
        f"\nTotal test queries: "
        f"{len(TEST_QUERIES)}"
    )

    print(
        "\nRunning benchmark...\n"
    )

    # =====================================
    # WARM-UP RUN
    # =====================================

    print("Performing warm-up run...")

    try:

        run_rag(
            TEST_QUERIES[0]
        )

        print("Warm-up completed.\n")

    except Exception as error:

        print(
            f"Warm-up failed: {error}"
        )

    # =====================================
    # BENCHMARK QUERIES
    # =====================================

    for index, query in enumerate(
        TEST_QUERIES,
        start=1
    ):

        print(
            f"[{index}/{len(TEST_QUERIES)}] "
            f"Testing: {query}"
        )

        start_time = time.perf_counter()

        response = run_rag(query)

        actual_total_latency = (
            time.perf_counter()
            - start_time
        ) * 1000

        if response.success:

            successful_queries += 1

            retrieval_latencies.append(
                response.retrieval_latency_ms
            )

            generation_latencies.append(
                response.generation_latency_ms
            )

            total_latencies.append(
                actual_total_latency
            )

            print(
                f"  Retrieval: "
                f"{response.retrieval_latency_ms:.2f} ms"
            )

            print(
                f"  Generation: "
                f"{response.generation_latency_ms:.2f} ms"
            )

            print(
                f"  Total: "
                f"{actual_total_latency:.2f} ms"
            )

        else:

            failed_queries += 1

            print(
                f"  FAILED/BLOCKED: "
                f"{response.error}"
            )

        print()

    # =====================================
    # RESULTS
    # =====================================

    print("\n" + "=" * 65)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 65)

    print(
        f"\nTotal Queries: "
        f"{len(TEST_QUERIES)}"
    )

    print(
        f"Successful Queries: "
        f"{successful_queries}"
    )

    print(
        f"Failed/Blocked Queries: "
        f"{failed_queries}"
    )

    if not total_latencies:

        print(
            "\nNo successful queries available "
            "for latency calculation."
        )

        return

    # =====================================
    # RETRIEVAL RESULTS
    # =====================================

    print("\n" + "-" * 65)
    print("RETRIEVAL LATENCY")
    print("-" * 65)

    print(
        f"P50:  "
        f"{percentile(retrieval_latencies, 50):.2f} ms"
    )

    print(
        f"P70:  "
        f"{percentile(retrieval_latencies, 70):.2f} ms"
    )

    print(
        f"P100: "
        f"{percentile(retrieval_latencies, 100):.2f} ms"
    )

    print(
        f"Average: "
        f"{np.mean(retrieval_latencies):.2f} ms"
    )

    # =====================================
    # GENERATION RESULTS
    # =====================================

    print("\n" + "-" * 65)
    print("GENERATION LATENCY")
    print("-" * 65)

    print(
        f"P50:  "
        f"{percentile(generation_latencies, 50):.2f} ms"
    )

    print(
        f"P70:  "
        f"{percentile(generation_latencies, 70):.2f} ms"
    )

    print(
        f"P100: "
        f"{percentile(generation_latencies, 100):.2f} ms"
    )

    print(
        f"Average: "
        f"{np.mean(generation_latencies):.2f} ms"
    )

    # =====================================
    # TOTAL PIPELINE RESULTS
    # =====================================

    print("\n" + "-" * 65)
    print("TOTAL PIPELINE LATENCY")
    print("-" * 65)

    print(
        f"P50:  "
        f"{percentile(total_latencies, 50):.2f} ms"
    )

    print(
        f"P70:  "
        f"{percentile(total_latencies, 70):.2f} ms"
    )

    print(
        f"P100: "
        f"{percentile(total_latencies, 100):.2f} ms"
    )

    print(
        f"Average: "
        f"{np.mean(total_latencies):.2f} ms"
    )

    # =====================================
    # FASTEST AND SLOWEST
    # =====================================

    print("\n" + "-" * 65)
    print("ADDITIONAL STATISTICS")
    print("-" * 65)

    print(
        f"Fastest Total: "
        f"{min(total_latencies):.2f} ms"
    )

    print(
        f"Slowest Total: "
        f"{max(total_latencies):.2f} ms"
    )

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETED")
    print("=" * 65 + "\n")


# =========================================
# MAIN
# =========================================

if __name__ == "__main__":

    run_benchmark()