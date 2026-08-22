import os
import time
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag_pipeline import run_rag
from backend.transcribe import transcribe_audio


# =========================================
# CREATE FASTAPI APP
# =========================================

app = FastAPI(
    title="HH Goa 2026 Voice RAG API",
    description="Voice-enabled Retrieval-Augmented Generation API",
    version="1.0.0"
)


# =========================================
# CORS
# =========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================
# REQUEST MODEL
# =========================================

class QueryRequest(BaseModel):
    query: str


# =========================================
# ROOT ENDPOINT
# =========================================

@app.get("/")
def root():
    return {
        "message": "HH Goa 2026 Voice RAG API is running"
    }


# =========================================
# HEALTH CHECK
# =========================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "HH Goa 2026 Voice RAG"
    }


# =========================================
# TEXT RAG ENDPOINT
# =========================================

@app.post("/ask")
def ask_question(request: QueryRequest):

    total_start = time.perf_counter()

    try:

        response = run_rag(
            request.query
        )

        total_latency = (
            time.perf_counter()
            - total_start
        ) * 1000

        return {
            "success": response.success,
            "blocked": response.blocked,
            "query": response.query,
            "answer": response.answer,
            "sources": response.sources,
            "error": response.error,

            "latency": {
                "retrieval_ms": round(
                    response.retrieval_latency_ms,
                    2
                ),

                "generation_ms": round(
                    response.generation_latency_ms,
                    2
                ),

                "total_ms": round(
                    total_latency,
                    2
                )
            }
        }

    except Exception as error:

        return {
            "success": False,
            "blocked": False,
            "query": request.query,
            "answer": None,
            "sources": [],
            "error": str(error)
        }


# =========================================
# VOICE RAG ENDPOINT
# =========================================

@app.post("/voice-query")
async def voice_query(
    audio: UploadFile = File(...)
):

    total_start = time.perf_counter()

    temp_path = None

    try:

        # ---------------------------------
        # GET FILE EXTENSION
        # ---------------------------------

        filename = audio.filename or "audio.wav"

        extension = os.path.splitext(
            filename
        )[1]

        if not extension:
            extension = ".wav"


        # ---------------------------------
        # SAVE TEMPORARY AUDIO FILE
        # ---------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_path = temp_file.name

            content = await audio.read()

            temp_file.write(content)


        # ---------------------------------
        # TRANSCRIBE AUDIO
        # ---------------------------------

        transcription_start = time.perf_counter()

        transcription = transcribe_audio(
            temp_path
        )

        transcription_latency = (
            time.perf_counter()
            - transcription_start
        ) * 1000


        # ---------------------------------
        # GET TRANSCRIPT
        # ---------------------------------

        if isinstance(transcription, dict):

            transcript = transcription.get(
                "transcript",
                ""
            )

            language = transcription.get(
                "language",
                None
            )

        else:

            transcript = str(
                transcription
            )

            language = None


        # ---------------------------------
        # CHECK EMPTY TRANSCRIPT
        # ---------------------------------

        if not transcript.strip():

            return {
                "success": False,
                "blocked": False,
                "transcript": "",
                "query": "",
                "answer": None,
                "sources": [],
                "error": "No speech could be detected."
            }


        # ---------------------------------
        # RUN RAG PIPELINE
        # ---------------------------------

        rag_response = run_rag(
            transcript
        )


        # ---------------------------------
        # TOTAL LATENCY
        # ---------------------------------

        total_latency = (
            time.perf_counter()
            - total_start
        ) * 1000


        # ---------------------------------
        # RETURN RESPONSE
        # ---------------------------------

        return {
            "success": rag_response.success,

            "blocked": rag_response.blocked,

            "transcript": transcript,

            "language": language,

            "query": rag_response.query,

            "answer": rag_response.answer,

            "sources": rag_response.sources,

            "error": rag_response.error,

            "latency": {

                "transcription_ms": round(
                    transcription_latency,
                    2
                ),

                "retrieval_ms": round(
                    rag_response.retrieval_latency_ms,
                    2
                ),

                "generation_ms": round(
                    rag_response.generation_latency_ms,
                    2
                ),

                "total_ms": round(
                    total_latency,
                    2
                )
            }
        }


    except Exception as error:

        return {
            "success": False,
            "blocked": False,
            "transcript": "",
            "query": "",
            "answer": None,
            "sources": [],
            "error": str(error)
        }


    finally:

        # ---------------------------------
        # DELETE TEMP AUDIO FILE
        # ---------------------------------

        if temp_path and os.path.exists(
            temp_path
        ):

            os.remove(
                temp_path
            )