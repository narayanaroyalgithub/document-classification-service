from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict
import logging
import time

from classifier import DocumentClassifier
from text_extractor import extract_text_from_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doc-class-api")

app = FastAPI(
    title="Document Classification Service",
    description="Simple FastAPI + BERT document classification demo.",
    version="1.0.0",
)

classifier = DocumentClassifier()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ClassificationResponse(BaseModel):
    filename: str
    predicted_label: str
    confidence: float
    all_scores: Dict[str, float]
    processing_time_ms: float


@app.on_event("startup")
def on_startup():
    logger.info("Starting up, loading model...")
    classifier.load()
    logger.info("Startup complete.")


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok" if classifier.is_loaded() else "loading",
        "model_loaded": classifier.is_loaded(),
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify(file: UploadFile = File(...)):
    start = time.perf_counter()

    if not (file.filename.lower().endswith(".txt") or file.filename.lower().endswith(".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .pdf files are supported in this demo.",
        )

    content = await file.read()
    text = extract_text_from_bytes(content, file.filename)

    if not text or not text.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not extract any text from the uploaded file.",
        )

    try:
        result = classifier.predict(text)
    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return ClassificationResponse(
        filename=file.filename,
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        all_scores=result["all_scores"],
        processing_time_ms=round(elapsed_ms, 2),
    )
