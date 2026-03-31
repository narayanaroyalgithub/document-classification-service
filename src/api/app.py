from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Classification Service", version="1.0")

class ClassificationResponse(BaseModel):
    filename: str
    predicted_category: str
    confidence: float
    processing_time_ms: float

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing ML models and loading weights into memory...")
    # In production, model weights would be loaded here from GCS/S3

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "document-classification"}

@app.post("/classify", response_model=ClassificationResponse)
async def classify_document(file: UploadFile = File(...)):
    start_time = time.time()
    
    if not file.filename.endswith(('.txt', '.pdf')):
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files supported")
        
    try:
        # Read file contents
        content = await file.read()
        
        # Mocking model inference for demonstration
        # In production: text = extract_text(content) -> features = tokenize(text) -> model.predict(features)
        predicted_category = "Medical Record" if "patient" in file.filename.lower() else "General Form"
        confidence = 0.94
        
        process_time = round((time.time() - start_time) * 1000, 2)
        
        return ClassificationResponse(
            filename=file.filename,
            predicted_category=predicted_category,
            confidence=confidence,
            processing_time_ms=process_time
        )
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during classification")
