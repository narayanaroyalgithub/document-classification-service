# Document Classification Platform

A simple, cloud-style **NLP document classification service** built with:

- **Python**
- **FastAPI** (REST API)
- **Transformers (BART, zero-shot classification)**
- **PyTorch**
- **GCP‑friendly structure**

It classifies documents into 5 practical types common in healthcare/finance workflows.

---

## 🔍 What It Does

You upload a `.pdf` or `.txt` file.

The service:

1. Extracts text from the document.
2. Runs a Transformer‑based zero‑shot classifier.
3. Assigns the document to **one of 5 tags**:

   - `Invoice`  
   - `Insurance Claim`  
   - `Bank or Billing Statement`  
   - `Contract or Agreement`  
   - `General Report`  

4. Returns:
   - the **predicted tag**  
   - a **confidence score**  
   - **scores for all 5 tags**  
   - processing time in ms

This mimics how a real backend service would auto‑tag operational documents for routing, search, and automation.

---

## 🧠 Model

- Uses HuggingFace’s **zero‑shot classification** pipeline:
  - Model: `facebook/bart-large-mnli`
- No custom training code is required:
  - We define the 5 candidate labels.
  - The model assigns probabilities to each label for a given document.
- Strategy:
  - Compute scores for all 5 tags.
  - Pick the **highest‑scoring tag** as the main classification.

---

## 🏗 Architecture

**Components:**

- `text_extractor.py`  
  - Extracts text from `.txt` and `.pdf` files (via `PyPDF2`).

- `classifier.py`  
  - Wraps a zero‑shot Transformer pipeline.
  - Defines the 5 document types.
  - Exposes a `predict(text)` method that returns:
    - `predicted_label`
    - `confidence`
    - `all_scores` (for all 5 tags)

- `app.py`  
  - FastAPI application exposing:
    - `GET /health` – health check (model loaded status)
    - `POST /classify` – file upload → text extraction → classification → JSON response

🌐 API Endpoints & Usage

Once the service is running locally, the main URLs are:

- Health check:  
  `GET http://localhost:8000/health`

- Classification endpoint (file upload):  
  `POST http://localhost:8000/classify`

- Interactive API docs (Swagger UI):  
  `http://localhost:8000/docs`

- Alternative docs (ReDoc):  
  `http://localhost:8000/redoc`

### Example deployment (Cloud Run)

> Note: This is an example of how the service **could** be deployed to Google Cloud Run.  
> The URL below is illustrative; replace it with your own if you deploy.

After deployment on Cloud Run, the service would be available at a URL like:

- Base URL (example):  
  `https://document-classification-service-xxxxx-uc.a.run.app`

Endpoints:

- Health:  
  `GET https://document-classification-service-xxxxx-uc.a.run.app/health`

- Classify:  
  `POST https://document-classification-service-xxxxx-uc.a.run.app/classify`

- Docs (Swagger UI):  
  `https://document-classification-service-xxxxx-uc.a.run.app/docs`


