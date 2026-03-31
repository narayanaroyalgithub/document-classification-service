# Document Classification Service

Simple document classification API built with **FastAPI** and **Transformers (BERT / DistilBERT)**.

It accepts `.txt` and `.pdf` files, extracts text, runs a pre-trained BERT model, and returns:
- top predicted label
- confidence score
- all label probabilities

This is a learning / demo project to showcase:
- building an ML-backed REST API
- integrating a Transformer model
- basic document preprocessing

## Tech Stack

- Python
- FastAPI (API)
- Uvicorn (ASGI server)
- HuggingFace Transformers (DistilBERT)
- PyTorch
- PyPDF2 (PDF text extraction)

## Install

```bash
pip install -r requirements.txt
