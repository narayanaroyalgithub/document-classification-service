# Document Classification API

A production-ready REST API for automated document classification using fine-tuned NLP models. Built with FastAPI and PyTorch, designed for scalable healthcare and enterprise workflows.

## Features
- **RESTful API**: Fast and asynchronous endpoints built with FastAPI.
- **NLP Classification**: Uses transformer-based models (BERT architecture) for high-accuracy text classification.
- **Production-Ready**: Includes logging, error handling, and modular architecture.
- **Multi-Format Extraction**: Text extraction pipelines for PDF and raw text.

## Tech Stack
- **Backend**: Python, FastAPI, Uvicorn
- **ML/NLP**: PyTorch, HuggingFace Transformers, Scikit-learn
- **Data Processing**: PyPDF2, Pandas

## API Endpoints
- `GET /health` - System health and model load status
- `POST /classify` - Upload a document for category prediction
