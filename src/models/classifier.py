from transformers import pipeline
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """
    Document-type classifier using a zero-shot classification pipeline.

    It classifies a document into one of 5 tags:
    - Invoice
    - Insurance Claim
    - Bank / Billing Statement
    - Contract / Agreement
    - General Report
    """

    def __init__(self):
        # Define the label space
        self.labels: List[str] = [
            "Invoice",
            "Insurance Claim",
            "Bank or Billing Statement",
            "Contract or Agreement",
            "General Report",
        ]
        self._pipeline = None
        self._loaded = False

    def load(self):
        """Load the zero-shot classification pipeline."""
        if self._loaded:
            return

        logger.info("Loading zero-shot classification pipeline (BART MNLI)...")
        # This model is good for zero-shot classification
        self._pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
        self._loaded = True
        logger.info("Zero-shot pipeline loaded successfully.")

    def is_loaded(self) -> bool:
        return self._loaded

    def get_labels(self) -> List[str]:
        return self.labels

    def predict(self, text: str) -> Dict:
        """
        Classify the input text into one of the 5 labels.

        Returns:
            {
                "predicted_label": str,
                "confidence": float,
                "all_scores": {label: prob}
            }
        """
        if not self._loaded or self._pipeline is None:
            raise RuntimeError("Model pipeline not loaded. Call load() first.")

        if not text or not text.strip():
            return {
                "predicted_label": "Unknown",
                "confidence": 0.0,
                "all_scores": {lbl: 0.0 for lbl in self.labels},
            }

        result = self._pipeline(
            text,
            candidate_labels=self.labels,
            multi_label=False,  # choose the single best label
        )

        # result["labels"] and result["scores"] are sorted by score desc
        top_label = result["labels"][0]
        top_score = float(result["scores"][0])

        all_scores = {
            label: float(score) for label, score in zip(result["labels"], result["scores"])
        }

        return {
            "predicted_label": top_label,
            "confidence": top_score,
            "all_scores": all_scores,
        }
