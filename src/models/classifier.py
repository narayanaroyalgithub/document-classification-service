
---

## 4. `classifier.py`

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """
    Simple wrapper around a HuggingFace Transformer model.

    For demo purposes, we use a sentiment model:
    - 'NEGATIVE'
    - 'POSITIVE'

    In a real project, you'd fine-tune on your own labels,
    e.g. ["invoice", "contract", "report", ...].
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.max_length = max_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.labels: List[str] = []  # filled after model load
        self._loaded = False

    def load(self):
        """Load tokenizer and model into memory."""
        if self._loaded:
            return

        logger.info(f"Loading tokenizer and model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # HuggingFace stores label mapping in config.id2label
        id2label = self.model.config.id2label
        # Ensure order by id
        self.labels = [id2label[i] for i in sorted(id2label.keys())]

        self._loaded = True
        logger.info("Model loaded successfully.")

    def is_loaded(self) -> bool:
        return self._loaded

    def get_labels(self) -> List[str]:
        return self.labels

    def predict(self, text: str) -> Dict:
        """
        Run inference on a single text string.

        Returns:
            {
                "predicted_label": str,
                "confidence": float,
                "all_scores": {label: prob}
            }
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not text or not text.strip():
            return {
                "predicted_label": "UNKNOWN",
                "confidence": 0.0,
                "all_scores": {lbl: 0.0 for lbl in self.labels},
            }

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        # top prediction
        confidence, pred_id = torch.max(probs, dim=0)
        pred_label = self.labels[pred_id.item()]

        all_scores = {
            self.labels[i]: float(probs[i].cpu().item()) for i in range(len(self.labels))
        }

        return {
            "predicted_label": pred_label,
            "confidence": float(confidence.cpu().item()),
            "all_scores": all_scores,
        }
