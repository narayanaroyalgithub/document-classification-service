import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

class DocumentClassifier:
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        
    def load(self, model_path: str = None):
        """Loads the tokenizer and model weights."""
        try:
            path = model_path if model_path else self.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, 
                num_labels=self.num_labels
            ).to(self.device)
            self.model.eval()
            logger.info(f"Successfully loaded model to {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def predict(self, text: str) -> dict:
        """Runs inference on a single text string."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        inputs = self.tokenizer(
            text, 
            truncation=True, 
            padding=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
        return {
            "class_id": predicted_class.item(),
            "confidence": confidence.item()
        }
