from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import re

class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def clean_text(self, text):
        # Remove characters that could break tokenization
        text = re.sub(r"[^A-Za-z0-9,.!?;:()\\[\\]\\n'\"\\s-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def summarize(self, text):
        # Clean and truncate to ensure safe input
        text = self.clean_text(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=50,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import re

class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def clean_text(self, text):
        # Remove characters that could break tokenization
        text = re.sub(r"[^A-Za-z0-9,.!?;:()\\[\\]\\n'\"\\s-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def summarize(self, text):
        # Clean and truncate to ensure safe input
        text = self.clean_text(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=50,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
