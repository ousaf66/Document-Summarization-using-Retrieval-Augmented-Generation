from transformers import pipeline

class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text):
        return self.summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
