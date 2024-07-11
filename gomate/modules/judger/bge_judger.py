import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from gomate.modules.judger.base import BaseJudger


class BgeJudgerConfig:
    """
    Configuration class for setting up a BERT-based judger.

    Attributes:
        model_name_or_path (str): Path or model identifier for the pretrained model from Hugging Face's model hub.
        device (str): Device to load the model onto ('cuda' or 'cpu').
    """

    def __init__(self, model_name_or_path='bert-base-uncased'):
        self.model_name_or_path = model_name_or_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def log_config(self):
        # Log the current configuration settings
        return f"""
        BgejudgerConfig:
            Model Name or Path: {self.model_name_or_path}
            Device: {self.device}
        """


class BgeJudger(BaseJudger):
    """
    A judger that utilizes a BERT-based model for sequence classification
    to judge a list of documents based on their relevance to a given query.



    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.judge_tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.judge_model = AutoModelForSequenceClassification.from_pretrained(config.model_name_or_path) \
            .half().to(config.device).eval()
        self.device = config.device
        print('Successful load judge model')

    def judge(self, query: str, documents: List[str], k: int = 5, is_sorted: bool = False) -> list[dict[str, Any]]:
        # Process input documents for uniqueness and formatting
        # documents = list(set(documents))
        pairs = [[query, d] for d in documents]

        # Tokenize and predict relevance scores
        with torch.no_grad():
            inputs = self.judge_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt',
                                          max_length=512).to(self.device)
            scores = self.judge_model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()

        # Pair documents with their scores, sort by scores in descending order
        # top_docs = [{'text': doc, 'score': score} for doc, score in zip(documents, scores)]
        judge_docs = [
            {
                'text': doc,
                'score': score,
                'label': 1 if score >= 0 else 0
            }
            for doc, score in zip(documents, scores)
        ]
        return judge_docs
