import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from typing import List
import numpy as np


class bge_large_reranker():
    """This is bge-reranker-large."""

    def __init__(self,
                 model_name_or_path: str = 'BAAI/bge-reranker-large',
                 use_fp16: bool = False):
        """Init the hyde reranker model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
            use_fp16 = False
        if use_fp16:
            self.model.half()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)

    @torch.no_grad()
    def run(self, query, contexts, batch_size: int = 256,
            max_length: int = 512) -> List[float]:
        """Get reranked contexts in runtime"""

        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus

        assert isinstance(query, str)
        assert isinstance(contexts, list)
        sentence_pairs = [[query, cxt] for cxt in contexts]

        all_scores = []
        for start_index in tqdm(range(0, len(sentence_pairs), batch_size), desc="Compute Reranking Scores",
                                disable=len(sentence_pairs) < 128):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)

            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            all_scores.extend(scores.cpu().numpy().tolist())

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        probabilities = sigmoid(np.array(all_scores))
        print(probabilities)
        return probabilities
