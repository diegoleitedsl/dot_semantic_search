from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TransformerEmbedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device("cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, texts: List[str]):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**enc)
        last_hidden = outputs.last_hidden_state            # (batch, seq_len, hidden)
        mask = enc["attention_mask"].unsqueeze(-1)         # (batch, seq_len, 1)

        masked = last_hidden * mask
        summed = masked.sum(dim=1)                         # (batch, hidden)
        counts = mask.sum(dim=1).clamp(min=1)              # (batch, 1)

        embeddings = summed / counts                       # mean pooling

        norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        embeddings = embeddings / norms                    # normaliza L2

        return embeddings.cpu()