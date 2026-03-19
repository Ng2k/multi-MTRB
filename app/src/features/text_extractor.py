"""Module for dual-stream textual feature extraction using RoBERTa and mT5.

Optimized for memory efficiency with utterance batching and robust pooling.
"""

from typing import List
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel, MT5EncoderModel

from src.utils import get_logger, settings

class MultiMTRBExtractor:
    """Orchestrates batch-wise feature extraction from RoBERTa and mT5 models."""

    def __init__(
        self, 
        roberta_name: str = "roberta-base", 
        mt5_name: str = "google/mt5-small",
        batch_size: int = 32
    ) -> None:
        self.logger = get_logger().bind(module="features.text_extractor")
        self.device = torch.device(settings.device)
        self.batch_size = batch_size
        self.max_length = settings.token_size
 
        # Initialize RoBERTa
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_name, use_fast=True)
        self.roberta_model = AutoModel.from_pretrained(roberta_name).to(self.device)
 
        # Initialize mT5
        self.mt5_tokenizer = AutoTokenizer.from_pretrained(mt5_name, use_fast=True)
        self.mt5_model = MT5EncoderModel.from_pretrained(mt5_name).to(self.device) # type: ignore
 
        self.roberta_model.eval()
        self.mt5_model.eval()
 
        self.logger.info(
            "Multi-MTRB Extractor initialized with batching", 
            batch_size=self.batch_size,
            device=str(self.device)
        )


    def _get_roberta_features(self, batch: List[str]) -> torch.Tensor:
        """Extracts RoBERTa [CLS] token embeddings for a batch."""
        inputs = self.roberta_tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu()


    def _get_mt5_features(self, batch: List[str]) -> torch.Tensor:
        """Extracts mT5 Mean-Pooled embeddings for a batch."""
        inputs = self.mt5_tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.mt5_model(**inputs)
 
        # Robust Mean Pooling using the attention mask
        mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask).cpu()


    def extract_session(self, df: pd.DataFrame) -> torch.Tensor:
        """Runs dual-stream extraction using batching to prevent OOM."""
        utterances = df['value'].tolist()
        if not utterances:
            return torch.empty(0)

        all_features = []

        # Process in chunks to manage VRAM
        for i in range(0, len(utterances), self.batch_size):
            batch = utterances[i : i + self.batch_size]

            # Extract from both streams
            roberta_batch = self._get_roberta_features(batch)
            mt5_batch = self._get_mt5_features(batch)

            # Combine batch features (Batch, 1280)
            combined_batch = torch.cat((roberta_batch, mt5_batch), dim=1)
            all_features.append(combined_batch)

        final_tensor = torch.cat(all_features, dim=0)
        return final_tensor

