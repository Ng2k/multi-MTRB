"""Module for dual-stream textual feature extraction using RoBERTa and mT5.

This module implements the textual branch of the Multi-MTRB architecture as 
described in the Nature 2025 publication. It extracts semantic and structural 
features in parallel to optimize GPU utilization and prepares data for 
Multiple Instance Learning (MIL).

Typical usage example:
    extractor = MultiMTRBExtractor()
    features = extractor.extract_session(cleaned_df)
"""

import concurrent.futures
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel, MT5EncoderModel
from typing import List

from src.utils.logger import get_logger

logger = get_logger().bind(module="features.text_extractor")


class MultiMTRBExtractor:
    """Orchestrates parallel feature extraction from RoBERTa and mT5 models.

    This class manages two Transformer models to capture complementary linguistic 
    features. It uses multi-threading to dispatch GPU kernels for both models 
    simultaneously, reducing total inference latency.

    Attributes:
        device: The torch.device (CUDA or CPU) where models are loaded.
        roberta_tokenizer: Tokenizer for the RoBERTa model.
        roberta_model: Pre-trained RoBERTa model for semantic embeddings.
        mt5_tokenizer: Tokenizer for the mT5 model.
        mt5_model: Pre-trained mT5 encoder for structural/diverse patterns.
    """

    def __init__(
        self, 
        roberta_name: str = "roberta-base", 
        mt5_name: str = "google/mt5-small"
    ) -> None:
        """Initializes the extractor with pre-trained Transformer models.

        Args:
            roberta_name: HuggingFace model identifier for RoBERTa.
            mt5_name: HuggingFace model identifier for mT5.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        # Initialize RoBERTa
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_name)
        self.roberta_model = AutoModel.from_pretrained(roberta_name).to(self.device)
 
        # Initialize mT5 (Encoder only for feature extraction)
        self.mt5_tokenizer = AutoTokenizer.from_pretrained(mt5_name)
        self.mt5_model = MT5EncoderModel.from_pretrained(mt5_name).to(self.device)
 
        self.roberta_model.eval()
        self.mt5_model.eval()
 
        logger.info(
            "Multi-MTRB Extractor initialized", 
            roberta=roberta_name, 
            mt5=mt5_name, 
            device=str(self.device)
        )

    def _get_roberta_features(self, utterances: List[str]) -> torch.Tensor:
        """Extracts RoBERTa [CLS] token embeddings.

        Args:
            utterances: A list of cleaned text strings.

        Returns:
            A tensor of shape (num_utterances, 768).
        """
        inputs = self.roberta_tokenizer(
            utterances, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
 
        # Return [CLS] token (global semantic representation)
        return outputs.last_hidden_state[:, 0, :].cpu()

    def _get_mt5_features(self, utterances: List[str]) -> torch.Tensor:
        """Extracts mT5 Mean-Pooled embeddings.

        Args:
            utterances: A list of cleaned text strings.

        Returns:
            A tensor of shape (num_utterances, 512).
        """
        inputs = self.mt5_tokenizer(
            utterances, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.mt5_model(**inputs)
 
        # Mean Pooling to account for mT5's lack of a dedicated [CLS] token
        mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask).cpu()

    def extract_session(self, df: pd.DataFrame) -> torch.Tensor:
        """Runs dual-stream extraction in parallel and concatenates results.

        This method uses a ThreadPoolExecutor to run RoBERTa and mT5 inference 
        concurrently. This is effective for GPU tasks as it allows the driver 
        to queue kernels for both models efficiently.

        Args:
            df: A pandas DataFrame containing a 'value' column of utterances.

        Returns:
            A concatenated tensor of shape (num_utterances, 1280).
            Returns an empty tensor if the DataFrame is empty.
        """
        utterances = df['value'].tolist()
        if not utterances:
            logger.warning("Empty transcript provided for extraction")
            return torch.empty(0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_roberta = executor.submit(self._get_roberta_features, utterances)
            future_mt5 = executor.submit(self._get_mt5_features, utterances)

            roberta_feats = future_roberta.result()
            mt5_feats = future_mt5.result()

        # Concatenate features: (Batch, 768 + 512) -> (Batch, 1280)
        combined_feats = torch.cat((roberta_feats, mt5_feats), dim=1)
 
        logger.info(
            "Multi-MTRB session features extracted", 
            instances=combined_feats.shape[0], 
            total_dim=combined_feats.shape[1]
        )
 
        return combined_feats

