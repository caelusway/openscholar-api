#!/usr/bin/env python3
"""
Core model components for OpenScholar API
Contains the original working Embedder and Reranker classes
"""

import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

# Device configuration - Auto-detect based on environment
import platform
import os

is_macos = platform.system() == "Darwin"
is_docker = os.path.exists("/.dockerenv")

if is_macos:
    # macOS: Force CPU
    DEVICE = "cpu"
    DTYPE = torch.float32
else:
    # Linux/Docker: Use GPU if available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

logger.info(f"🔧 Device: {DEVICE}, Data Type: {DTYPE}")


class Embedder:
    """Embedding model for query encoding - Original working implementation"""
    def __init__(self, model_id: str, dtype=DTYPE, device=DEVICE):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        
        logger.info(f"Loading embedder model: {model_id}")
        # Match exact working code: use_fast=True and bfloat16
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.enc = (
            AutoModel.from_pretrained(model_id, torch_dtype=dtype).eval().to(device)
        )
        logger.info(f"Embedder model loaded successfully")

    @torch.inference_mode()
    def embed_texts(
        self, texts: List[str], batch_size=64, max_length=512
    ) -> np.ndarray:
        """Embed a list of texts into vectors"""
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc_in = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc_in["input_ids"].to(self.device, non_blocking=True)
            attn = enc_in["attention_mask"].to(self.device, non_blocking=True)
            
            # Use autocast for GPU, no_grad for CPU
            with (
                torch.autocast(device_type="cuda", dtype=self.dtype)
                if self.device == "cuda"
                else torch.no_grad()
            ):
                out = self.enc(input_ids=input_ids, attention_mask=attn)
                last = out.last_hidden_state
                mask = attn.unsqueeze(-1)
                emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            vecs.append(emb.float().cpu().numpy())
        return np.vstack(vecs)


class Reranker:
    """Reranker for scoring query-passage pairs - Original working implementation"""
    def __init__(self, model_id: str):
        self.model_id = model_id
        
        logger.info(f"Loading reranker model: {model_id}")
        # Match exact working code pattern
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_id)
            .eval()
            .to(DEVICE)
        )
        logger.info(f"Reranker model loaded successfully")

    @torch.inference_mode()
    def score(
        self, query: str, passages: List[str], batch_size=32, max_len=512
    ) -> List[float]:
        """Score query-passage pairs"""
        scores = []
        for i in range(0, len(passages), batch_size):
            pairs = [
                [query, passages[j]]
                for j in range(i, min(i + batch_size, len(passages)))
            ]
            enc = self.tok(
                pairs,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(DEVICE)
            logits = self.model(**enc).logits.squeeze(-1)
            scores.extend(
                logits.detach().cpu().tolist()
                if logits.ndim
                else [float(logits.detach().cpu())]
            )
        return scores