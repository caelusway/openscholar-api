#!/usr/bin/env python3
"""
Caching system for OpenScholar API
Handles model and data caching to optimize startup times
"""

import os
import pickle
import shutil
import logging
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = Path("./model_cache")
DATASET_CACHE_FILE = CACHE_DIR / "dataset_cache.pkl"
MAPPINGS_CACHE_FILE = CACHE_DIR / "mappings_cache.pkl"
INDEX_CACHE_FILE = CACHE_DIR / "index.faiss"
META_CACHE_FILE = CACHE_DIR / "meta.jsonl"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def setup_cache_directory():
    """Create cache directory if it doesn't exist"""
    CACHE_DIR.mkdir(exist_ok=True)
    logger.info(f"Cache directory: {CACHE_DIR.absolute()}")


def is_dataset_cached() -> bool:
    """Check if dataset is already cached"""
    return (DATASET_CACHE_FILE.exists() and 
            MAPPINGS_CACHE_FILE.exists() and 
            INDEX_CACHE_FILE.exists() and 
            META_CACHE_FILE.exists())


def save_dataset_cache(chunks_df: pd.DataFrame, hash_to_chunk_id: dict, chunk_id_to_row: dict):
    """Save dataset and mappings to cache"""
    logger.info("Saving dataset to cache...")
    
    # Save DataFrame
    with open(DATASET_CACHE_FILE, 'wb') as f:
        pickle.dump(chunks_df, f)
    
    # Save mappings
    mappings = {
        'hash_to_chunk_id': hash_to_chunk_id,
        'chunk_id_to_row': chunk_id_to_row
    }
    with open(MAPPINGS_CACHE_FILE, 'wb') as f:
        pickle.dump(mappings, f)
    
    logger.info("Dataset cached successfully")


def load_dataset_cache() -> Tuple[pd.DataFrame, dict, dict]:
    """Load dataset and mappings from cache"""
    logger.info("Loading dataset from cache...")
    
    # Load DataFrame
    with open(DATASET_CACHE_FILE, 'rb') as f:
        chunks_df = pickle.load(f)
    
    # Load mappings
    with open(MAPPINGS_CACHE_FILE, 'rb') as f:
        mappings = pickle.load(f)
    
    logger.info(f"Loaded cached dataset with {len(chunks_df)} chunks")
    return chunks_df, mappings['hash_to_chunk_id'], mappings['chunk_id_to_row']


def cache_index_files(original_index_path: str, original_meta_path: str):
    """Copy index files to cache directory"""
    logger.info("Caching index files...")
    
    # Copy FAISS index
    shutil.copy2(original_index_path, INDEX_CACHE_FILE)
    
    # Copy metadata
    shutil.copy2(original_meta_path, META_CACHE_FILE)
    
    logger.info("Index files cached successfully")


def get_model_cache_dir(model_name: str) -> Path:
    """Get cache directory for a specific model"""
    model_cache = CACHE_DIR / "models" / model_name.replace("/", "_")
    model_cache.mkdir(parents=True, exist_ok=True)
    return model_cache


def is_model_cached(model_name: str) -> bool:
    """Check if model is already cached locally"""
    model_cache_dir = get_model_cache_dir(model_name)
    # Check for key model files
    return (
        (model_cache_dir / "config.json").exists() and
        (model_cache_dir / "tokenizer_config.json").exists() and
        (
            (model_cache_dir / "pytorch_model.bin").exists() or
            (model_cache_dir / "model.safetensors").exists() or
            list(model_cache_dir.glob("pytorch_model-*.bin")) or
            list(model_cache_dir.glob("model-*.safetensors"))
        )
    )


class CachedEmbedder:
    """Embedder with caching support"""
    def __init__(self, model_id: str, dtype=DTYPE, device=DEVICE):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        
        # Use cached model if available
        cache_dir = get_model_cache_dir(model_id)
        
        if is_model_cached(model_id):
            logger.info(f"Loading {model_id} from cache: {cache_dir}")
            self.tok = AutoTokenizer.from_pretrained(cache_dir, use_fast=False, local_files_only=True)
            self.enc = (
                AutoModel.from_pretrained(cache_dir, torch_dtype=dtype, local_files_only=True)
                .eval().to(device)
            )
        else:
            logger.info(f"Downloading and caching {model_id}...")
            self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            self.enc = (
                AutoModel.from_pretrained(model_id, torch_dtype=dtype).eval().to(device)
            )
            
            # Cache the model
            logger.info(f"Caching {model_id} to {cache_dir}")
            self.tok.save_pretrained(cache_dir)
            self.enc.save_pretrained(cache_dir)
            logger.info(f"Model {model_id} cached successfully")

    @torch.inference_mode()
    def embed_texts(
        self, texts: list, batch_size=64, max_length=512
    ):
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
        return torch.vstack([torch.from_numpy(v) for v in vecs]).numpy()


class CachedReranker:
    """Reranker with caching support"""
    def __init__(self, model_id: str):
        self.model_id = model_id
        
        # Use cached model if available
        cache_dir = get_model_cache_dir(model_id)
        
        if is_model_cached(model_id):
            logger.info(f"Loading {model_id} from cache: {cache_dir}")
            self.tok = AutoTokenizer.from_pretrained(cache_dir, use_fast=False, local_files_only=True)
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(cache_dir, local_files_only=True)
                .eval()
                .to(DEVICE)
            )
        else:
            logger.info(f"Downloading and caching {model_id}...")
            self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(model_id)
                .eval()
                .to(DEVICE)
            )
            
            # Cache the model
            logger.info(f"Caching {model_id} to {cache_dir}")
            self.tok.save_pretrained(cache_dir)
            self.model.save_pretrained(cache_dir)
            logger.info(f"Model {model_id} cached successfully")

    @torch.inference_mode()
    def score(
        self, query: str, passages: list, batch_size=32, max_len=512
    ):
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