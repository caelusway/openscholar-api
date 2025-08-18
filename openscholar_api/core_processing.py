#!/usr/bin/env python3
"""
Core processing functions for OpenScholar API
Contains the original working retrieval and reranking logic
"""

import hashlib
import json
import faiss
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

# Configuration
REPO_ID = "bio-protocol/bio-faiss-longevity-v1"
RETRIEVER_MODEL = "bio-protocol/scientific-retriever"
RERANKER_MODEL = "bio-protocol/scientific-reranker"


def stable64(s: str) -> np.int64:
    """Generate stable 64-bit hash - Original working implementation"""
    if hasattr(faiss, "hash64"):
        return np.int64(faiss.hash64(s))
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return np.frombuffer(h, dtype="<i8")[0]


def load_meta_map(meta_path: str) -> Dict[int, dict]:
    """Load metadata mapping - Original working implementation"""
    id2meta = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            md = json.loads(line)
            iid = int(stable64(md["chunk_id"]))
            id2meta[iid] = md
    return id2meta


def apply_boosts(
    sim: np.ndarray, ids: np.ndarray, id2meta: Dict[int, dict], mode="mul", lam=0.1
) -> np.ndarray:
    """Apply boosting to similarity scores - Original working implementation"""
    boosts = np.array(
        [id2meta.get(int(i), {}).get("boost", 1.0) for i in ids], dtype=np.float32
    )
    if mode == "mul":
        return sim * boosts
    return sim + lam * np.log(np.clip(boosts, 1e-6, None))


def cap_per_paper(
    ids_sorted: List[int],
    scores_sorted: List[float],
    id2meta: Dict[int, dict],
    per_paper=2,
    keep=50,
) -> List[Tuple[int, float]]:
    """Cap results per paper - Original working implementation"""
    per_count = defaultdict(int)
    taken = []
    for iid, sc in zip(ids_sorted, scores_sorted):
        md = id2meta.get(int(iid), {})
        pid = md.get("paper_id")
        if pid is None:
            continue
        if per_count[pid] >= per_paper:
            continue
        per_count[pid] += 1
        taken.append((int(iid), float(sc)))
        if len(taken) >= keep:
            break
    return taken


def get_chunks_by_hash(chunk_hashes: List[int], chunks_df: pd.DataFrame, 
                      hash_to_chunk_id: dict, chunk_id_to_row: dict) -> pd.DataFrame:
    """Get chunk data by hash values - Original working implementation"""
    # Convert hashes to chunk_ids
    chunk_ids = [hash_to_chunk_id.get(h) for h in chunk_hashes if h in hash_to_chunk_id]

    # Get rows by chunk_ids
    indices = [chunk_id_to_row[cid] for cid in chunk_ids if cid in chunk_id_to_row]

    return chunks_df.iloc[indices]


def make_header(md: dict) -> str:
    """Create header from metadata - Original working implementation"""
    title = md.get("title", "")
    section = (md.get("section") or "").upper()
    subsection = md.get("subsection")
    hdr = f"{title} — {section}" if title or section else ""
    if subsection:
        hdr += f" / {subsection}"
    return hdr


def load_dataset_and_index():
    """Load dataset and FAISS index - Original working logic"""
    logger.info("Loading dataset and FAISS index...")
    
    # Load dataset
    ds = load_dataset(REPO_ID)
    chunks_df = ds['train'].to_pandas()
    
    # Create hash mappings
    hash_to_chunk_id = dict(zip(chunks_df["chunk_hash"], chunks_df["chunk_id"]))
    chunk_id_to_row = dict(zip(chunks_df["chunk_id"], chunks_df.index))
    
    # Download index and metadata
    index_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="index.faiss",
        repo_type="dataset"
    )
    
    meta_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="meta.jsonl",
        repo_type="dataset"
    )
    
    logger.info(f"Downloaded index to: {index_path}")
    logger.info(f"Downloaded metadata to: {meta_path}")
    
    # Load FAISS index
    import os
    index_dir = index_path.replace("/index.faiss", "")
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    
    # Load metadata
    id2meta = load_meta_map(os.path.join(index_dir, "meta.jsonl"))
    
    return {
        'chunks_df': chunks_df,
        'hash_to_chunk_id': hash_to_chunk_id,
        'chunk_id_to_row': chunk_id_to_row,
        'index': index,
        'id2meta': id2meta,
        'index_files': {
            'index_path': os.path.join(index_dir, "index.faiss"),
            'meta_path': os.path.join(index_dir, "meta.jsonl")
        }
    }