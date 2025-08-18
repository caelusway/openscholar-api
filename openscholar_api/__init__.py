#!/usr/bin/env python3
"""
OpenScholar API Package
Modular components for scientific document retrieval and reranking
"""

from .models import Embedder, Reranker
from .cache_manager import (
    setup_cache_directory, is_dataset_cached, load_dataset_cache,
    save_dataset_cache, cache_index_files, CachedEmbedder, CachedReranker
)
from .core_processing import (
    RETRIEVER_MODEL, RERANKER_MODEL, load_dataset_and_index,
    apply_boosts, cap_per_paper, get_chunks_by_hash, make_header,
    stable64, load_meta_map
)

__version__ = "2.0.0"
__author__ = "OpenScholar API Team"
__description__ = "Modular API for scientific document retrieval and reranking"