#!/usr/bin/env python3
"""
RunPod Serverless Handler for OpenScholar API
Adapts the FastAPI application for serverless execution
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def runpod_handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function
    
    Expected input format:
    {
        "input": {
            "query": "search query text",
            "initial_topk": 200,
            "keep_for_rerank": 50,  
            "final_topk": 10,
            "per_paper_cap": 2,
            "endpoint": "search" | "health" | "admin_stats"  # optional, defaults to "search"
        }
    }
    """
    try:
        # Get input from event
        input_data = event.get("input", {})
        endpoint = input_data.get("endpoint", "search")
        
        logger.info(f"Processing RunPod request for endpoint: {endpoint}")
        
        if endpoint == "health":
            return handle_health()
        elif endpoint == "admin_stats":
            return handle_admin_stats()
        elif endpoint == "search":
            return handle_search(input_data)
        else:
            return {
                "error": f"Unknown endpoint: {endpoint}",
                "available_endpoints": ["search", "health", "admin_stats"]
            }
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def handle_health() -> Dict[str, Any]:
    """Handle health check requests"""
    try:
        # Import here to avoid startup delays
        from main import index, retriever, reranker, chunks_df
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "type": "serverless",
            "models_loaded": {
                "retriever": retriever is not None,
                "reranker": reranker is not None, 
                "index": index is not None
            },
            "dataset_stats": {
                "total_chunks": len(chunks_df) if chunks_df is not None else 0,
                "index_size": index.ntotal if index is not None else 0
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def handle_admin_stats() -> Dict[str, Any]:
    """Handle admin stats requests"""
    try:
        import psutil
        import torch
        from openscholar_api.models import DEVICE, DTYPE
        from main import retriever, reranker, index, chunks_df, hash_to_chunk_id
        
        return {
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
            },
            "pytorch": {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": DEVICE,
                "dtype": str(DTYPE),
            },
            "models": {
                "retriever_loaded": retriever is not None,
                "reranker_loaded": reranker is not None,
                "index_loaded": index is not None,
            },
            "dataset": {
                "total_chunks": len(chunks_df) if chunks_df is not None else 0,
                "cache_size": len(hash_to_chunk_id) if hash_to_chunk_id is not None else 0,
            }
        }
    except Exception as e:
        return {
            "error": str(e)
        }

def handle_search(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle search requests"""
    try:
        # Validate required fields
        query = input_data.get("query")
        if not query:
            return {"error": "Missing required field: query"}
        
        # Set up search parameters with defaults
        search_params = {
            "query": query,
            "initial_topk": input_data.get("initial_topk", 200),
            "keep_for_rerank": input_data.get("keep_for_rerank", 50),
            "final_topk": input_data.get("final_topk", int(os.getenv("FINAL_TOPK", "10"))),
            "per_paper_cap": input_data.get("per_paper_cap", 2),
            "boost_mode": input_data.get("boost_mode", "mul"),
            "boost_lambda": input_data.get("boost_lambda", 0.1),
            "max_length": input_data.get("max_length", 512)
        }
        
        logger.info(f"Search params: {search_params}")
        
        # Run the search synchronously 
        result = asyncio.run(run_search(search_params))
        return result
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

async def run_search(params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the search operation"""
    import time
    import numpy as np
    from main import (
        index, id2meta, chunks_df, retriever, reranker, 
        hash_to_chunk_id, chunk_id_to_row, initialization_error
    )
    from openscholar_api.core_processing import (
        apply_boosts, cap_per_paper, get_chunks_by_hash, make_header
    )
    
    if initialization_error:
        return {"error": f"System initialization failed: {initialization_error}"}
    
    if index is None or retriever is None or reranker is None:
        return {"error": "System not fully initialized"}
    
    logger.info(f"Starting search for query: '{params['query']}'")
    start_time = time.time()
    
    try:
        # Step 1: Embed query
        logger.info("Step 1: Embedding query...")
        q_vec = retriever.embed_texts(
            [params["query"]], batch_size=1, max_length=params["max_length"]
        ).astype('float32')
        
        # Step 2: FAISS search
        logger.info("Step 2: FAISS search...")
        sims, ids = index.search(q_vec, params["initial_topk"])
        sims, ids = sims[0], ids[0]
        
        # Step 3: Apply boosts
        logger.info("Step 3: Applying boosts...")
        resc = apply_boosts(
            sims, ids, id2meta, 
            mode=params["boost_mode"], 
            lam=params["boost_lambda"]
        )
        order = np.argsort(-resc)
        ids_sorted = ids[order]
        scores_sorted = resc[order]
        
        # Step 4: Cap per paper
        logger.info("Step 4: Capping per paper...")
        kept = cap_per_paper(
            ids_sorted.tolist(),
            scores_sorted.tolist(),
            id2meta,
            per_paper=params["per_paper_cap"],
            keep=params["keep_for_rerank"],
        )
        
        # Step 5: Get chunks
        logger.info("Step 5: Getting chunks...")
        kept_hash_ids = [iid for iid, _ in kept]
        results_df = get_chunks_by_hash(kept_hash_ids, chunks_df, hash_to_chunk_id, chunk_id_to_row)
        
        # Step 6: Create text mapping
        cid_to_text = {}
        for _, row in results_df.iterrows():
            cid_to_text[row["chunk_id"]] = row["text"]
        
        # Step 7: Build passages for reranking
        logger.info("Step 7: Building passages...")
        passages, meta_list = [], []
        for iid, pre_sc in kept:
            md = id2meta[iid]
            cid = md["chunk_id"]
            body = cid_to_text.get(cid, "")
            header = make_header(md)
            passage = f"{header}\n\n{body}" if header else body
            passages.append(passage)
            meta_list.append((iid, pre_sc, md))
        
        # Step 8: Rerank
        logger.info("Step 8: Reranking...")
        ce_scores = reranker.score(
            params["query"], passages, 
            batch_size=16, max_len=params["max_length"]
        )
        
        # Step 9: Build final results
        logger.info("Step 9: Building final results...")
        # Sort by reranker scores (highest first)
        order2 = np.argsort(-np.array(ce_scores))
        
        results = []
        for rank, idx in enumerate(order2[:params["final_topk"]], 1):
            iid, retrieval_score, md = meta_list[idx]
            rerank_score = ce_scores[idx]
            
            passage = passages[idx]
            cid = md["chunk_id"]
            
            # Extract title and text
            if "\n\n" in passage:
                title_part, text_part = passage.split("\n\n", 1)
            else:
                title_part = ""
                text_part = passage
            
            # Clean up title by removing section suffixes
            def clean_title(title):
                if not title:
                    return title
                # Remove patterns like "— ABSTRACT", "— INTRODUCTION", etc.
                import re
                # Remove anything after " — " or " - " followed by uppercase words
                cleaned = re.sub(r'\s*[—-]\s*[A-Z][A-Z\s]*$', '', title.strip())
                return cleaned.strip()
            
            title_part = clean_title(title_part)
            
            # Remove title from text if it appears at the beginning
            if title_part and text_part.startswith(title_part):
                text_part = text_part[len(title_part):].lstrip("\n").strip()
            
            results.append({
                "reranker_score": float(rerank_score),
                "paper_id": md.get("paper_id", ""),
                "chunk_id": cid,
                "text": text_part,
                "title": title_part
            })
        
        processing_time = time.time() - start_time
        logger.info(f"Search completed successfully in {processing_time:.2f}s")
        
        return {
            "query": params["query"],
            "results_count": len(results),
            "processing_time": processing_time,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Search execution error: {e}")
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# For local testing
if __name__ == "__main__":
    # Test the handler locally
    test_event = {
        "input": {
            "query": "CRISPR gene editing mechanisms",
            "final_topk": 3
        }
    }
    
    result = runpod_handler(test_event)
    print(json.dumps(result, indent=2))