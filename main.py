#!/usr/bin/env python3
"""
OpenScholar API - macOS Compatible Version
Fixes OpenMP conflicts on macOS
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fix OpenMP library conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
# Force CPU-only on macOS to avoid GPU conflicts
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
# Set single thread for macOS stability
torch.set_num_threads(1)

import time
import logging
import faiss
import numpy as np
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Configure logging based on environment
log_level = logging.DEBUG if os.getenv("DEBUG_LOGGING", "false").lower() == "true" else logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Security Configuration - Production ready
API_KEY = os.getenv("OPENSCHOLAR_API_KEY")
if not API_KEY:
    raise ValueError("OPENSCHOLAR_API_KEY environment variable is required. Please set it in your .env file.")

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Alternative: Header-based API key
def verify_api_key_header(x_api_key: str = Header(None)):
    """Verify API key from header"""
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key in X-API-Key header"
        )
    return x_api_key

# Global variables for the system state
index = None
id2meta = None
chunks_df = None
retriever = None
reranker = None
hash_to_chunk_id = None
chunk_id_to_row = None
initialization_error = None

# API Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    initial_topk: int = Field(default=200, ge=1, le=1000, description="Initial retrieval count")
    keep_for_rerank: int = Field(default=50, ge=1, le=200, description="Documents to rerank")
    final_topk: int = Field(default=10, ge=1, le=50, description="Final results count")
    per_paper_cap: int = Field(default=2, ge=1, le=10, description="Max results per paper")
    boost_mode: str = Field(default="mul", description="Boost mode: mul or add")
    boost_lambda: float = Field(default=0.1, ge=0.0, le=1.0, description="Boost lambda for add mode")
    max_length: int = Field(default=512, ge=128, le=1024, description="Max token length")

class RetrievalResult(BaseModel):
    rank: int
    hash_id: int
    paper_id: str
    section: str
    subsection: Optional[str]  # Allow None values
    paragraph_index: int
    boost: float
    text_preview: str
    retrieval_score: float
    rerank_score: float

class QueryResponse(BaseModel):
    query: str
    results_count: int
    processing_time: float
    results: List[RetrievalResult]

# F32-only initialization function
async def safe_initialize_system():
    """Initialize the system with F32-only, no quantization"""
    global index, id2meta, chunks_df, retriever, reranker, hash_to_chunk_id, chunk_id_to_row, initialization_error
    
    try:
        logger.info("🚀 Starting F32-only system initialization...")
        
        # Step 1: Import modules safely
        logger.info("Step 1: Importing modules...")
        from openscholar_api.models import Embedder, Reranker
        from openscholar_api.cache_manager import (
            setup_cache_directory, is_dataset_cached, load_dataset_cache,
            save_dataset_cache, cache_index_files, INDEX_CACHE_FILE, META_CACHE_FILE
        )
        from openscholar_api.core_processing import (
            RETRIEVER_MODEL, RERANKER_MODEL, load_dataset_and_index,
            apply_boosts, cap_per_paper, get_chunks_by_hash, make_header, load_meta_map
        )
        logger.info("✅ Modules imported successfully")
        
        # Step 2: Setup cache directory
        logger.info("Step 2: Setting up cache...")
        setup_cache_directory()
        logger.info("✅ Cache directory ready")
        
        # Step 3: Load data (cached or fresh)
        logger.info("Step 3: Loading dataset and index...")
        if is_dataset_cached():
            logger.info("Loading from cache...")
            chunks_df, hash_to_chunk_id, chunk_id_to_row = load_dataset_cache()
            index = faiss.read_index(str(INDEX_CACHE_FILE))
            id2meta = load_meta_map(str(META_CACHE_FILE))
            logger.info("✅ Cached data loaded")
        else:
            logger.info("Loading fresh data...")
            data = load_dataset_and_index()
            chunks_df = data['chunks_df']
            hash_to_chunk_id = data['hash_to_chunk_id']
            chunk_id_to_row = data['chunk_id_to_row']
            index = data['index']
            id2meta = data['id2meta']
            
            # Cache for future use
            save_dataset_cache(chunks_df, hash_to_chunk_id, chunk_id_to_row)
            cache_index_files(
                data['index_files']['index_path'],
                data['index_files']['meta_path']
            )
            logger.info("✅ Fresh data loaded and cached")
        
        # Step 4: Load models with F32-only
        logger.info("Step 4: Loading retriever model (F32-only)...")
        retriever = Embedder(RETRIEVER_MODEL)
        logger.info("✅ Retriever loaded successfully")
        
        logger.info("Step 5: Loading reranker model (F32-only)...")
        reranker = Reranker(RERANKER_MODEL)
        logger.info("✅ Reranker loaded successfully")
        
        logger.info("🎉 F32-only system initialization completed successfully!")
        
    except Exception as e:
        initialization_error = str(e)
        logger.error(f"💥 System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info("🚀 Starting OpenScholar F32-only API...")
    try:
        await safe_initialize_system()
        logger.info("✅ API startup completed")
    except Exception as e:
        logger.error(f"❌ API startup failed: {e}")
        # Don't raise - let the API start in error mode
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down OpenScholar API...")

# FastAPI app with lifespan
app = FastAPI(
    title="OpenScholar Retriever + Reranker API",
    description="Secure API for scientific document retrieval and reranking. Protected routes require API key authentication.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Public health check endpoint (no authentication required)"""
    if initialization_error:
        return {
            "status": "error",
            "version": "1.0.0",
            "security": "enabled",
            "tensor_type": "float32",
            "error": initialization_error,
            "models_loaded": {
                "retriever": retriever is not None,
                "reranker": reranker is not None,
                "index": index is not None
            }
        }
    
    return {
        "status": "healthy",
        "version": "2.3.0-secure",
        "security": "enabled",
        "tensor_type": "float32",
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

@app.post("/search", response_model=QueryResponse)
async def search(request: QueryRequest, api_key: str = Depends(verify_api_key_header)):
    """Perform retrieval and reranking with F32-only operations (Protected Route)"""
    if initialization_error:
        raise HTTPException(
            status_code=503, 
            detail=f"System initialization failed: {initialization_error}"
        )
    
    if index is None or retriever is None or reranker is None:
        raise HTTPException(status_code=503, detail="System not fully initialized")
    
    logger.info(f"🔍 F32-only search for query: '{request.query}'")
    start_time = time.time()
    
    try:
        # Import processing functions
        from openscholar_api.core_processing import (
            apply_boosts, cap_per_paper, get_chunks_by_hash, make_header
        )
        
        logger.info("Step 1: F32 embedding query...")
        # Embed query with F32-only
        q_vec = retriever.embed_texts(
            [request.query], batch_size=1, max_length=request.max_length
        ).astype('float32')
        logger.info(f"✅ Query embedded F32, shape: {q_vec.shape}")
        
        logger.info("Step 2: FAISS search...")
        # Retrieve initial candidates
        sims, ids = index.search(q_vec, request.initial_topk)
        sims, ids = sims[0], ids[0]
        logger.info(f"✅ FAISS search completed, found {len(ids)} candidates")
        
        logger.info("Step 3: Applying boosts...")
        # Apply boosts
        resc = apply_boosts(
            sims, ids, id2meta, 
            mode=request.boost_mode, 
            lam=request.boost_lambda
        )
        order = np.argsort(-resc)
        ids_sorted = ids[order]
        scores_sorted = resc[order]
        logger.info(f"✅ Boosts applied, sorted {len(ids_sorted)} results")
        
        logger.info("Step 4: Capping per paper...")
        # Cap per paper
        kept = cap_per_paper(
            ids_sorted.tolist(),
            scores_sorted.tolist(),
            id2meta,
            per_paper=request.per_paper_cap,
            keep=request.keep_for_rerank,
        )
        logger.info(f"✅ Paper capping completed, kept {len(kept)} results")
        
        logger.info("Step 5: Getting chunks...")
        # Get chunks for reranking
        kept_hash_ids = [iid for iid, _ in kept]
        results_df = get_chunks_by_hash(kept_hash_ids, chunks_df, hash_to_chunk_id, chunk_id_to_row)
        logger.info(f"✅ Retrieved chunks dataframe with {len(results_df)} rows")
        
        logger.info("Step 6: Creating text mapping...")
        # Create chunk_id to text mapping
        cid_to_text = {}
        for _, row in results_df.iterrows():
            cid_to_text[row["chunk_id"]] = row["text"]
        logger.info(f"✅ Created text mapping for {len(cid_to_text)} chunks")
        
        logger.info("Step 7: Building passages...")
        # Build passages for reranking
        passages, meta_list = [], []
        for iid, pre_sc in kept:
            md = id2meta[iid]
            cid = md["chunk_id"]
            body = cid_to_text.get(cid, "")
            header = make_header(md)
            passage = f"{header}\n\n{body}" if header else body
            passages.append(passage)
            meta_list.append((iid, pre_sc, md))
        logger.info(f"✅ Built {len(passages)} passages for reranking")
        
        logger.info("Step 8: F32-only reranking...")
        # Rerank with F32-only
        ce_scores = reranker.score(
            request.query, passages, 
            batch_size=16, max_len=request.max_length
        )
        logger.info(f"✅ F32 reranking completed, got {len(ce_scores)} scores")
        
        logger.info("Step 9: Final results...")
        # Sort by rerank scores
        order2 = np.argsort(-np.array(ce_scores))
        
        # Build results
        results = []
        for rank, idx in enumerate(order2[:request.final_topk], 1):
            iid, retrieval_score, md = meta_list[idx]
            rerank_score = ce_scores[idx]
            
            # Get text
            cid = md["chunk_id"]
            text = cid_to_text.get(cid, "")
            preview = (text[:240] + "…") if len(text) > 260 else text
            
            results.append(RetrievalResult(
                rank=rank,
                hash_id=int(iid),
                paper_id=md.get("paper_id", ""),
                section=md.get("section", ""),
                subsection=md.get("subsection"),  # Allow None
                paragraph_index=md.get("paragraph_index", 0),
                boost=md.get("boost", 1.0),
                text_preview=preview,
                retrieval_score=float(retrieval_score),
                rerank_score=float(rerank_score)
            ))
        
        processing_time = time.time() - start_time
        logger.info(f"🎉 F32-only search completed successfully in {processing_time:.2f}s")
        
        return QueryResponse(
            query=request.query,
            results_count=len(results),
            processing_time=processing_time,
            results=results
        )
        
    except Exception as e:
        logger.error(f"F32-only search error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/admin/stats")
async def admin_stats(api_key: str = Depends(verify_api_key_header)):
    """Protected admin endpoint for detailed system statistics"""
    import psutil
    import torch
    from openscholar_api.models import DEVICE, DTYPE
    
    return {
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
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

@app.post("/admin/reload")
async def admin_reload(api_key: str = Depends(verify_api_key_header)):
    """Protected admin endpoint to reload models (use with caution)"""
    return {
        "message": "Model reload not implemented yet - requires restart",
        "status": "info"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Load configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8002"))
    debug = os.getenv("DEBUG_LOGGING", "true").lower() == "true"
    
    logger.info(f"🚀 Starting OpenScholar API on {host}:{port}")
    if debug:
        logger.debug(f"🔑 API Key configured: {len(API_KEY)} characters")
    else:
        logger.info("🔐 API Key authentication enabled")
    
    uvicorn.run(app, host=host, port=port, log_level="info" if debug else "warning")