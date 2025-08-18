# RunPod Hub Deployment Guide

[![Runpod](https://api.runpod.io/badge/caelusway/openscholar-api)](https://console.runpod.io/hub/caelusway/openscholar-api)

## RunPod Hub Serverless Deployment

This project is configured for **RunPod Hub** serverless deployment with automatic GPU scaling.

## Quick Setup for RunPod Hub

### 1. Prepare Environment File
Copy `.env.runpod` to `.env` and update the API key:
```bash
cp .env.runpod .env
# Edit .env and change OPENSCHOLAR_API_KEY to a secure value
```

### 2. Build Docker Image
```bash
docker build -t openscholar-api:latest .
```

### 3. Test Locally with GPU (Optional)
```bash
docker-compose up --build
```

### 4. RunPod Hub Configuration

The project includes:
- `.runpod/hub.json` - Hub listing configuration
- `.runpod/tests.json` - Automated testing configuration
- `handler.py` - Serverless handler function

**Supported Endpoints:**
- `health` - System health check (no auth required)
- `search` - Main search functionality (requires API key)
- `admin_stats` - System statistics (requires API key)

**GPU Requirements:**
- **Minimum:** 8GB VRAM (RTX 3070, RTX 4060 Ti)
- **Recommended:** 16GB+ VRAM (RTX 4080, RTX 4090, A40, A100)

### 5. Serverless API Usage

**Health Check:**
```python
import requests

response = requests.post("https://api.runpod.ai/v2/your-endpoint-id/run", json={
    "input": {"endpoint": "health"}
})
```

**Search Request:**
```python
response = requests.post("https://api.runpod.ai/v2/your-endpoint-id/run", json={
    "input": {
        "query": "CRISPR gene editing mechanisms",
        "final_topk": 5
    }
})
```

**Advanced Search:**
```python
response = requests.post("https://api.runpod.ai/v2/your-endpoint-id/run", json={
    "input": {
        "query": "machine learning drug discovery",
        "initial_topk": 100,
        "keep_for_rerank": 30,
        "final_topk": 10,
        "per_paper_cap": 2
    }
})
```

## Performance Optimization

### GPU Memory Management
- Models automatically use bfloat16 on GPU for efficiency
- CUDA memory allocation configured for optimal performance
- Batch sizes optimized for GPU: embedding=64, reranking=32

### Timeout Settings
- **GPU Mode:** 30s search timeout, 60s request timeout
- **First-time startup:** Allow 2-3 minutes for model downloads

### Caching
- Models cached in `/app/model_cache` volume
- Persistent across container restarts
- ~2GB storage for cached models

## Troubleshooting

**Out of Memory:**
- Reduce batch sizes in environment variables
- Use smaller GPU or increase VRAM

**Slow Performance:**
- Verify GPU is detected: Check `/admin/stats` endpoint
- Ensure CUDA drivers are properly installed

**Model Download Issues:**
- Check internet connectivity
- Verify HuggingFace access (models are public)

## Security Notes
- Always change the default API key
- API endpoints are protected except `/health`
- CORS configured for production use