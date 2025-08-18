# RunPod GPU Deployment Guide

## Quick Setup for RunPod

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

### 4. RunPod Template Configuration

**Container Settings:**
- **Image:** Upload your built image or use a registry
- **Container Disk:** 10GB minimum
- **Exposed HTTP Port:** 8002
- **Environment Variables:**
  ```
  OPENSCHOLAR_API_KEY=your-secure-api-key-here
  PRODUCTION_MODE=true
  DEBUG_LOGGING=false
  CUDA_VISIBLE_DEVICES=0
  ```

**GPU Requirements:**
- **Minimum:** 8GB VRAM (RTX 3070, RTX 4060 Ti)
- **Recommended:** 16GB+ VRAM (RTX 4080, RTX 4090, A40, A100)

### 5. Health Check Endpoint
After deployment, verify the API is running:
```
GET https://your-runpod-url.com/health
```

### 6. API Usage
Protected endpoints require the API key in the `X-API-Key` header:
```bash
curl -X POST "https://your-runpod-url.com/search" \
  -H "X-API-Key: your-secure-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"query": "CRISPR gene editing mechanisms"}'
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