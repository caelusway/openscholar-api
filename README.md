# OpenScholar Retriever + Reranker API

A clean, efficient API for scientific document retrieval and reranking using OpenScholar models.

## Overview

This API provides a focused implementation of scientific document search and ranking. It performs retrieval and reranking only - no generation step - making it fast, reliable, and easy to deploy.

## Quick Start

### Local Development
```bash
git clone https://github.com/bio-xyz/open-scholar-inference.git
cd open-scholar-inference

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and set OPENSCHOLAR_API_KEY=your-secure-api-key

# Start the API (downloads models automatically on first run)
python main.py
```

### RunPod Deployment (Recommended)

**One-time setup on RunPod server:**

1. Connect to your RunPod instance:
```bash
ssh root@your-runpod-ip -p port -i ~/.ssh/id_ed25519
```

2. Download and run the setup script:
```bash
curl -O https://raw.githubusercontent.com/bio-xyz/open-scholar-inference/main/runpod-setup.sh
chmod +x runpod-setup.sh
OPENSCHOLAR_API_KEY='your-actual-api-key' bash runpod-setup.sh
```

3. The API will be available at **http://localhost:8002** with interactive documentation at **/docs**.

**For updates:**
```bash
curl -O https://raw.githubusercontent.com/bio-xyz/open-scholar-inference/main/runpod-update.sh
chmod +x runpod-update.sh
bash runpod-update.sh
```

**Service management:**
```bash
# Check status
/workspace/start-openscholar.sh status

# View logs
tail -f /workspace/logs/app.log

# Restart application
/workspace/start-openscholar.sh restart

# Stop application
/workspace/start-openscholar.sh stop
```

### Docker Deployment
```bash
docker run -p 8002:8002 -e OPENSCHOLAR_API_KEY=YOUR_SECURE_API_KEY bio-xyz/open-scholar-inference:latest
```

## Models Used

- **Retriever**: `bio-protocol/scientific-retriever` (BERT-based embedding model)
- **Reranker**: `bio-protocol/scientific-reranker` (Cross-encoder for scoring)
- **Dataset**: `bio-protocol/bio-faiss-longevity-v1` (Scientific paper corpus)

## Features

- **Fast Retrieval**: FAISS-based vector search
- **Smart Reranking**: Cross-encoder reranking for improved relevance
- **Document Boosting**: Configurable boost factors for different document types
- **Per-paper Capping**: Ensures diversity in results across papers
- **GPU/CPU Support**: Automatic device detection and optimization
- **Auto-download**: Models and indices downloaded automatically
- **REST API**: Clean FastAPI interface with automatic documentation
- **API Key Authentication**: Secure access to protected endpoints

## API Endpoints

### `GET /health`
System health check and model loading status. No authentication required.

### `POST /search`
Main search endpoint with configurable parameters. Requires API key in `X-API-Key` header.

**Request:**
```json
{
  "query": "Platelet-derived transcription factors",
  "initial_topk": 200,
  "keep_for_rerank": 50, 
  "final_topk": 10,
  "per_paper_cap": 2,
  "boost_mode": "mul",
  "max_length": 512
}
```

**Response:**
```json
{
  "query": "Platelet-derived transcription factors",
  "results_count": 10,
  "processing_time": 2.45,
  "results": [
    {
      "rank": 1,
      "hash_id": 12345,
      "paper_id": "PMC123456",
      "section": "RESULTS",
      "text_preview": "Platelet-derived transcription factors...",
      "retrieval_score": 0.8534,
      "rerank_score": 2.1567
    }
  ]
}
```

## Configuration

Set environment variables in `.env` file:

```bash
# API Security - REQUIRED
OPENSCHOLAR_API_KEY=CHANGE_THIS_TO_A_SECURE_KEY

# Server Configuration
API_HOST=0.0.0.0
API_PORT=8002

# Model Configuration
RETRIEVER_MODEL=bio-protocol/scientific-retriever
RERANKER_MODEL=bio-protocol/scientific-reranker

# Performance Settings
MAX_WORKERS=1
BATCH_SIZE_EMBEDDING=64
BATCH_SIZE_RERANKING=32

# Timeout Settings
REQUEST_TIMEOUT=120
SEARCH_TIMEOUT=180

# Cache Configuration
CACHE_DIR=./model_cache
ENABLE_CACHE=true

# Production Settings
PRODUCTION_MODE=false
DEBUG_LOGGING=true
```

## Example Usage

### Python Client
```python
import requests

headers = {"X-API-Key": "YOUR_SECURE_API_KEY"}
response = requests.post('http://localhost:8002/search', 
    json={'query': 'How does CRISPR gene editing work?', 'final_topk': 5},
    headers=headers)

results = response.json()
for result in results['results']:
    print(f"Rank {result['rank']}: {result['text_preview']}")
```

### cURL
```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_SECURE_API_KEY" \
  -d '{"query": "mitochondrial complex I function", "final_topk": 3}'
```

## Architecture

The API follows a simple, efficient pipeline:

1. **Query Embedding** → Encode user query with scientific retriever
2. **Vector Search** → FAISS retrieval from scientific paper corpus
3. **Boost Application** → Apply document-specific relevance boosts
4. **Per-paper Filtering** → Ensure result diversity across papers
5. **Reranking** → Cross-encoder scoring of query-passage pairs
6. **Result Formatting** → Return ranked results with metadata

## Hardware Requirements

- **GPU Recommended**: CUDA-capable GPU for faster inference
- **CPU Fallback**: Works on CPU but slower
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB for models and index (downloaded automatically)

## Performance

- **Cold start**: 30-60s (model loading)
- **Query time**: 1-3s per query
- **Throughput**: 20-30 queries/minute (single worker)
- **Memory**: 4-8GB RAM recommended

## Security

- All API endpoints except `/health` require authentication
- Set `OPENSCHOLAR_API_KEY` in environment variables
- Never commit API keys to version control
- Use HTTPS in production

## Deployment Scripts

The repository includes automated deployment scripts for RunPod:

- **`runpod-setup.sh`** - Initial setup with systemd service creation
- **`runpod-update.sh`** - Quick updates from GitHub

These scripts handle:
- Automatic dependency installation
- Process management with PID tracking
- Logging and monitoring
- Environment variable configuration
- Simple start/stop/restart/status commands

## Development

### Running Tests
```bash
python test_retriever_api.py
```

### Running Examples
```bash
python example_usage.py
```

## License

MIT License - see LICENSE file for details.