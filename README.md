# OpenScholar Retriever + Reranker API

A clean, efficient API for scientific document retrieval and reranking using OpenScholar models.

## 🎯 Overview

This API provides a focused implementation of scientific document search and ranking, extracted from the proven `retriever_+_reranker.py` pipeline. It performs retrieval and reranking only - no generation step - making it fast, reliable, and easy to deploy.

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd openscholar-api

# Install dependencies
pip install -r requirements.txt

# Start the API (downloads models automatically on first run)
python start_retriever_api.py

# Test it works
python test_retriever_api.py
```

The API will be available at **http://localhost:8002** with interactive documentation at **/docs**.

## 📁 Repository Structure

```
openscholar-api/
├── main.py                   # FastAPI application
├── start_retriever_api.py    # Easy startup script
├── test_retriever_api.py     # API tests
├── example_usage.py          # Usage examples
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── .gitignore               # Git ignore rules
```

## 🔬 Models Used

- **Retriever**: `bio-protocol/scientific-retriever` (BERT-based embedding model)
- **Reranker**: `bio-protocol/scientific-reranker` (Cross-encoder for scoring)
- **Dataset**: `bio-protocol/bio-faiss-longevity-v1` (Scientific paper corpus)

## 🔧 Features

- **Fast Retrieval**: FAISS-based vector search
- **Smart Reranking**: Cross-encoder reranking for improved relevance
- **Document Boosting**: Configurable boost factors for different document types
- **Per-paper Capping**: Ensures diversity in results across papers
- **GPU/CPU Support**: Automatic device detection and optimization
- **Auto-download**: Models and indices downloaded automatically
- **REST API**: Clean FastAPI interface with automatic documentation

## 📊 API Endpoints

### `GET /health`
System health check and model loading status.

### `POST /search`
Main search endpoint with configurable parameters:

- `query`: Search query string
- `initial_topk`: Initial retrieval count (default: 200)
- `keep_for_rerank`: Documents to rerank (default: 50)  
- `final_topk`: Final results returned (default: 10)
- `per_paper_cap`: Max results per paper (default: 2)
- `boost_mode`: Boosting mode - "mul" or "add"
- `max_length`: Max token length (default: 512)

**Request:**
```json
{
  "query": "Platelet-derived transcription factors",
  "initial_topk": 200,
  "keep_for_rerank": 50, 
  "final_topk": 10,
  "per_paper_cap": 2,
  "boost_mode": "mul",
  "boost_lambda": 0.1,
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
      "subsection": "Cell Analysis",
      "paragraph_index": 3,
      "boost": 1.2,
      "text_preview": "Platelet-derived transcription factors...",
      "retrieval_score": 0.8534,
      "rerank_score": 2.1567
    }
  ]
}
```

## Configuration

All configuration is handled via request parameters:

- `initial_topk`: Initial retrieval count (1-1000, default: 200)
- `keep_for_rerank`: Documents to rerank (1-200, default: 50)
- `final_topk`: Final results (1-50, default: 10)
- `per_paper_cap`: Max results per paper (1-10, default: 2)
- `boost_mode`: "mul" or "add" (default: "mul")
- `boost_lambda`: Lambda for add mode (0.0-1.0, default: 0.1)
- `max_length`: Max tokens (128-1024, default: 512)

## Example Usage

### Python Client

```python
import requests

# Search for documents
response = requests.post('http://localhost:8002/search', json={
    'query': 'How does CRISPR gene editing work?',
    'final_topk': 5
})

results = response.json()
for result in results['results']:
    print(f"Rank {result['rank']}: {result['text_preview']}")
```

### cURL

```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mitochondrial complex I function",
    "final_topk": 3
  }'
```

## Architecture

The API is built from the proven `retriever_+_reranker.py` pipeline:

1. **Query Embedding**: Scientific retriever encodes the query
2. **FAISS Search**: Retrieves initial candidates from bio-protocol dataset  
3. **Boost Application**: Applies document-specific boosts
4. **Per-paper Capping**: Ensures diversity across papers
5. **Reranking**: Cross-encoder scores query-passage pairs
6. **Results**: Returns ranked, scored passages with metadata

## Hardware Requirements

- **GPU Recommended**: CUDA-capable GPU for faster inference
- **CPU Fallback**: Works on CPU but slower
- **Memory**: ~4GB RAM minimum, ~8GB recommended
- **Storage**: ~2GB for models and index (downloaded automatically)

## Models Used

- **Retriever**: `bio-protocol/scientific-retriever` (BERT-based)
- **Reranker**: `bio-protocol/scientific-reranker` (Cross-encoder)  
- **Dataset**: `bio-protocol/bio-faiss-longevity-v1` (Scientific papers)

## ⚡ Performance

- **Cold start**: ~30-60s (model loading)
- **Query time**: ~1-3s per query
- **Throughput**: ~20-30 queries/minute (single worker)
- **Memory**: ~4-8GB RAM recommended
- **Storage**: ~2GB for models and indices

## 🛠️ Development

### Running Tests
```bash
python test_retriever_api.py
```

### Running Examples
```bash
python example_usage.py
```

### Custom Configuration
All parameters are configurable via the API request parameters.

## 📈 Architecture

The API follows a simple, efficient pipeline:

1. **Query Embedding** → Encode user query with scientific retriever
2. **Vector Search** → FAISS retrieval from scientific paper corpus
3. **Boost Application** → Apply document-specific relevance boosts
4. **Per-paper Filtering** → Ensure result diversity across papers
5. **Reranking** → Cross-encoder scoring of query-passage pairs
6. **Result Formatting** → Return ranked results with metadata

## 🔄 Migration from Legacy

This clean implementation replaces the previous complex pipeline:

- ✅ **Simpler**: Focused on retrieval + reranking only
- ✅ **Faster**: Removed generation bottleneck  
- ✅ **Reliable**: Based on proven working code
- ✅ **Maintainable**: Clean structure
- ✅ **Deployable**: Ready for production use

Generation functionality can be added later as a separate service.

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your changes
5. Submit a pull request

## 📞 Support

- **API Documentation**: Visit `/docs` endpoint when running
- **Issues**: Open GitHub issues for bugs or feature requests
- **Examples**: Check `example_usage.py` for usage patterns