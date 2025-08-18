# 🧪 OpenScholar API - Curl Test Commands

Make sure the API is running first:
```bash
python start_retriever_api.py
# or
uvicorn main:app --host 0.0.0.0 --port 8002
```

## 🏥 1. Health Check

```bash
curl -X GET "http://localhost:8002/health" \
  -H "Content-Type: application/json"
```

**Expected Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0 (modular)",
  "models_loaded": {
    "retriever": true,
    "reranker": true,
    "index": true
  },
  "dataset_stats": {
    "total_chunks": 138065,
    "index_size": 138065
  }
}
```

## 🔍 2. Basic Search

```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "CRISPR gene editing mechanism",
    "final_topk": 3
  }'
```

## ⚙️ 3. Advanced Search with All Parameters

```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mitochondrial complex I function",
    "initial_topk": 100,
    "keep_for_rerank": 20,
    "final_topk": 5,
    "per_paper_cap": 2,
    "boost_mode": "mul",
    "boost_lambda": 0.1,
    "max_length": 512
  }'
```

## 🧬 4. Scientific Queries (Examples)

### COVID-19 Research
```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "COVID-19 vaccine efficacy and immune response",
    "final_topk": 5
  }'
```

### Cancer Research
```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "p53 tumor suppressor gene mutations in cancer",
    "final_topk": 4,
    "per_paper_cap": 1
  }'
```

### Neuroscience
```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "synaptic plasticity and memory formation",
    "final_topk": 3,
    "boost_mode": "add"
  }'
```

### Biochemistry
```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "protein folding mechanisms and chaperones",
    "final_topk": 6
  }'
```

## ⏱️ 5. Performance Testing

```bash
# Measure response time
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning in biology",
    "final_topk": 1
  }' \
  -w "\nResponse time: %{time_total}s\n"
```

## 🎯 6. Quick One-Liners

### Health Status Only
```bash
curl -s -X GET "http://localhost:8002/health" | jq '.status'
```

### Quick Search Result
```bash
curl -s -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"stem cells","final_topk":1}' | jq '.results[0].text_preview'
```

### Results Count
```bash
curl -s -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"immunotherapy","final_topk":5}' | jq '.results_count'
```

### Processing Time
```bash
curl -s -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"gene therapy","final_topk":3}' | jq '.processing_time'
```

## ❌ 7. Error Testing

### Invalid Endpoint
```bash
curl -X GET "http://localhost:8002/invalid"
```

### Invalid Request Body
```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{"invalid_field": "test"}'
```

### Empty Query
```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{"query": ""}'
```

## 📊 8. Batch Testing Script

Run all tests at once:
```bash
./curl_tests.sh
```

## 💡 Tips

1. **Install jq for pretty JSON**: `brew install jq` (macOS) or `apt-get install jq` (Linux)

2. **Verbose output**: Add `-v` flag to see detailed request/response info

3. **Save response**: Add `-o response.json` to save response to file

4. **Headers only**: Add `-I` flag to see only response headers

5. **Silent mode**: Add `-s` flag to suppress progress bar

6. **Timeout**: Add `--connect-timeout 10` to set connection timeout

## 🚀 Example Response

```json
{
  "query": "CRISPR gene editing mechanism",
  "results_count": 3,
  "processing_time": 2.14,
  "results": [
    {
      "rank": 1,
      "hash_id": 12345678,
      "paper_id": "PMC7234567",
      "section": "RESULTS",
      "subsection": "Gene Editing Analysis",
      "paragraph_index": 2,
      "boost": 1.2,
      "text_preview": "CRISPR-Cas9 system enables precise genome editing through guide RNA-directed DNA cleavage...",
      "retrieval_score": 0.8924,
      "rerank_score": 2.1456
    }
  ]
}
```