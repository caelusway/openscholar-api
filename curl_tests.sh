#!/bin/bash
# OpenScholar API - Curl Test Commands
# Make sure the API is running first: python start_retriever_api.py

API_URL="http://localhost:8002"

echo "🧪 OpenScholar API - Curl Tests"
echo "================================="
echo "API URL: $API_URL"
echo

# Test 1: Health Check
echo "1. 🏥 Testing Health Endpoint..."
echo "curl -X GET '$API_URL/health'"
echo
curl -X GET "$API_URL/health" \
  -H "Content-Type: application/json" \
  | jq '.' 2>/dev/null || echo "Raw response (install jq for pretty JSON)"
echo -e "\n"

# Test 2: Basic Search
echo "2. 🔍 Testing Basic Search..."
echo "curl -X POST '$API_URL/search' with basic query"
echo
curl -X POST "$API_URL/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "CRISPR gene editing mechanism",
    "final_topk": 3
  }' \
  | jq '.' 2>/dev/null || echo "Raw response (install jq for pretty JSON)"
echo -e "\n"

# Test 3: Advanced Search with Parameters
echo "3. ⚙️ Testing Advanced Search with Parameters..."
echo "curl -X POST '$API_URL/search' with custom parameters"
echo
curl -X POST "$API_URL/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mitochondrial complex I function",
    "initial_topk": 100,
    "keep_for_rerank": 20,
    "final_topk": 5,
    "per_paper_cap": 2,
    "boost_mode": "mul",
    "max_length": 512
  }' \
  | jq '.' 2>/dev/null || echo "Raw response (install jq for pretty JSON)"
echo -e "\n"

# Test 4: Scientific Query
echo "4. 🧬 Testing Scientific Query..."
echo "curl -X POST '$API_URL/search' with complex scientific query"
echo
curl -X POST "$API_URL/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the mechanisms of autophagy in neurodegenerative diseases?",
    "final_topk": 5,
    "per_paper_cap": 1
  }' \
  | jq '.' 2>/dev/null || echo "Raw response (install jq for pretty JSON)"
echo -e "\n"

# Test 5: Error Handling - Invalid Endpoint
echo "5. ❌ Testing Error Handling (Invalid Endpoint)..."
echo "curl -X GET '$API_URL/invalid'"
echo
curl -X GET "$API_URL/invalid" \
  -H "Content-Type: application/json"
echo -e "\n"

# Test 6: Error Handling - Invalid Request Body
echo "6. ❌ Testing Error Handling (Invalid Request)..."
echo "curl -X POST '$API_URL/search' with invalid data"
echo
curl -X POST "$API_URL/search" \
  -H "Content-Type: application/json" \
  -d '{
    "invalid_field": "test"
  }'
echo -e "\n"

echo "✅ All curl tests completed!"
echo
echo "💡 Tips:"
echo "- Install 'jq' for pretty JSON formatting: brew install jq (macOS) or apt-get install jq (Linux)"
echo "- Add -v flag to curl for verbose output: curl -v ..."
echo "- Add -w '\\n\\nTotal time: %{time_total}s\\n' to measure response time"
echo
echo "🎯 Quick one-liners for testing:"
echo
echo "# Health check:"
echo "curl -X GET $API_URL/health | jq '.status'"
echo
echo "# Quick search:"
echo "curl -X POST $API_URL/search -H 'Content-Type: application/json' -d '{\"query\":\"COVID-19 vaccine\",\"final_topk\":3}' | jq '.results[0].text_preview'"
echo
echo "# Performance test:"
echo "curl -X POST $API_URL/search -H 'Content-Type: application/json' -d '{\"query\":\"machine learning\",\"final_topk\":1}' -w '\\nResponse time: %{time_total}s\\n'"