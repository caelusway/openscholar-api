#!/usr/bin/env python3
"""
Test script for the Retriever + Reranker API
"""

import requests
import json
import time
import sys

API_URL = "http://localhost:8002"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Device: {data.get('device')}")
            print(f"   Models loaded: {data.get('models_loaded')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_search():
    """Test search endpoint"""
    print("\n🔍 Testing search endpoint...")
    
    test_query = "Platelet-derived transcription factors license human monocyte inflammation"
    
    payload = {
        "query": test_query,
        "initial_topk": 100,
        "keep_for_rerank": 20,
        "final_topk": 5,
        "per_paper_cap": 2
    }
    
    try:
        print(f"   Query: {test_query}")
        start_time = time.time()
        
        response = requests.post(f"{API_URL}/search", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            elapsed = time.time() - start_time
            
            print(f"✅ Search completed in {elapsed:.2f}s")
            print(f"   Results found: {data['results_count']}")
            print(f"   Processing time: {data['processing_time']:.2f}s")
            
            if data['results']:
                print("\n📄 Top result:")
                result = data['results'][0]
                print(f"   Rank: {result['rank']}")
                print(f"   Paper ID: {result['paper_id']}")
                print(f"   Section: {result['section']}")
                print(f"   Retrieval score: {result['retrieval_score']:.4f}")
                print(f"   Rerank score: {result['rerank_score']:.4f}")
                print(f"   Preview: {result['text_preview'][:100]}...")
            
            return True
        else:
            print(f"❌ Search failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False

def main():
    print("🧪 Testing Retriever + Reranker API")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("\n❌ Health check failed. Make sure the API is running.")
        print("Start with: python start_retriever_api.py")
        sys.exit(1)
    
    # Wait a moment for models to load if they're still loading
    print("\n⏳ Waiting for models to finish loading...")
    time.sleep(2)
    
    # Test search
    if not test_search():
        print("\n❌ Search test failed")
        sys.exit(1)
    
    print("\n🎉 All tests passed!")
    print(f"📚 Visit {API_URL}/docs for interactive API documentation")

if __name__ == "__main__":
    main()