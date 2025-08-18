#!/usr/bin/env python3
"""
Example usage of the Retriever + Reranker API
"""

import requests
import json
import time

API_URL = "http://localhost:8002"

def example_queries():
    """Run example queries to demonstrate the API"""
    
    queries = [
        "Platelet-derived transcription factors license human monocyte inflammation",
        "How does CRISPR gene editing work?",
        "What is mitochondrial complex I?",
        "Explain the mechanism of photosynthesis",
        "What causes Alzheimer's disease?"
    ]
    
    print("🔍 Running example queries...")
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 40)
        
        # Configure search parameters
        payload = {
            "query": query,
            "initial_topk": 100,      # Retrieve 100 initial candidates
            "keep_for_rerank": 20,    # Rerank top 20
            "final_topk": 3,          # Return top 3 results
            "per_paper_cap": 2,       # Max 2 results per paper
            "boost_mode": "mul",      # Multiplicative boosting
            "max_length": 512         # Max token length
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{API_URL}/search", json=payload)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Found {data['results_count']} results in {elapsed:.2f}s")
                print(f"   Processing time: {data['processing_time']:.2f}s")
                
                for result in data['results']:
                    print(f"\n   📄 Rank {result['rank']}:")
                    print(f"      Paper: {result['paper_id']}")
                    print(f"      Section: {result['section']}")
                    if result['subsection']:
                        print(f"      Subsection: {result['subsection']}")
                    print(f"      Retrieval score: {result['retrieval_score']:.4f}")
                    print(f"      Rerank score: {result['rerank_score']:.4f}")
                    print(f"      Preview: {result['text_preview'][:120]}...")
                    
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to API. Is the server running?")
            print("   Start with: python start_retriever_api.py")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Small delay between queries
        time.sleep(1)

def compare_parameters():
    """Compare results with different parameters"""
    
    query = "mitochondrial complex I function"
    
    configs = [
        {
            "name": "High Precision",
            "params": {
                "initial_topk": 50,
                "keep_for_rerank": 10,
                "final_topk": 3,
                "per_paper_cap": 1
            }
        },
        {
            "name": "High Recall", 
            "params": {
                "initial_topk": 200,
                "keep_for_rerank": 50,
                "final_topk": 10,
                "per_paper_cap": 3
            }
        },
        {
            "name": "Fast Search",
            "params": {
                "initial_topk": 50,
                "keep_for_rerank": 15,
                "final_topk": 5,
                "per_paper_cap": 2
            }
        }
    ]
    
    print(f"\n🔬 Comparing configurations for: '{query}'")
    print("=" * 60)
    
    for config in configs:
        print(f"\n📊 {config['name']} Configuration:")
        print("-" * 30)
        
        payload = {"query": query, **config['params']}
        
        try:
            start_time = time.time()
            response = requests.post(f"{API_URL}/search", json=payload)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ {data['results_count']} results in {elapsed:.2f}s")
                
                if data['results']:
                    top_result = data['results'][0]
                    print(f"   Top result score: {top_result['rerank_score']:.4f}")
                    print(f"   Paper: {top_result['paper_id']}")
                    
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    print("🚀 Retriever + Reranker API Examples")
    print("Make sure the API is running: python start_retriever_api.py")
    print()
    
    # Check if API is running
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code != 200:
            print("❌ API health check failed")
            return
    except:
        print("❌ Cannot connect to API. Start with: python start_retriever_api.py")
        return
    
    # Run examples
    example_queries()
    compare_parameters()
    
    print(f"\n🎉 Examples completed!")
    print(f"📚 Visit {API_URL}/docs for interactive API documentation")

if __name__ == "__main__":
    main()