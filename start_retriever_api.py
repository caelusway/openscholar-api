#!/usr/bin/env python3
"""
Startup script for the Retriever + Reranker API
"""

import sys
import subprocess
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        "fastapi", "uvicorn", "torch", "transformers", 
        "datasets", "faiss-cpu", "huggingface_hub"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("📦 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def main():
    print("🚀 Starting Retriever + Reranker API...")
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ Please run this script from the openscholar-api root directory")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    print("🔧 Starting API server on http://localhost:8002")
    print("📚 API docs will be available at http://localhost:8002/docs")
    print("⏳ Initial loading may take a few minutes to download models...")
    print("📋 Press Ctrl+C to stop the server")
    print()
    
    # Start the API
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app",
            "--host", "0.0.0.0",
            "--port", "8002",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")

if __name__ == "__main__":
    main()