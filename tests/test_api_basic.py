#!/usr/bin/env python3
"""
Basic API test without ML dependencies
Tests the FastAPI structure and endpoints without model loading
"""

import time
import subprocess
import sys
import os
import signal
from pathlib import Path

def test_api_imports():
    """Test that the main API file can be imported without ML dependencies"""
    print("🔍 Testing API imports...")
    
    try:
        # Test basic FastAPI imports
        import fastapi
        import uvicorn
        import pydantic
        print("✅ Core API dependencies available")
    except ImportError as e:
        print(f"❌ Missing core dependencies: {e}")
        return False
    
    # Test if main.py can be parsed (syntax check)
    try:
        with open("main.py", "r") as f:
            code = f.read()
        compile(code, "main.py", "exec")
        print("✅ main.py syntax is valid")
    except Exception as e:
        print(f"❌ main.py syntax error: {e}")
        return False
    
    return True

def test_startup_script():
    """Test the startup script logic"""
    print("\n🚀 Testing startup script...")
    
    if not Path("main.py").exists():
        print("❌ main.py not found")
        return False
    
    if not Path("start_retriever_api.py").exists():
        print("❌ start_retriever_api.py not found")
        return False
    
    print("✅ Required files exist")
    
    # Test startup script syntax
    try:
        with open("start_retriever_api.py", "r") as f:
            code = f.read()
        compile(code, "start_retriever_api.py", "exec")
        print("✅ start_retriever_api.py syntax is valid")
    except Exception as e:
        print(f"❌ start_retriever_api.py syntax error: {e}")
        return False
    
    return True

def test_api_structure():
    """Test API endpoint structure by examining the code"""
    print("\n📊 Testing API structure...")
    
    try:
        with open("main.py", "r") as f:
            content = f.read()
        
        # Check for required endpoints
        required_endpoints = [
            '@app.get("/health")',
            '@app.post("/search',  # Partial match since it has response_model
            'app = FastAPI('
        ]
        
        missing_endpoints = []
        for endpoint in required_endpoints:
            if endpoint not in content:
                missing_endpoints.append(endpoint)
        
        if missing_endpoints:
            print(f"❌ Missing endpoints: {missing_endpoints}")
            return False
        
        print("✅ Required API endpoints found")
        
        # Check for required models
        required_models = ['QueryRequest', 'QueryResponse', 'RetrievalResult']
        missing_models = []
        for model in required_models:
            if f"class {model}" not in content:
                missing_models.append(model)
        
        if missing_models:
            print(f"❌ Missing models: {missing_models}")
            return False
        
        print("✅ Required Pydantic models found")
        return True
        
    except Exception as e:
        print(f"❌ Error checking API structure: {e}")
        return False

def test_example_scripts():
    """Test example scripts syntax"""
    print("\n📚 Testing example scripts...")
    
    scripts = ["test_retriever_api.py", "example_usage.py"]
    
    for script in scripts:
        if not Path(script).exists():
            print(f"❌ {script} not found")
            continue
        
        try:
            with open(script, "r") as f:
                code = f.read()
            compile(code, script, "exec")
            print(f"✅ {script} syntax is valid")
        except Exception as e:
            print(f"❌ {script} syntax error: {e}")
            return False
    
    return True

def test_requirements():
    """Test requirements.txt format"""
    print("\n📦 Testing requirements.txt...")
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        with open("requirements.txt", "r") as f:
            lines = f.readlines()
        
        # Check for key packages
        key_packages = ['fastapi', 'uvicorn', 'torch', 'transformers']
        content = ''.join(lines).lower()
        
        missing = []
        for pkg in key_packages:
            if pkg not in content:
                missing.append(pkg)
        
        if missing:
            print(f"❌ Missing key packages in requirements: {missing}")
            return False
        
        print("✅ requirements.txt looks good")
        print(f"   Found {len([l for l in lines if l.strip() and not l.startswith('#')])} dependencies")
        return True
        
    except Exception as e:
        print(f"❌ Error checking requirements.txt: {e}")
        return False

def main():
    """Run all basic tests"""
    print("🧪 OpenScholar API - Basic Structure Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_api_imports),
        ("Startup Script Test", test_startup_script),
        ("API Structure Test", test_api_structure),
        ("Example Scripts Test", test_example_scripts),
        ("Requirements Test", test_requirements)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        print("\n📋 Next Steps:")
        print("1. Install ML dependencies: pip install -r requirements.txt")
        print("2. Start the API: python start_retriever_api.py")
        print("3. Run full tests: python test_retriever_api.py")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)