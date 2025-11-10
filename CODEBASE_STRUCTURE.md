# OpenScholar API Codebase Structure

## Overview
This document describes the organization of the OpenScholar API codebase after cleanup (November 10, 2025).

## Directory Structure

```
openscholar-api/
├── main.py                      # FastAPI application entry point
├── handler.py                   # RunPod serverless handler
├── requirements.txt             # Python dependencies
├── README.md                    # Main documentation
├── CODEBASE_STRUCTURE.md       # This file
│
├── openscholar_api/            # Core API module
│   ├── __init__.py
│   ├── models.py               # Pydantic models & data structures
│   ├── core_processing.py      # Retrieval & reranking logic
│   └── cache_manager.py        # Caching utilities
│
├── tests/                      # Test suite
│   ├── test_api_basic.py       # Basic API endpoint tests
│   ├── test_api_mock.py        # Mock-based unit tests
│   ├── test_retriever_api.py   # Retriever functionality tests
│   └── example_usage.py        # Usage examples & integration tests
│
├── archive/                    # Historical/deprecated files
│   ├── deprecated-docker-nodejs/  # Docker & Node.js artifacts
│   ├── debug_main.py          # Debugging version of main
│   ├── main_original.py       # Original implementation
│   ├── main_safe.py           # Safe fallback version
│   └── retriever_+_reranker.py # Standalone script version
│
├── .runpod/                    # RunPod configuration
│   ├── hub.json               # RunPod Hub metadata
│   └── tests.json             # RunPod test scenarios
│
├── .env.example                # Environment template
├── .env.runpod                 # RunPod-specific env template
├── .gitignore                  # Git exclusions
│
├── runpod-setup.sh            # RunPod deployment setup script
├── runpod-update.sh           # RunPod code update script
├── curl_tests.sh              # API testing script
└── start_retriever_api.py     # Alternative startup script

# Ignored/Generated directories (not in repo)
├── venv/                       # Python virtual environment
├── model_cache/               # Cached Hugging Face models
├── __pycache__/               # Python bytecode cache
└── logs/                      # Application logs
```

## Core Files

### Application Entry Points
- **`main.py`** - Primary FastAPI application with full API implementation
- **`handler.py`** - RunPod serverless wrapper for `main.py`
- **`start_retriever_api.py`** - Helper script with dependency checks

### API Module (`openscholar_api/`)
Organized package for core functionality:
- **`models.py`** - Data models (QueryRequest, QueryResponse, etc.)
- **`core_processing.py`** - Search, retrieval, and reranking algorithms
- **`cache_manager.py`** - Query result caching

### Tests (`tests/`)
Comprehensive test suite covering:
- Basic API functionality (`test_api_basic.py`)
- Unit tests with mocking (`test_api_mock.py`)
- Retriever-specific tests (`test_retriever_api.py`)
- Usage examples (`example_usage.py`)

**All test files are kept** - they provide valuable validation and documentation.

## Deployment Files

### RunPod Deployment
- `runpod-setup.sh` - Initial setup on RunPod server
- `runpod-update.sh` - Update deployed code
- `.runpod/` - RunPod configuration files (hub.json, tests.json)

### Docker Deployment (Deprecated)
Moved to `archive/deprecated-docker-nodejs/`:
- `Dockerfile` - GPU-optimized container definition
- `.dockerignore` - Docker build exclusions
- `ecosystem.config.js` - PM2/Node.js process manager config

## Environment Configuration

### Environment Files
- `.env.example` - Template for local development
- `.env.runpod` - Template for RunPod deployment
- `.env` - Actual environment (gitignored, must create from template)

### Required Variables
```bash
OPENSCHOLAR_API_KEY=your-secure-api-key-here
API_HOST=0.0.0.0
API_PORT=8002
```

## Archive Directory

Contains historical versions and deprecated files:
- **`deprecated-docker-nodejs/`** - Docker & Node.js files from pre-RunPod era
- **`debug_main.py`** - Debug version with extra logging
- **`main_original.py`** - Original implementation (before refactor)
- **`main_safe.py`** - Conservative fallback version
- **`retriever_+_reranker.py`** - Standalone script version

These are kept for reference but not used in production.

## Removed During Cleanup

### Completely Removed
- `node_modules/` (136 packages) - Node.js dependencies no longer needed
- Docker files moved to archive (project uses RunPod native deployment)

### Why Node.js Was Removed
The project is pure Python. Node.js dependencies were from early development experiments with PM2 process management, which is no longer used.

## Dependencies

### Python Packages (`requirements.txt`)
Key dependencies:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `torch` - PyTorch for ML models
- `transformers` - Hugging Face models
- `datasets` - Data loading
- `faiss-cpu` - Vector similarity search
- `runpod` - RunPod serverless SDK

## Development Workflow

### Local Development
```bash
# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API key

# Run
python main.py
# or
python start_retriever_api.py

# Test
curl http://localhost:8002/health
bash curl_tests.sh
```

### RunPod Deployment
```bash
# Initial setup (SSH into RunPod)
bash runpod-setup.sh

# Update code
bash runpod-update.sh
```

## Key Design Decisions

### 1. Modular Structure
Core logic separated into `openscholar_api/` package for:
- Better code organization
- Easier testing
- Future extensibility

### 2. Test Retention
All tests kept despite cleanup because they:
- Document expected behavior
- Prevent regressions
- Provide usage examples

### 3. Archive Strategy
Old files archived rather than deleted to:
- Maintain history
- Enable rollback if needed
- Document evolution

### 4. RunPod Focus
Docker support archived because:
- RunPod native deployment more reliable
- Simpler workflow
- Better resource management
- Faster startup

## API Endpoints

See [README.md](README.md) for detailed API documentation.

Key endpoints:
- `GET /health` - Health check
- `POST /search` - Scientific document search
- `GET /admin/stats` - Admin statistics (authenticated)

## Recent Changes

### November 10, 2025 - Query Limit Removal
- Removed 1000 character limit on query field
- Updated both `main.py` and `tests/test_api_mock.py`
- Queries now accept any length (minimum 1 character)

### November 10, 2025 - Codebase Cleanup
- Moved Docker files to archive
- Removed 136 Node.js packages
- Created CODEBASE_STRUCTURE.md documentation
- Updated .gitignore with Node.js exclusions

### August-September 2025 - RunPod Transition
- Transitioned from Docker to RunPod deployment
- Added RunPod serverless handler
- Improved connection reliability
- Streamlined deployment workflow

## Contributing

When adding new files:
1. Follow existing directory structure
2. Add tests to `tests/` directory
3. Update this document
4. Add configuration examples to `.env.example`

## License & Credits

See [README.md](README.md) for license and attribution information.
