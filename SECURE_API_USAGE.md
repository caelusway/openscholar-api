# 🔐 OpenScholar Secure API Usage

## API Key Authentication

The OpenScholar API is now protected with API key authentication. All protected routes require a valid API key.

**Default API Key:** `openscholar-2024-secure-key` (fallback only)

⚠️ **IMPORTANT**: Set `OPENSCHOLAR_API_KEY` in your `.env` file!

## 🔓 Public Endpoints (No Authentication)

### Health Check
```bash
curl -X GET "http://localhost:8002/health"
```

## 🔐 Protected Endpoints (Require API Key)

### 1. Search Documents
```bash
# Using X-API-Key header
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: openscholar-2024-secure-key" \
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

### 2. Admin Statistics
```bash
curl -X GET "http://localhost:8002/admin/stats" \
  -H "X-API-Key: openscholar-2024-secure-key"
```

### 3. Admin Reload (Future)
```bash
curl -X POST "http://localhost:8002/admin/reload" \
  -H "X-API-Key: openscholar-2024-secure-key"
```

## 🚫 Authentication Errors

**Missing API Key:**
```json
{
  "detail": "Invalid or missing API key in X-API-Key header"
}
```

**Invalid API Key:**
```json
{
  "detail": "Invalid or missing API key in X-API-Key header"
}
```

## 🔧 Environment Configuration

### 1. Setup Environment File
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file with your secure values
vim .env
```

### 2. Configure API Key
```bash
# In .env file
OPENSCHOLAR_API_KEY=your-super-secure-api-key-here
API_HOST=0.0.0.0
API_PORT=8002
PRODUCTION_MODE=true
DEBUG_LOGGING=false
```

### 3. Generate Secure API Key
```bash
# Generate a secure random API key
python3 -c "import secrets; print(f'OPENSCHOLAR_API_KEY={secrets.token_urlsafe(32)}')"
```

### 4. Install Dependencies
```bash
# Install required packages including python-dotenv
pip install -r requirements.txt
```

## 📊 API Endpoints Summary

| Endpoint | Method | Protected | Description |
|----------|--------|-----------|-------------|
| `/health` | GET | ❌ | System health check |
| `/search` | POST | ✅ | Document search and ranking |
| `/admin/stats` | GET | ✅ | System statistics |
| `/admin/reload` | POST | ✅ | Reload models (future) |

## 🛡️ Security Features

- ✅ **API Key Authentication**: Header-based authentication
- ✅ **Route Protection**: Critical endpoints protected  
- ✅ **Error Handling**: Secure error responses
- ✅ **CORS Support**: Configurable cross-origin requests
- ✅ **Admin Endpoints**: Protected administrative functions

## 🧪 Testing Authentication

**Test without API key (should fail):**
```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

**Test with wrong API key (should fail):**
```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: wrong-key" \
  -d '{"query": "test"}'
```

**Test with correct API key (should work):**
```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: openscholar-2024-secure-key" \
  -d '{"query": "mitochondrial function", "final_topk": 2}'
```