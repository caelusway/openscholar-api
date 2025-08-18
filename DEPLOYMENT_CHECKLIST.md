# 🚀 Production Deployment Checklist

## 🔐 Security & Environment

### ✅ Environment Configuration
- [x] `.env` file created with production settings
- [x] Secure API key generated and set in `OPENSCHOLAR_API_KEY`
- [x] `.env.example` file created for team reference
- [x] Production mode enabled: `PRODUCTION_MODE=true`
- [x] Debug logging disabled: `DEBUG_LOGGING=false`

### ✅ Git Security
- [x] `.gitignore` updated to exclude sensitive files
- [x] `.env` files excluded from version control
- [x] Model cache directory excluded
- [x] API keys and certificates excluded
- [x] Deployment configurations excluded

## 📦 Code Readiness

### ✅ API Security
- [x] API key authentication implemented
- [x] Protected routes configured
- [x] CORS properly configured
- [x] Error handling implemented
- [x] Admin endpoints secured

### ✅ Dependencies
- [x] `requirements.txt` updated with all dependencies
- [x] `python-dotenv` added for environment loading
- [x] `psutil` added for system monitoring
- [x] Version compatibility verified

## 🏗️ Deployment Preparation

### Before Deployment
- [ ] **Review `.env` file** - Ensure all sensitive values are set
- [ ] **Generate new API key** for production if needed
- [ ] **Test API endpoints** with production configuration
- [ ] **Verify model caching** works correctly
- [ ] **Check memory requirements** (8-16GB recommended)

### Environment Variables to Set
```bash
OPENSCHOLAR_API_KEY=<secure-api-key>
API_HOST=0.0.0.0
API_PORT=8002
PRODUCTION_MODE=true
DEBUG_LOGGING=false
```

### Production Server Requirements
- **RAM**: Minimum 8GB, Recommended 16GB
- **Storage**: 5GB for models and cache
- **Python**: 3.9+
- **OS**: Linux (Ubuntu 20.04+ recommended)

## 🔧 Deployment Steps

### 1. Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# Install system dependencies
sudo apt install build-essential -y
```

### 2. Application Setup
```bash
# Clone repository
git clone <your-repo-url>
cd openscholar-api

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your production values
nano .env
```

### 3. Service Configuration
```bash
# Create systemd service file
sudo nano /etc/systemd/system/openscholar-api.service
```

### 4. Start Service
```bash
# Enable and start service
sudo systemctl enable openscholar-api
sudo systemctl start openscholar-api

# Check status
sudo systemctl status openscholar-api
```

## 🧪 Testing

### Health Check
```bash
curl -X GET "http://your-server:8002/health"
```

### Authentication Test
```bash
curl -X POST "http://your-server:8002/search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query": "test", "final_topk": 1}'
```

## 🔍 Monitoring

### Logs
```bash
# View API logs
sudo journalctl -u openscholar-api -f

# Check system resources
htop
```

### Admin Endpoint
```bash
curl -X GET "http://your-server:8002/admin/stats" \
  -H "X-API-Key: your-api-key"
```

## 🛡️ Security Best Practices

- [x] API key is complex and unique
- [x] Environment variables used for all secrets
- [x] Debug mode disabled in production
- [x] CORS configured appropriately
- [x] No hardcoded secrets in code
- [x] `.env` file never committed to git

## 📋 Post-Deployment

- [ ] Monitor initial performance
- [ ] Test all API endpoints
- [ ] Verify model loading and caching
- [ ] Set up log rotation
- [ ] Configure monitoring/alerts
- [ ] Document API usage for team

## 🔄 Updates

When updating the application:
1. Stop the service
2. Pull latest changes
3. Update dependencies if needed
4. Restart the service
5. Test functionality

```bash
sudo systemctl stop openscholar-api
git pull origin main
pip install -r requirements.txt
sudo systemctl start openscholar-api
```