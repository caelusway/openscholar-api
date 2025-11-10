#!/bin/bash

# OpenScholar API - Quick Update Script for RunPod
# Updates the code from GitHub and restarts the service

set -e

APP_DIR="/workspace/open-scholar-inference"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_status "🔄 Updating OpenScholar API..."

# Check if directory exists
if [ ! -d "$APP_DIR" ]; then
    print_warning "App directory not found. Run runpod-setup.sh first!"
    exit 1
fi

# Pull latest changes
print_status "Pulling latest changes from GitHub..."
cd "$APP_DIR"
git pull origin main

# Update dependencies if requirements changed
if git diff HEAD~1 --name-only | grep -q requirements.txt; then
    print_status "Requirements changed, updating dependencies..."
    pip3 install -r requirements.txt
fi

# Restart the application
print_status "Restarting OpenScholar API..."
/workspace/start-openscholar.sh restart

# Wait and check status
sleep 3
if /workspace/start-openscholar.sh status | grep -q "running"; then
    print_status "✅ Update completed successfully!"
    print_status "🌐 API is running at: http://localhost:8002"
else
    print_warning "❌ Application failed to start. Check logs: tail -f /workspace/logs/app.log"
    exit 1
fi