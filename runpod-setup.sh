#!/bin/bash

# OpenScholar API - RunPod One-Click Setup Script
# This script clones the repo, installs dependencies, and runs the application

set -e

echo "🚀 Starting OpenScholar API Setup on RunPod..."

# Configuration
REPO_URL="https://github.com/bio-xyz/open-scholar-inference.git"
APP_DIR="/workspace/open-scholar-inference"
LOG_FILE="/workspace/logs/app.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Stop existing processes
print_status "Stopping existing OpenScholar processes..."
pkill -f "python.*main.py" || print_warning "No existing processes found"

# Create directories
print_status "Creating necessary directories..."
mkdir -p /workspace/logs
mkdir -p /workspace/model_cache

# Backup existing installation
if [ -d "$APP_DIR" ]; then
    print_status "Backing up existing installation..."
    rm -rf /workspace/open-scholar-inference-backup
    mv "$APP_DIR" /workspace/open-scholar-inference-backup
fi

# Clone the repository
print_status "Cloning OpenScholar API repository..."
cd /workspace
git clone "$REPO_URL" open-scholar-inference
cd "$APP_DIR"

# Install Python dependencies
print_status "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Set environment variables
print_status "Setting up environment variables..."
export API_HOST=0.0.0.0
export API_PORT=8002
export PRODUCTION_MODE=true
export DEBUG_LOGGING=false

# Check if API key is provided
if [ -z "$OPENSCHOLAR_API_KEY" ]; then
    print_warning "OPENSCHOLAR_API_KEY environment variable not set!"
    print_warning "Please set it before running: export OPENSCHOLAR_API_KEY='your-key-here'"
    print_warning "Or pass it when running this script: OPENSCHOLAR_API_KEY='your-key' $0"
fi

# Create startup script for the application
print_status "Creating startup script..."
cat > /workspace/start-openscholar.sh << 'EOF'
#!/bin/bash

APP_DIR="/workspace/open-scholar-inference"
LOG_FILE="/workspace/logs/app.log"
PID_FILE="/workspace/open-scholar-inference.pid"

# Function to start the application
start_app() {
    cd "$APP_DIR"
    export API_HOST=0.0.0.0
    export API_PORT=8002
    export PRODUCTION_MODE=true
    export DEBUG_LOGGING=false
    
    echo "Starting OpenScholar API..." >> "$LOG_FILE"
    nohup python3 main.py >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Started with PID: $(cat $PID_FILE)"
}

# Function to stop the application
stop_app() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping OpenScholar API (PID: $PID)..."
            kill "$PID"
            rm -f "$PID_FILE"
        else
            echo "Process not running, removing stale PID file"
            rm -f "$PID_FILE"
        fi
    fi
    pkill -f "python3 main.py" || true
}

# Function to check if app is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Main logic
case "$1" in
    start)
        if is_running; then
            echo "OpenScholar API is already running"
        else
            start_app
        fi
        ;;
    stop)
        stop_app
        ;;
    restart)
        stop_app
        sleep 2
        start_app
        ;;
    status)
        if is_running; then
            echo "OpenScholar API is running (PID: $(cat $PID_FILE))"
        else
            echo "OpenScholar API is not running"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
EOF

chmod +x /workspace/start-openscholar.sh

# Start the application
print_status "Starting OpenScholar API..."
/workspace/start-openscholar.sh stop  # Stop any existing process
sleep 2
OPENSCHOLAR_API_KEY="$OPENSCHOLAR_API_KEY" /workspace/start-openscholar.sh start

# Wait a moment for startup
sleep 5

# Check if it's running
if /workspace/start-openscholar.sh status | grep -q "running"; then
    print_status "✅ OpenScholar API is running successfully!"
    print_status "📝 Logs: tail -f $LOG_FILE"
    print_status "🌐 API should be available at: http://localhost:8002"
    print_status "❤️  Health check: curl http://localhost:8002/health"
else
    print_error "❌ Failed to start OpenScholar API"
    print_error "Check logs: tail -f $LOG_FILE"
    exit 1
fi

# Show useful commands
echo
print_status "📚 Useful commands:"
echo "  • Check status:     /workspace/start-openscholar.sh status"
echo "  • View logs:        tail -f $LOG_FILE"
echo "  • Restart app:      /workspace/start-openscholar.sh restart"
echo "  • Stop app:         /workspace/start-openscholar.sh stop"
echo "  • Update code:      cd $APP_DIR && git pull && /workspace/start-openscholar.sh restart"

echo
print_status "🎉 Setup complete! OpenScholar API is now running and will auto-restart on failures."