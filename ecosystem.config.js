module.exports = {
  apps: [{
    name: 'openscholar-api',
    script: 'python',
    args: 'main.py',
    cwd: '/workspace/openscholar-api',
    interpreter: 'none',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '2G',
    env: {
      NODE_ENV: 'production',
      API_HOST: '0.0.0.0',
      API_PORT: '8002'
    },
    log_file: './logs/combined.log',
    out_file: './logs/out.log',
    error_file: './logs/error.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    kill_timeout: 5000,
    restart_delay: 4000
  }]
};