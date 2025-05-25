#!/bin/bash

# OpenFold Production Deployment Script
# Author: Nik Jois <nikjois@llamasearch.ai>
# Version: 1.0.0

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="openfold"
VERSION="${VERSION:-1.0.0}"
ENVIRONMENT="${ENVIRONMENT:-production}"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check if Git is installed
    if ! command -v git &> /dev/null; then
        error "Git is not installed. Please install Git first."
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker first."
    fi
    
    log "Prerequisites check passed"
}

# Create environment file
create_env_file() {
    log "Creating environment file..."
    
    if [[ ! -f "$ENV_FILE" ]]; then
        cat > "$ENV_FILE" << EOF
# OpenFold Environment Configuration
# Generated on $(date)

# Application
VERSION=$VERSION
ENVIRONMENT=$ENVIRONMENT
DEBUG=false
LOG_LEVEL=INFO

# Database
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_DB=openfold
POSTGRES_USER=openfold

# Redis
REDIS_PASSWORD=$(openssl rand -base64 32)

# API Keys (set these manually)
OPENAI_API_KEY=your-openai-api-key-here
LANGCHAIN_API_KEY=your-langchain-api-key-here

# Monitoring
GRAFANA_PASSWORD=$(openssl rand -base64 16)

# Storage
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=$(openssl rand -base64 16)

# Jupyter
JUPYTER_TOKEN=$(openssl rand -base64 32)

# pgAdmin
PGADMIN_DEFAULT_EMAIL=admin@openfold.com
PGADMIN_DEFAULT_PASSWORD=$(openssl rand -base64 16)
EOF
        
        log "Environment file created. Please update API keys in $ENV_FILE"
    else
        warn "Environment file already exists"
    fi
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    directories=(
        "data"
        "models"
        "logs"
        "nginx/ssl"
        "nginx/logs"
        "database/init"
        "monitoring/grafana/provisioning"
        "monitoring/grafana/dashboards"
        "monitoring/logstash/pipeline"
        "redis"
        "notebooks"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log "Created directory: $dir"
    done
}

# Create Nginx configuration
create_nginx_config() {
    log "Creating Nginx configuration..."
    
    cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream openfold_api {
        server openfold-api:8000;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    server {
        listen 80;
        server_name localhost;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://openfold_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }
        
        # Health check
        location /health {
            proxy_pass http://openfold_api;
            proxy_set_header Host $host;
            access_log off;
        }
        
        # Documentation
        location /docs {
            proxy_pass http://openfold_api;
            proxy_set_header Host $host;
        }
        
        # Root
        location / {
            proxy_pass http://openfold_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
EOF
    
    log "Nginx configuration created"
}

# Create monitoring configurations
create_monitoring_configs() {
    log "Creating monitoring configurations..."
    
    # Prometheus configuration
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'openfold-api'
    static_configs:
      - targets: ['openfold-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
  
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
    metrics_path: '/metrics'
    scrape_interval: 30s
EOF
    
    # Logstash pipeline configuration
    cat > monitoring/logstash/pipeline/logstash.conf << 'EOF'
input {
  file {
    path => "/app/logs/*.log"
    start_position => "beginning"
    sincedb_path => "/dev/null"
  }
}

filter {
  if [path] =~ "access" {
    mutate { replace => { "type" => "nginx_access" } }
    grok {
      match => { "message" => "%{NGINXACCESS}" }
    }
  } else if [path] =~ "error" {
    mutate { replace => { "type" => "nginx_error" } }
  } else {
    mutate { replace => { "type" => "app_log" } }
  }
  
  date {
    match => [ "timestamp", "dd/MMM/yyyy:HH:mm:ss Z" ]
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "openfold-logs-%{+YYYY.MM.dd}"
  }
  stdout { codec => rubydebug }
}
EOF
    
    log "Monitoring configurations created"
}

# Build and deploy
deploy() {
    log "Starting deployment..."
    
    # Pull latest images
    log "Pulling latest base images..."
    docker-compose pull
    
    # Build application images
    log "Building application images..."
    docker-compose build --no-cache
    
    # Start services
    log "Starting services..."
    docker-compose up -d
    
    log "Deployment completed!"
}

# Health check
health_check() {
    log "Performing health checks..."
    
    # Wait for services to start
    sleep 30
    
    # Check API health
    if curl -f http://localhost:8000/health &> /dev/null; then
        log "API health check passed"
    else
        error "API health check failed"
    fi
    
    # Check database
    if docker-compose exec -T postgres pg_isready -U openfold -d openfold &> /dev/null; then
        log "Database health check passed"
    else
        error "Database health check failed"
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping &> /dev/null; then
        log "Redis health check passed"
    else
        error "Redis health check failed"
    fi
    
    log "All health checks passed"
}

# Show service URLs
show_urls() {
    log "Service URLs:"
    echo ""
    echo "OpenFold API:        http://localhost:8000"
    echo "API Documentation:   http://localhost:8000/docs"
    echo "Grafana Monitoring:  http://localhost:3000"
    echo "Prometheus:          http://localhost:9090"
    echo "Flower (Celery):     http://localhost:5555"
    echo "Kibana Logs:         http://localhost:5601"
    echo "MinIO Storage:       http://localhost:9001"
    echo "Jupyter Notebook:    http://localhost:8888"
    echo "pgAdmin:             http://localhost:5050"
    echo ""
    echo "Default credentials are in the .env file"
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    docker-compose down -v
    docker system prune -f
    log "Cleanup completed"
}

# Backup function
backup() {
    log "Creating backup..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    docker-compose exec -T postgres pg_dump -U openfold openfold > "$BACKUP_DIR/database.sql"
    
    # Backup volumes
    docker run --rm -v openfold_postgres_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/postgres_data.tar.gz -C /data .
    docker run --rm -v openfold_redis_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/redis_data.tar.gz -C /data .
    
    log "Backup completed: $BACKUP_DIR"
}

# Show logs
show_logs() {
    service="${1:-openfold-api}"
    docker-compose logs -f "$service"
}

# Main function
main() {
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            create_env_file
            create_directories
            create_nginx_config
            create_monitoring_configs
            deploy
            health_check
            show_urls
            ;;
        "start")
            docker-compose up -d
            health_check
            show_urls
            ;;
        "stop")
            docker-compose down
            ;;
        "restart")
            docker-compose restart
            health_check
            ;;
        "logs")
            show_logs "${2:-openfold-api}"
            ;;
        "health")
            health_check
            ;;
        "backup")
            backup
            ;;
        "cleanup")
            cleanup
            ;;
        "urls")
            show_urls
            ;;
        *)
            echo "Usage: $0 {deploy|start|stop|restart|logs|health|backup|cleanup|urls}"
            echo ""
            echo "Commands:"
            echo "  deploy   - Full deployment (default)"
            echo "  start    - Start services"
            echo "  stop     - Stop services"
            echo "  restart  - Restart services"
            echo "  logs     - Show logs (optionally specify service)"
            echo "  health   - Run health checks"
            echo "  backup   - Create backup"
            echo "  cleanup  - Clean up containers and volumes"
            echo "  urls     - Show service URLs"
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 