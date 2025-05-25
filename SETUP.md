# OpenFold Setup and Deployment Guide

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Development Setup](#development-setup)
- [Production Deployment](#production-deployment)
- [Configuration](#configuration)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Troubleshooting](#troubleshooting)
- [API Usage](#api-usage)

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+ with WSL2
- **Memory**: Minimum 8GB RAM (16GB+ recommended for production)
- **Storage**: Minimum 50GB free space (100GB+ recommended)
- **CPU**: Multi-core processor (GPU recommended for ML inference)

### Software Dependencies
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **Git**: Version 2.20+
- **OpenSSL**: For generating secure passwords
- **curl**: For health checks and API testing

### Installation Commands

#### Ubuntu/Debian
```bash
# Update package index
sudo apt update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install other dependencies
sudo apt install -y git curl openssl
```

#### macOS
```bash
# Install Docker Desktop from https://docker.com/products/docker-desktop
# Or using Homebrew:
brew install --cask docker
brew install git curl openssl
```

#### Windows (WSL2)
```bash
# Install Docker Desktop with WSL2 backend
# Follow instructions at https://docs.docker.com/desktop/windows/wsl/

# In WSL2 terminal:
sudo apt update
sudo apt install -y git curl openssl
```

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/llamasearchai/OpenFold.git
cd OpenFold
```

### 2. Deploy with One Command
```bash
./deploy.sh
```

This will:
- Check prerequisites
- Create environment configuration
- Set up necessary directories
- Configure services
- Deploy the complete stack
- Run health checks
- Display service URLs

### 3. Access the Platform
After deployment, access these services:

- **OpenFold API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Monitoring Dashboard**: http://localhost:3000
- **Jupyter Notebooks**: http://localhost:8888

## Development Setup

### 1. Development Environment
```bash
# Clone repository
git clone https://github.com/llamasearchai/OpenFold.git
cd OpenFold

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp config.env .env
# Edit .env with your configuration
```

### 2. Run Development Server
```bash
# Start database and Redis
docker-compose up -d postgres redis

# Run API server with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=api --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## Production Deployment

### 1. Server Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose (see prerequisites)

# Create deployment user
sudo useradd -m -s /bin/bash openfold
sudo usermod -aG docker openfold
sudo su - openfold
```

### 2. Clone and Configure
```bash
# Clone repository
git clone https://github.com/llamasearchai/OpenFold.git
cd OpenFold

# Run deployment script
./deploy.sh
```

### 3. Configure Environment Variables
Edit the generated `.env` file:

```bash
nano .env
```

**Required configurations:**
```env
# OpenAI API Key (required for AI features)
OPENAI_API_KEY=your_actual_openai_api_key

# Database passwords (auto-generated, keep secure)
POSTGRES_PASSWORD=your_secure_password

# Monitoring credentials
GRAFANA_PASSWORD=your_grafana_password

# Cloud storage (optional)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

### 4. SSL/TLS Configuration (Production)
```bash
# Generate SSL certificates (Let's Encrypt recommended)
sudo apt install certbot

# Get certificates
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates to nginx directory
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/

# Update nginx configuration for HTTPS
# Edit nginx/nginx.conf to add SSL configuration
```

### 5. Firewall Configuration
```bash
# Configure UFW firewall
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key for AI features | - | Yes |
| `POSTGRES_PASSWORD` | Database password | Auto-generated | Yes |
| `REDIS_PASSWORD` | Redis password | Auto-generated | Yes |
| `ENVIRONMENT` | Deployment environment | production | No |
| `LOG_LEVEL` | Logging level | INFO | No |
| `AWS_ACCESS_KEY_ID` | AWS access key | - | No |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | - | No |

### Model Configuration
Place model files in the `models/` directory:

```
models/
├── alphafold3/
│   ├── model.pkl
│   └── config.json
├── esm2/
│   ├── pytorch_model.bin
│   └── config.json
└── openfold/
    ├── model.pt
    └── config.yaml
```

### Database Initialization
Custom database initialization scripts can be placed in `database/init/`:

```sql
-- database/init/01_create_tables.sql
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sequence TEXT NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Monitoring and Maintenance

### Service Management
```bash
# Check service status
./deploy.sh health

# View logs
./deploy.sh logs                    # API logs
./deploy.sh logs postgres          # Database logs
./deploy.sh logs redis             # Redis logs

# Restart services
./deploy.sh restart

# Stop services
./deploy.sh stop

# Start services
./deploy.sh start
```

### Monitoring Dashboards

#### Grafana (http://localhost:3000)
- **Default credentials**: admin / (check .env file)
- **Pre-configured dashboards**:
  - API Performance
  - Database Metrics
  - System Resources
  - Error Rates

#### Prometheus (http://localhost:9090)
- **Metrics collection** from all services
- **Custom queries** for specific metrics
- **Alerting rules** for critical issues

#### Kibana (http://localhost:5601)
- **Log aggregation** from all services
- **Search and filter** logs
- **Create visualizations** and dashboards

### Backup and Recovery
```bash
# Create backup
./deploy.sh backup

# Restore from backup
docker-compose exec -T postgres psql -U openfold -d openfold < backups/20231201_120000/database.sql
```

### Performance Tuning

#### Database Optimization
```sql
-- Optimize PostgreSQL settings
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
SELECT pg_reload_conf();
```

#### Redis Optimization
```bash
# Edit redis/redis.conf
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using the port
sudo lsof -i :8000

# Kill the process
sudo kill -9 <PID>

# Or change port in docker-compose.yml
```

#### 2. Docker Permission Denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or run:
newgrp docker
```

#### 3. Out of Memory
```bash
# Check memory usage
docker stats

# Increase swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. SSL Certificate Issues
```bash
# Renew Let's Encrypt certificates
sudo certbot renew

# Test certificate renewal
sudo certbot renew --dry-run
```

### Log Analysis
```bash
# Check API logs for errors
docker-compose logs openfold-api | grep ERROR

# Check database connections
docker-compose logs postgres | grep "connection"

# Monitor real-time logs
docker-compose logs -f --tail=100
```

### Health Checks
```bash
# Manual health check
curl -f http://localhost:8000/health

# Database health
docker-compose exec postgres pg_isready -U openfold

# Redis health
docker-compose exec redis redis-cli ping
```

## API Usage

### Authentication
```bash
# Get API key (if authentication is enabled)
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'
```

### Submit Prediction
```bash
curl -X POST http://localhost:8000/api/prediction/submit \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "model_type": "alphafold3",
    "confidence_threshold": 0.7
  }'
```

### Check Job Status
```bash
curl http://localhost:8000/api/prediction/status/{job_id}
```

### Get Results
```bash
curl http://localhost:8000/api/prediction/result/{job_id}
```

### AI Analysis
```bash
curl -X POST http://localhost:8000/api/agents/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "structure_data": "PDB_CONTENT_HERE",
    "analysis_type": "comprehensive"
  }'
```

## Support and Contributing

### Getting Help
- **Documentation**: http://localhost:8000/docs
- **Issues**: https://github.com/llamasearchai/OpenFold/issues
- **Discussions**: https://github.com/llamasearchai/OpenFold/discussions
- **Email**: nikjois@llamasearch.ai

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guide
- Add type hints
- Write comprehensive tests
- Update documentation
- Use conventional commits

---

**Built by the LlamaSearch AI team** 