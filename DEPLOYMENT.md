# MOSAIC Deployment Guide

This document provides comprehensive instructions for deploying the MOSAIC (Multi-Omics Survival Analysis with Interpretable Cross-modal attention) application.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Production Considerations](#production-considerations)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32+ GB |
| GPU | - | NVIDIA GPU with 8+ GB VRAM |
| Storage | 50 GB | 200+ GB (for WSI files) |

### Software Requirements

- **Docker** >= 24.0
- **Docker Compose** >= 2.20
- **NVIDIA Container Toolkit** (for GPU support)

### Installation

#### Docker & Docker Compose

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Verify installation
docker --version
docker compose version
```

#### NVIDIA Container Toolkit (Optional, for GPU)

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/Survival-Prediction.git
cd Survival-Prediction
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### 3. Start Services

```bash
# Build and start all services
docker compose up --build -d

# View logs
docker compose logs -f

# Check service status
docker compose ps
```

### 4. Access the Application

| Service | URL | Description |
|---------|-----|-------------|
| **Dashboard** | http://localhost:80 | React frontend |
| **API** | http://localhost:80/api | FastAPI backend |
| **Swagger UI** | http://localhost:80/api/docs | Interactive API docs |
| **ReDoc** | http://localhost:80/api/redoc | Alternative API docs |

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# =============================================================================
# MOSAIC Configuration
# =============================================================================

# Model Settings
MODEL_CHECKPOINT_PATH=/app/checkpoints/mosaic_best.ckpt
DEVICE=cuda  # or 'cpu' for CPU-only deployment

# WSI Settings
WSI_CACHE_DIR=/app/data/wsi_cache
WSI_CACHE_SIZE=100
WSI_TILE_SIZE=256
WSI_TILE_OVERLAP=0

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_LOG_LEVEL=info

# Security
API_KEY=your-secure-api-key-here
CORS_ORIGINS=http://localhost,http://localhost:80

# Database (optional, for patient records)
DATABASE_URL=postgresql://user:password@db:5432/mosaic

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

### Model Checkpoint

Place your trained model checkpoint in the `checkpoints/` directory:

```bash
mkdir -p checkpoints
cp /path/to/your/mosaic_best.ckpt checkpoints/
```

### WSI Data

Mount your whole-slide images directory:

```bash
# Create data directory
mkdir -p data/wsi

# Copy or symlink WSI files
ln -s /path/to/wsi/files/* data/wsi/
```

---

## Docker Deployment

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        nginx (Port 80)                       │
│                     Reverse Proxy + Static                   │
├─────────────────────────────────────────────────────────────┤
│                              │                               │
│    ┌─────────────────────────┴─────────────────────────┐    │
│    │                                                     │    │
│    ▼                                                     ▼    │
│  /api/*                                              /*       │
│    │                                                     │    │
│    ▼                                                     ▼    │
│ ┌─────────────────────┐                 ┌──────────────────┐ │
│ │   Backend (8000)    │                 │  Frontend (nginx) │ │
│ │   FastAPI + Torch   │                 │   React + Vite    │ │
│ │   + OpenSlide       │                 │                    │ │
│ └─────────────────────┘                 └──────────────────┘ │
│          │                                                    │
│          ▼                                                    │
│ ┌─────────────────────┐                                      │
│ │  GPU (optional)     │                                      │
│ │  NVIDIA Runtime     │                                      │
│ └─────────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
```

### Docker Compose Commands

```bash
# Build images
docker compose build

# Start services (detached)
docker compose up -d

# Start with build
docker compose up --build -d

# View logs
docker compose logs -f
docker compose logs -f backend   # Backend only
docker compose logs -f frontend  # Frontend only

# Stop services
docker compose down

# Stop and remove volumes
docker compose down -v

# Restart a specific service
docker compose restart backend

# Scale backend (for load balancing)
docker compose up -d --scale backend=3

# Check resource usage
docker compose stats
```

### CPU-Only Deployment

For systems without NVIDIA GPU, modify `docker-compose.yml`:

```yaml
services:
  backend:
    # Remove or comment out GPU section
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
    environment:
      - DEVICE=cpu
```

---

## API Documentation

### Base URL

```
http://localhost:80/api
```

### Authentication

Include API key in request headers:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:80/api/health
```

### Endpoints

#### Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "version": "1.0.0"
}
```

#### Survival Prediction

```http
POST /api/predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "patient_id": "TCGA-A1-A0SK",
  "clinical": {
    "age": 65.0,
    "gender": "female",
    "stage": "II",
    "grade": 2,
    "tumor_size": 3.5,
    "lymph_nodes_positive": 2
  },
  "omics": {
    "transcriptomics": [0.5, -1.2, 0.8, ...],
    "methylation": [0.1, 0.9, 0.3, ...],
    "mutations": ["TP53", "BRCA1", "PIK3CA"]
  },
  "wsi": {
    "slide_id": "TCGA-A1-A0SK-01Z-00-DX1",
    "features": [0.1, 0.2, 0.3, ...]
  }
}
```

**Response:**
```json
{
  "patient_id": "TCGA-A1-A0SK",
  "risk_score": 0.73,
  "survival_probability": {
    "12_months": 0.85,
    "24_months": 0.72,
    "36_months": 0.61,
    "60_months": 0.48
  },
  "risk_group": "high",
  "confidence_interval": {
    "lower": 0.65,
    "upper": 0.81
  },
  "attention_weights": {
    "clinical": 0.25,
    "transcriptomics": 0.30,
    "methylation": 0.15,
    "mutations": 0.10,
    "pathology": 0.20
  }
}
```

#### WSI Tile Server

```http
GET /wsi/{slide_id}/info
GET /wsi/{slide_id}/tile/{level}/{col}/{row}
GET /wsi/{slide_id}/dzi
```

**Example:**
```bash
# Get slide info
curl http://localhost:80/wsi/TCGA-A1-A0SK-01Z/info

# Get tile at level 0, column 5, row 3
curl http://localhost:80/wsi/TCGA-A1-A0SK-01Z/tile/0/5/3 --output tile.jpeg

# Get Deep Zoom Image descriptor
curl http://localhost:80/wsi/TCGA-A1-A0SK-01Z/dzi
```

### Swagger UI

Interactive API documentation is available at:

```
http://localhost:80/api/docs
```

**Features:**
- Try out endpoints directly in the browser
- View request/response schemas
- Download OpenAPI specification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🔬 MOSAIC API                                                    Swagger UI │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Health                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ GET  /api/health        Check API and model health status             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Prediction                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ POST /api/predict       Generate survival prediction for patient      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  WSI Tiles                                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ GET  /wsi/{slide_id}/info     Get slide metadata                      │ │
│  │ GET  /wsi/{slide_id}/tile/... Retrieve image tile                     │ │
│  │ GET  /wsi/{slide_id}/dzi      Deep Zoom Image descriptor              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Schemas                                                                     │
│  ├── ClinicalData                                                           │
│  ├── OmicsData                                                              │
│  ├── WSIData                                                                │
│  ├── PredictionRequest                                                      │
│  ├── PredictionResponse                                                     │
│  └── HealthResponse                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:80/api"
API_KEY = "your-api-key"

headers = {"X-API-Key": API_KEY}

# Health check
response = requests.get(f"{BASE_URL}/health", headers=headers)
print(response.json())

# Prediction
payload = {
    "patient_id": "TCGA-A1-A0SK",
    "clinical": {
        "age": 65.0,
        "gender": "female",
        "stage": "II",
        "grade": 2
    },
    "omics": {
        "transcriptomics": [0.5] * 1000,
        "methylation": [0.3] * 500
    }
}

response = requests.post(
    f"{BASE_URL}/predict",
    json=payload,
    headers=headers
)
result = response.json()
print(f"Risk Score: {result['risk_score']}")
print(f"5-Year Survival: {result['survival_probability']['60_months']:.1%}")
```

---

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker compose logs backend

# Common fixes:
# 1. Port already in use
sudo lsof -i :80
sudo kill -9 <PID>

# 2. GPU not available
nvidia-smi  # Check GPU status
docker compose down
# Edit docker-compose.yml to use CPU mode
docker compose up -d
```

#### Model Not Loading

```bash
# Verify checkpoint exists
ls -la checkpoints/

# Check model path in .env
cat .env | grep MODEL_CHECKPOINT

# View backend logs
docker compose logs backend | grep -i model
```

#### WSI Tiles Not Loading

```bash
# Verify WSI files are mounted
docker compose exec backend ls -la /app/data/wsi/

# Check OpenSlide installation
docker compose exec backend python -c "import openslide; print(openslide.__version__)"

# Test specific slide
docker compose exec backend python -c "
import openslide
slide = openslide.OpenSlide('/app/data/wsi/your-slide.svs')
print(slide.dimensions)
"
```

#### Out of Memory (GPU)

```bash
# Check GPU memory
nvidia-smi

# Reduce batch size in config
# Edit configs/train_config.yaml
batch_size: 4  # Reduce from 16

# Or switch to CPU
docker compose down
# Set DEVICE=cpu in .env
docker compose up -d
```

### Health Check Script

```bash
#!/bin/bash
# health_check.sh

echo "=== MOSAIC Health Check ==="

# Check containers
echo -e "\n[Containers]"
docker compose ps

# Check API health
echo -e "\n[API Health]"
curl -s http://localhost:80/api/health | jq .

# Check GPU (if available)
echo -e "\n[GPU Status]"
docker compose exec backend python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "GPU check failed"

# Check disk space
echo -e "\n[Disk Space]"
df -h | grep -E "^/dev|Filesystem"

# Check memory
echo -e "\n[Memory]"
free -h

echo -e "\n=== Health Check Complete ==="
```

---

## Production Considerations

### Security

1. **API Key Authentication**
   ```bash
   # Generate secure API key
   openssl rand -hex 32
   ```

2. **HTTPS with Let's Encrypt**
   ```yaml
   # Add to docker-compose.yml
   services:
     certbot:
       image: certbot/certbot
       volumes:
         - ./certbot/conf:/etc/letsencrypt
         - ./certbot/www:/var/www/certbot
   ```

3. **Network Isolation**
   ```yaml
   networks:
     frontend:
       driver: bridge
     backend:
       driver: bridge
       internal: true  # No external access
   ```

### Monitoring

1. **Prometheus Metrics**
   ```yaml
   services:
     prometheus:
       image: prom/prometheus
       ports:
         - "9090:9090"
       volumes:
         - ./prometheus.yml:/etc/prometheus/prometheus.yml
   ```

2. **Grafana Dashboard**
   ```yaml
   services:
     grafana:
       image: grafana/grafana
       ports:
         - "3000:3000"
       environment:
         - GF_SECURITY_ADMIN_PASSWORD=admin
   ```

### Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/mosaic/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup checkpoints
cp -r checkpoints/ $BACKUP_DIR/

# Backup configuration
cp .env docker-compose.yml $BACKUP_DIR/

# Backup database (if using)
docker compose exec -T db pg_dump -U postgres mosaic > $BACKUP_DIR/database.sql

echo "Backup completed: $BACKUP_DIR"
```

### Scaling

For high availability:

```yaml
# docker-compose.prod.yml
services:
  backend:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  nginx:
    depends_on:
      - backend
    # nginx will load balance across backend replicas
```

---

## Support

- **Documentation**: [docs/](./docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/Survival-Prediction/issues)
- **Email**: mosaic-support@your-org.com

---

*Last updated: February 2026*
