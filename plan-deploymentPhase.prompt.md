# 🚀 MOSAIC Phase 8: Full-Stack Deployment & Productionization
## Professional React + FastAPI + Docker Architecture

---

## PHASE 8: FULL-STACK DEPLOYMENT

### 8.1 Backend Microservice (FastAPI)
- [ ] 🟢 Initialize `src/serving/` directory
- [ ] 🟢 Install deployment dependencies
  - [ ] Run: `pip install fastapi uvicorn[standard] python-multipart pydantic`
- [ ] 🟡 Implement **Model Inference Service** (`src/serving/model_service.py`)
  - [ ] Load best trained checkpoint (MOSAIC model)
  - [ ] Implement `predict_survival(patient_data)` function
  - [ ] Implement `get_attention_maps(patient_data)` function (returns heatmaps)
- [ ] 🔴 Implement **WSI Tile Server** (`src/serving/tile_server.py`)
  - [ ] Use `openslide` to read SVS files dynamically
  - [ ] Create endpoint: `GET /wsi/{slide_id}/deepzoom/{level}/{col}_{row}.jpg`
  - [ ] Implement caching for frequently accessed tiles
- [ ] 🟢 Create **API Endpoints** (`src/serving/main.py`)
  - [ ] `POST /api/predict`: Receives JSON + File upload, returns risk score & survival curve
  - [ ] `GET /api/health`: Health check
- [ ] 🟡 Implement **Input Validation** (Pydantic models)
  - [ ] Schema for Clinical Data (age, stage, etc.)
  - [ ] Schema for Omics Data (gene expression vector)

### 8.2 Frontend Dashboard (React + TypeScript)
- [ ] 🟢 Initialize Frontend Project
  - [ ] Run: `npm create vite@latest mosaic-dashboard -- --template react-ts`
  - [ ] Install UI Libs: `npm install @mui/material @emotion/react @emotion/styled recharts axios`
- [ ] 🔴 Implement **WSI Viewer Component** (`src/components/WSIViewer.tsx`)
  - [ ] Install: `npm install openseadragon`
  - [ ] Initialize OpenSeadragon viewer
  - [ ] Connect to backend Tile Server URL
  - [ ] **Challenge**: Overlay Attention Heatmap on top of OSD viewer (requires coordinate mapping)
- [ ] 🟡 Implement **Survival Visualization** (`src/components/SurvivalCurve.tsx`)
  - [ ] Use `recharts` to plot Time vs. Survival Probability
- [ ] 🟡 Implement **Explainability Dashboard**
  - [ ] Bar chart for Top-20 Gene Importance
  - [ ] Radar chart for Modality Contributions (Clinical vs. WSI vs. Omics)
- [ ] 🟢 Create **Patient Upload Form**
  - [ ] File dropper for SVS files
  - [ ] Form fields for Clinical variables

### 8.3 Containerization (Docker)
- [ ] 🔴 Create `backend.Dockerfile`
  - [ ] Base: `python:3.10-slim`
  - [ ] **Critical**: Install system dependencies (`libopenslide0`, `build-essential`)
  - [ ] Copy `src/` and installed model checkpoints
  - [ ] Entrypoint: `uvicorn src.serving.main:app --host 0.0.0.0 --port 8000`
- [ ] 🟢 Create `frontend.Dockerfile`
  - [ ] Build stage: Node.js (build React app)
  - [ ] Serve stage: Nginx (serve static files)
- [ ] 🟡 Create `docker-compose.yml`
  - [ ] Define `backend` service (GPU enabled)
  - [ ] Define `frontend` service (ports 80:80)
  - [ ] Set up internal networking

### 8.4 CI/CD Pipeline (GitHub Actions)
- [ ] 🟡 Create `.github/workflows/docker-build.yml`
  - [ ] Trigger on push to `main`
  - [ ] Build Docker images
  - [ ] Run Unit Tests (PyTest)
  - [ ] Push images to Docker Hub or GitHub Container Registry (GHCR)

### 8.5 Documentation & Handoff
- [ ] 🟢 Create `DEPLOYMENT.md`
  - [ ] Instructions to run `docker-compose up`
  - [ ] API Documentation (Screenshot of Swagger UI)
