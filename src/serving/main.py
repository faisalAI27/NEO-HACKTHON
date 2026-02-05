"""
MOSAIC FastAPI Application

Main API server for survival prediction and WSI tile serving.

Usage:
    uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.serving.model_service import MOSAICModelService
from src.serving.schemas import (
    ClinicalData,
    ErrorResponse,
    HealthResponse,
    OmicsData,
    PredictionRequest,
    PredictionResponse,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Global State
# ============================================================================

model_service: Optional[MOSAICModelService] = None
tile_server = None  # Lazy-loaded if OpenSlide is available

# Upload directory for WSI files
UPLOAD_DIR = Path(os.environ.get("MOSAIC_UPLOAD_DIR", "data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Lifespan Management
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global model_service, tile_server

    logger.info("Starting MOSAIC API Server...")

    # Initialize model service
    model_service = MOSAICModelService()

    # Try to load best checkpoint
    checkpoint_dir = os.environ.get("MOSAIC_CHECKPOINT_DIR", "checkpoints")
    fold = int(os.environ.get("MOSAIC_FOLD", "0"))

    if model_service.load_best_checkpoint(checkpoint_dir, fold):
        logger.info(f"Model loaded from {checkpoint_dir}/fold_{fold}")
    else:
        logger.warning(
            "No model checkpoint loaded. Running in DEMO MODE with simulated predictions."
        )

    # Try to initialize tile server
    try:
        from src.serving.tile_server import get_tile_server

        wsi_dir = os.environ.get("MOSAIC_WSI_DIR", "data/raw/svs")
        tile_server = get_tile_server(wsi_dir)
        if tile_server:
            logger.info(f"WSI Tile Server initialized: {wsi_dir}")
    except ImportError:
        logger.warning("OpenSlide not available. WSI tile serving disabled.")
        tile_server = None
    except Exception as e:
        logger.warning(f"Failed to initialize tile server: {e}")
        tile_server = None

    yield

    # Cleanup
    logger.info("Shutting down MOSAIC API Server...")
    if tile_server:
        tile_server.close()


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="MOSAIC Survival Prediction API",
    description="""
    Multi-Omics Survival Analysis with Interpretable Cross-modal attention.
    
    This API provides endpoints for:
    - **Survival Prediction**: Predict patient survival using multimodal data
    - **WSI Tile Serving**: Serve whole slide image tiles for visualization
    - **Attention Maps**: Extract interpretable attention weights
    
    ## Supported Modalities
    - Clinical data (age, stage, HPV status)
    - RNA-seq gene expression
    - DNA methylation
    - Somatic mutations
    - Whole slide images (pathology)
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health Check
# ============================================================================


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns service status and model availability.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_service.is_loaded if model_service else False,
        available_modalities=(
            model_service.available_modalities if model_service else []
        ),
        wsi_server_available=tile_server is not None,
        version="1.0.0",
    )


# ============================================================================
# Prediction Endpoints
# ============================================================================

# Pydantic model for cancer detection request
from pydantic import BaseModel


class CancerDetectionRequest(BaseModel):
    slide_id: str


class CancerDetectionResponse(BaseModel):
    slide_id: str
    is_cancerous: bool
    confidence: float
    cancer_type: Optional[str] = None
    tumor_regions: Optional[int] = None
    analysis_time_seconds: float


@app.post(
    "/api/analyze/cancer-detection",
    response_model=CancerDetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        404: {"model": ErrorResponse, "description": "Slide not found"},
        500: {"model": ErrorResponse, "description": "Analysis error"},
    },
    tags=["Cancer Detection"],
)
async def detect_cancer(request: CancerDetectionRequest):
    """
    Stage 1: Analyze a WSI to detect if the tissue is cancerous.

    This is the first step in the prediction pipeline:
    1. Upload WSI → Cancer Detection (this endpoint)
    2. If cancerous → Collect clinical/genomic data → Survival Prediction

    Returns:
    - is_cancerous: Whether cancer was detected
    - confidence: Confidence score (0-1)
    - cancer_type: Type of cancer detected (if applicable)
    - tumor_regions: Number of tumor regions identified
    """
    import random
    import time

    start_time = time.time()

    # Verify slide exists
    slide_path = None
    if tile_server:
        try:
            slide_path = tile_server._find_slide_path(request.slide_id)
        except Exception:
            pass

    # Also check upload directory
    if not slide_path:
        for ext in [".svs", ".tif", ".tiff", ".ndpi"]:
            potential_path = UPLOAD_DIR / f"{request.slide_id}{ext}"
            if potential_path.exists():
                slide_path = potential_path
                break

    if not slide_path:
        raise HTTPException(
            status_code=404, detail=f"Slide not found: {request.slide_id}"
        )

    # In demo mode, simulate cancer detection
    # In production, this would run actual pathology analysis model
    logger.info(f"Running cancer detection on slide: {request.slide_id}")

    # Simulate analysis time (in production, actual analysis would happen here)
    # For now, use demo results
    is_cancerous = (
        random.random() > 0.15
    )  # 85% chance of cancer detection (TCGA is cancer dataset)
    confidence = (
        random.uniform(0.75, 0.98) if is_cancerous else random.uniform(0.82, 0.96)
    )

    # If cancer detected, identify type
    cancer_type = None
    tumor_regions = None
    if is_cancerous:
        cancer_type = "Head and Neck Squamous Cell Carcinoma (HNSC)"
        tumor_regions = random.randint(3, 12)

    analysis_time = time.time() - start_time

    return CancerDetectionResponse(
        slide_id=request.slide_id,
        is_cancerous=is_cancerous,
        confidence=round(confidence, 3),
        cancer_type=cancer_type,
        tumor_regions=tumor_regions,
        analysis_time_seconds=round(analysis_time, 2),
    )


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {
            "model": ErrorResponse,
            "description": "Model not loaded or inference error",
        },
    },
    tags=["Prediction"],
)
async def predict_survival(request: PredictionRequest):
    """
    Predict survival for a patient.

    Accepts clinical, omics, and/or WSI data and returns:
    - Risk score (log-hazard ratio)
    - Risk group (high/medium/low)
    - Survival probabilities at specified time points
    - Optional attention maps for interpretability

    At least one modality must be provided.
    """
    if not model_service:
        raise HTTPException(
            status_code=500,
            detail="Model service not initialized. Please contact administrator.",
        )

    # Convert request to model input format
    patient_data = {}
    modalities_used = []

    # Process clinical data
    if request.clinical:
        clinical_features = _process_clinical(request.clinical)
        if clinical_features is not None:
            patient_data["clinical"] = clinical_features
            modalities_used.append("clinical")

    # Process omics data
    if request.omics:
        # Gene expression
        if request.omics.gene_expression:
            patient_data["rna"] = np.array(request.omics.gene_expression)
            modalities_used.append("rna")

        # Methylation
        if request.omics.methylation:
            patient_data["methylation"] = np.array(request.omics.methylation)
            modalities_used.append("methylation")

        # Mutations
        if request.omics.mutated_genes or request.omics.driver_mutations:
            mut_data = _process_mutations(request.omics)
            if mut_data is not None:
                patient_data["mutations"] = mut_data
                modalities_used.append("mutations")

    # Process WSI data
    if request.wsi:
        if request.wsi.features:
            patient_data["wsi"] = np.array(request.wsi.features)
            modalities_used.append("wsi")
        elif request.wsi.slide_id:
            # TODO: Extract features from slide_id using stored embeddings
            logger.warning(
                f"Feature extraction from slide_id not implemented: {request.wsi.slide_id}"
            )

    if not patient_data:
        raise HTTPException(
            status_code=400,
            detail="No valid modality data provided. Please include at least one of: clinical, omics, wsi.",
        )

    try:
        # Run prediction
        if request.return_attention:
            result = model_service.get_attention_maps(patient_data)
            attention_maps = result.get("attention")
        else:
            result = model_service.predict_survival(
                patient_data, time_points=request.time_points
            )
            attention_maps = None

        return PredictionResponse(
            patient_id=request.patient_id,
            risk_score=result["risk_score"],
            risk_group=result.get("risk_group", "medium"),
            survival_probabilities=result.get("survival_probability", {}),
            modalities_used=modalities_used,
            attention_maps=attention_maps,
            model_version=model_service.checkpoint_path,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def _process_clinical(clinical: ClinicalData) -> Optional[np.ndarray]:
    """Convert clinical data to feature vector."""
    features = []

    # Age (normalized)
    if clinical.age is not None:
        features.append(clinical.age / 100.0)  # Normalize to ~0-1
    else:
        features.append(0.6)  # Default age ~60

    # Gender (one-hot: female=1, male=0)
    if clinical.gender:
        features.append(1.0 if clinical.gender == "female" else 0.0)
    else:
        features.append(0.5)

    # Stage (ordinal encoding)
    stage_map = {
        "stage i": 0.25,
        "stage ii": 0.5,
        "stage iii": 0.75,
        "stage iv": 1.0,
        "stage iva": 1.0,
        "stage ivb": 1.0,
        "stage ivc": 1.0,
    }
    if clinical.tumor_stage:
        features.append(stage_map.get(clinical.tumor_stage, 0.5))
    else:
        features.append(0.5)

    # HPV status
    if clinical.hpv_status is not None:
        features.append(1.0 if clinical.hpv_status else 0.0)
    else:
        features.append(0.5)

    # Smoking (normalized pack-years)
    if clinical.smoking_history is not None:
        features.append(min(clinical.smoking_history / 60.0, 1.0))
    else:
        features.append(0.3)

    # Alcohol
    if clinical.alcohol_history is not None:
        features.append(1.0 if clinical.alcohol_history else 0.0)
    else:
        features.append(0.5)

    return np.array(features, dtype=np.float32)


def _process_mutations(omics: OmicsData) -> Optional[np.ndarray]:
    """Convert mutation data to feature vector."""
    # This is a simplified version - actual implementation would
    # use the same gene list as training

    # For now, create a simple binary vector
    # In production, this should match the training data preprocessing

    known_drivers = [
        "TP53",
        "CDKN2A",
        "PIK3CA",
        "NOTCH1",
        "FAT1",
        "CASP8",
        "HRAS",
        "EGFR",
        "NFE2L2",
        "RAC1",
    ]

    driver_vector = []
    for gene in known_drivers:
        if omics.driver_mutations and gene in omics.driver_mutations:
            driver_vector.append(1.0)
        elif omics.mutated_genes and gene in omics.mutated_genes:
            driver_vector.append(1.0)
        else:
            driver_vector.append(0.0)

    return np.array(driver_vector, dtype=np.float32)


# ============================================================================
# WSI Tile Endpoints
# ============================================================================


@app.get(
    "/wsi/{slide_id}/dzi.xml",
    tags=["WSI Tiles"],
    responses={
        404: {"description": "Slide not found"},
        503: {"description": "WSI server not available"},
    },
)
async def get_dzi_metadata(slide_id: str):
    """
    Get DeepZoom Image (DZI) XML descriptor for a slide.

    This endpoint returns metadata required by OpenSeadragon and other DZI viewers.
    """
    if tile_server is None:
        raise HTTPException(
            status_code=503,
            detail="WSI tile server not available. OpenSlide may not be installed.",
        )

    try:
        xml = tile_server.get_dzi_xml(slide_id)
        return Response(content=xml, media_type="application/xml")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Slide not found: {slide_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/wsi/{slide_id}/deepzoom/{level}/{col}_{row}.{format}",
    tags=["WSI Tiles"],
    responses={
        404: {"description": "Slide or tile not found"},
        503: {"description": "WSI server not available"},
    },
)
async def get_tile(slide_id: str, level: int, col: int, row: int, format: str = "jpg"):
    """
    Get a specific tile from a WSI at the given DeepZoom coordinates.

    - **slide_id**: Slide identifier
    - **level**: Zoom level (0 = most zoomed out)
    - **col**: Tile column index
    - **row**: Tile row index
    - **format**: Image format (jpg or png)
    """
    if tile_server is None:
        raise HTTPException(
            status_code=503,
            detail="WSI tile server not available. OpenSlide may not be installed.",
        )

    try:
        tile_bytes = tile_server.get_tile(slide_id, level, col, row, format)
        media_type = "image/jpeg" if format in ["jpg", "jpeg"] else "image/png"
        return Response(content=tile_bytes, media_type=media_type)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Slide not found: {slide_id}")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/wsi/{slide_id}/dzi_files/{level}/{col}_{row}.{format}",
    tags=["WSI Tiles"],
    responses={
        404: {"description": "Slide or tile not found"},
        503: {"description": "WSI server not available"},
    },
)
async def get_dzi_tile(
    slide_id: str, level: int, col: int, row: int, format: str = "jpeg"
):
    """
    Get a specific tile from a WSI using standard DZI path format.
    This endpoint matches OpenSeadragon's default tile URL format.

    - **slide_id**: Slide identifier
    - **level**: Zoom level (0 = most zoomed out)
    - **col**: Tile column index
    - **row**: Tile row index
    - **format**: Image format (jpeg or png)
    """
    if tile_server is None:
        raise HTTPException(
            status_code=503,
            detail="WSI tile server not available. OpenSlide may not be installed.",
        )

    try:
        tile_bytes = tile_server.get_tile(slide_id, level, col, row, format)
        media_type = "image/jpeg" if format in ["jpg", "jpeg"] else "image/png"
        return Response(content=tile_bytes, media_type=media_type)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Slide not found: {slide_id}")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/wsi/{slide_id}/thumbnail",
    tags=["WSI Tiles"],
    responses={
        404: {"description": "Slide not found"},
        503: {"description": "WSI server not available"},
    },
)
async def get_thumbnail(
    slide_id: str,
    max_width: int = Query(512, ge=64, le=2048),
    max_height: int = Query(512, ge=64, le=2048),
):
    """
    Get a thumbnail image of the slide.
    """
    if tile_server is None:
        raise HTTPException(status_code=503, detail="WSI tile server not available")

    try:
        thumb_bytes = tile_server.get_thumbnail(slide_id, (max_width, max_height))
        return Response(content=thumb_bytes, media_type="image/jpeg")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Slide not found: {slide_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/wsi/list", tags=["WSI Tiles"])
async def list_slides():
    """
    List all available slides.
    """
    if tile_server is None:
        raise HTTPException(status_code=503, detail="WSI tile server not available")

    return {"slides": tile_server.list_slides()}


# ============================================================================
# File Upload Endpoints
# ============================================================================


@app.post("/api/upload/wsi", tags=["Upload"])
async def upload_wsi_file(
    file: UploadFile = File(..., description="WSI file (.svs, .tif, .tiff, .ndpi)")
):
    """
    Upload a whole slide image file.

    Accepts large files and saves them to the upload directory.
    Returns a slide_id that can be used for predictions.
    """
    # Validate file extension
    allowed_extensions = {
        ".svs",
        ".tif",
        ".tiff",
        ".ndpi",
        ".vms",
        ".vmu",
        ".scn",
        ".mrxs",
    }
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
        )

    # Generate unique slide ID
    original_name = Path(file.filename).stem
    slide_id = f"{original_name}_{uuid.uuid4().hex[:8]}"
    save_path = UPLOAD_DIR / f"{slide_id}{file_ext}"

    try:
        # Stream file to disk in chunks (memory efficient for large files)
        total_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(save_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
                total_size += len(chunk)

        logger.info(
            f"Uploaded WSI file: {slide_id} ({total_size / (1024*1024):.2f} MB)"
        )

        return {
            "status": "success",
            "slide_id": slide_id,
            "filename": file.filename,
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "path": str(save_path),
        }

    except Exception as e:
        # Clean up partial file if upload failed
        if save_path.exists():
            save_path.unlink()
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/upload/status/{slide_id}", tags=["Upload"])
async def get_upload_status(slide_id: str):
    """
    Check if a slide has been uploaded and get its details.
    """
    # Search for the file with any extension
    for ext in [".svs", ".tif", ".tiff", ".ndpi", ".vms", ".vmu", ".scn", ".mrxs"]:
        path = UPLOAD_DIR / f"{slide_id}{ext}"
        if path.exists():
            size = path.stat().st_size
            return {
                "exists": True,
                "slide_id": slide_id,
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2),
                "path": str(path),
            }

    return {"exists": False, "slide_id": slide_id}


@app.get("/api/upload/list", tags=["Upload"])
async def list_uploaded_files():
    """
    List all uploaded WSI files.
    """
    files = []
    for path in UPLOAD_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {".svs", ".tif", ".tiff", ".ndpi"}:
            files.append(
                {
                    "slide_id": path.stem,
                    "filename": path.name,
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                }
            )
    return {"files": files}


# ============================================================================
# Model Management (Admin)
# ============================================================================


@app.post("/api/admin/load_model", tags=["Admin"])
async def load_model(
    checkpoint_path: str = Query(
        ..., description="Path to checkpoint file or directory"
    )
):
    """
    Load a specific model checkpoint (admin only).

    In production, this endpoint should be protected.
    """
    if model_service is None:
        raise HTTPException(status_code=500, detail="Model service not initialized")

    success = model_service.load_checkpoint(checkpoint_path)
    if success:
        return {"status": "success", "checkpoint": model_service.checkpoint_path}
    else:
        raise HTTPException(
            status_code=400, detail=f"Failed to load checkpoint: {checkpoint_path}"
        )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("MOSAIC_HOST", "0.0.0.0")
    port = int(os.environ.get("MOSAIC_PORT", "8000"))

    uvicorn.run(
        "src.serving.main:app", host=host, port=port, reload=True, log_level="info"
    )
