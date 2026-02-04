"""
Pydantic Schemas for MOSAIC API

Defines input validation schemas for clinical data, omics data,
and prediction request/response models.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class GenderEnum(str, Enum):
    male = "male"
    female = "female"
    unknown = "unknown"


class TumorStageEnum(str, Enum):
    stage_i = "stage i"
    stage_ii = "stage ii"
    stage_iii = "stage iii"
    stage_iv = "stage iv"
    stage_iva = "stage iva"
    stage_ivb = "stage ivb"
    stage_ivc = "stage ivc"
    not_reported = "not reported"
    unknown = "unknown"


class RiskGroupEnum(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


# ============================================================================
# Clinical Data Schema
# ============================================================================

class ClinicalData(BaseModel):
    """
    Clinical data input schema.
    
    Represents patient demographic and clinical information used
    for survival prediction.
    """
    
    age: Optional[float] = Field(
        None,
        ge=0,
        le=120,
        description="Patient age at diagnosis in years"
    )
    
    gender: Optional[GenderEnum] = Field(
        None,
        description="Patient gender"
    )
    
    tumor_stage: Optional[TumorStageEnum] = Field(
        None,
        description="AJCC tumor stage"
    )
    
    # Additional clinical features (extensible)
    hpv_status: Optional[bool] = Field(
        None,
        description="HPV positive status (for HNSC)"
    )
    
    smoking_history: Optional[int] = Field(
        None,
        ge=0,
        description="Pack-years of smoking history"
    )
    
    alcohol_history: Optional[bool] = Field(
        None,
        description="History of alcohol use"
    )
    
    tumor_site: Optional[str] = Field(
        None,
        description="Primary tumor site (e.g., 'oral cavity', 'oropharynx')"
    )
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "age": 62.5,
                "gender": "male",
                "tumor_stage": "stage iii",
                "hpv_status": True,
                "smoking_history": 30
            }
        }


# ============================================================================
# Omics Data Schema
# ============================================================================

class OmicsData(BaseModel):
    """
    Omics data input schema.
    
    Contains gene expression, methylation, and mutation data.
    """
    
    # Gene expression (RNA-seq)
    gene_expression: Optional[List[float]] = Field(
        None,
        description="Normalized gene expression values (log2 TPM). "
                   "Should be a vector of ~3000 values for top variable genes."
    )
    
    gene_names: Optional[List[str]] = Field(
        None,
        description="Gene symbols corresponding to expression values"
    )
    
    # Methylation data
    methylation: Optional[List[float]] = Field(
        None,
        description="Beta values for CpG sites (0-1 range). "
                   "Should be ~5000 most variable probes."
    )
    
    probe_ids: Optional[List[str]] = Field(
        None,
        description="CpG probe IDs corresponding to methylation values"
    )
    
    # Mutation data
    mutated_genes: Optional[List[str]] = Field(
        None,
        description="List of mutated gene symbols"
    )
    
    mutation_types: Optional[Dict[str, str]] = Field(
        None,
        description="Mapping of gene to mutation type (e.g., {'TP53': 'missense'})"
    )
    
    driver_mutations: Optional[List[str]] = Field(
        None,
        description="List of known driver gene mutations present"
    )
    
    @validator('gene_expression')
    def validate_gene_expression(cls, v):
        if v is not None and len(v) < 100:
            raise ValueError("Gene expression vector should have at least 100 genes")
        return v
    
    @validator('methylation')
    def validate_methylation(cls, v):
        if v is not None:
            if len(v) < 100:
                raise ValueError("Methylation vector should have at least 100 probes")
            if any(x < 0 or x > 1 for x in v):
                raise ValueError("Methylation values must be between 0 and 1")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "gene_expression": [2.5, 3.1, 0.8, 5.2],
                "gene_names": ["TP53", "CDKN2A", "PIK3CA", "EGFR"],
                "mutated_genes": ["TP53", "CDKN2A"],
                "driver_mutations": ["TP53"]
            }
        }


# ============================================================================
# WSI Data Schema
# ============================================================================

class WSIData(BaseModel):
    """
    Whole slide image data schema.
    
    Accepts either pre-extracted features or a slide reference.
    """
    
    slide_id: Optional[str] = Field(
        None,
        description="Slide identifier for server-side feature extraction"
    )
    
    features: Optional[List[List[float]]] = Field(
        None,
        description="Pre-extracted patch features (N x D matrix where N is "
                   "number of patches and D is feature dimension, typically 1024)"
    )
    
    coordinates: Optional[List[List[int]]] = Field(
        None,
        description="Patch coordinates (N x 2 for x, y positions)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "slide_id": "TCGA-CV-7432-01A-01-TS1"
            }
        }


# ============================================================================
# Prediction Request/Response
# ============================================================================

class PredictionRequest(BaseModel):
    """
    Full prediction request combining all data modalities.
    """
    
    patient_id: Optional[str] = Field(
        None,
        description="Optional patient identifier for tracking"
    )
    
    clinical: Optional[ClinicalData] = Field(
        None,
        description="Clinical data"
    )
    
    omics: Optional[OmicsData] = Field(
        None,
        description="Omics data (expression, methylation, mutations)"
    )
    
    wsi: Optional[WSIData] = Field(
        None,
        description="Whole slide image data"
    )
    
    # Time points for survival curve
    time_points: Optional[List[int]] = Field(
        default=[365, 730, 1095, 1825, 3650],
        description="Time points (in days) for survival probability estimation"
    )
    
    return_attention: bool = Field(
        default=False,
        description="Whether to return attention maps for interpretability"
    )
    
    @validator('time_points')
    def validate_time_points(cls, v):
        if v is not None and any(t <= 0 for t in v):
            raise ValueError("Time points must be positive")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "TCGA-CV-7432",
                "clinical": {
                    "age": 62.5,
                    "gender": "male",
                    "tumor_stage": "stage iii"
                },
                "time_points": [365, 730, 1095],
                "return_attention": False
            }
        }


class SurvivalCurve(BaseModel):
    """Survival probability at specific time points."""
    time_days: int
    probability: float


class PredictionResponse(BaseModel):
    """
    Prediction response with risk score and survival estimates.
    """
    
    patient_id: Optional[str] = Field(
        None,
        description="Patient identifier from request"
    )
    
    risk_score: float = Field(
        ...,
        description="Predicted log-hazard ratio (higher = higher risk)"
    )
    
    risk_group: RiskGroupEnum = Field(
        ...,
        description="Risk stratification category"
    )
    
    survival_probabilities: Dict[int, float] = Field(
        ...,
        description="Estimated survival probability at each time point (days)"
    )
    
    modalities_used: List[str] = Field(
        ...,
        description="List of modalities that were available and used"
    )
    
    attention_maps: Optional[Dict[str, Any]] = Field(
        None,
        description="Attention weights for interpretability (if requested)"
    )
    
    model_version: Optional[str] = Field(
        None,
        description="Model checkpoint identifier"
    )
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "patient_id": "TCGA-CV-7432",
                "risk_score": 0.75,
                "risk_group": "high",
                "survival_probabilities": {
                    365: 0.82,
                    730: 0.65,
                    1095: 0.48
                },
                "modalities_used": ["clinical", "rna", "wsi"],
                "model_version": "fold_0_v1"
            }
        }


# ============================================================================
# Health Check
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(
        ...,
        description="Service status"
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded and ready"
    )
    
    available_modalities: List[str] = Field(
        ...,
        description="List of modalities the model supports"
    )
    
    wsi_server_available: bool = Field(
        ...,
        description="Whether WSI tile server is available"
    )
    
    version: str = Field(
        default="1.0.0",
        description="API version"
    )


# ============================================================================
# Error Response
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(
        ...,
        description="Error message"
    )
    
    detail: Optional[str] = Field(
        None,
        description="Detailed error information"
    )
    
    code: Optional[str] = Field(
        None,
        description="Error code for programmatic handling"
    )
