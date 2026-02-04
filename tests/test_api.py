"""
MOSAIC Test Suite

Unit tests for the survival prediction API and model components.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelService:
    """Tests for the model inference service."""
    
    def test_model_service_init(self):
        """Test that model service initializes correctly."""
        from src.serving.model_service import MOSAICModelService
        
        service = MOSAICModelService(device='cpu')
        assert service is not None
        assert service.device.type == 'cpu'
        assert service.is_loaded == False
    
    def test_prepare_input(self):
        """Test input preparation for model."""
        from src.serving.model_service import MOSAICModelService
        
        service = MOSAICModelService(device='cpu')
        
        # Test with numpy array
        patient_data = {
            'clinical': np.array([0.6, 0.5, 0.75, 0.5, 0.3, 0.5]),
            'rna': np.random.randn(3000).astype(np.float32),
        }
        
        inputs = service._prepare_input(patient_data)
        
        assert 'clinical' in inputs
        assert 'rna' in inputs
        assert inputs['clinical'].shape[0] == 1  # Batch dimension added
        assert inputs['rna'].shape[0] == 1


class TestSchemas:
    """Tests for Pydantic validation schemas."""
    
    def test_clinical_data_valid(self):
        """Test valid clinical data."""
        from src.serving.schemas import ClinicalData
        
        data = ClinicalData(
            age=65,
            gender='male',
            tumor_stage='stage iii',
            hpv_status=True
        )
        
        assert data.age == 65
        assert data.gender == 'male'
    
    def test_clinical_data_age_bounds(self):
        """Test age validation bounds."""
        from src.serving.schemas import ClinicalData
        from pydantic import ValidationError
        
        # Valid age
        data = ClinicalData(age=50)
        assert data.age == 50
        
        # Invalid age (too high)
        with pytest.raises(ValidationError):
            ClinicalData(age=150)
    
    def test_prediction_request(self):
        """Test prediction request schema."""
        from src.serving.schemas import PredictionRequest, ClinicalData
        
        request = PredictionRequest(
            patient_id='test-001',
            clinical=ClinicalData(age=60, gender='female'),
            time_points=[365, 730, 1095],
            return_attention=True
        )
        
        assert request.patient_id == 'test-001'
        assert len(request.time_points) == 3
        assert request.return_attention == True
    
    def test_omics_methylation_validation(self):
        """Test methylation value validation (must be 0-1)."""
        from src.serving.schemas import OmicsData
        from pydantic import ValidationError
        
        # Valid methylation
        data = OmicsData(methylation=[0.1, 0.5, 0.9] * 100)
        assert len(data.methylation) == 300
        
        # Invalid methylation (value > 1)
        with pytest.raises(ValidationError):
            OmicsData(methylation=[0.5, 1.5, 0.8] * 100)


class TestAPIEndpoints:
    """Tests for FastAPI endpoints (requires httpx for async testing)."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        try:
            from fastapi.testclient import TestClient
            from src.serving.main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("httpx not installed")
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert data['status'] == 'healthy'
    
    def test_predict_no_data(self, client):
        """Test prediction with no modality data."""
        response = client.post("/api/predict", json={})
        
        # Should return 400 or 422 (validation error)
        assert response.status_code in [400, 422, 500]


class TestEvaluationMetrics:
    """Tests for evaluation metrics."""
    
    def test_c_index_calculation(self):
        """Test concordance index calculation."""
        from lifelines.utils import concordance_index
        
        # Perfect prediction
        times = np.array([1, 2, 3, 4, 5])
        events = np.array([1, 1, 1, 1, 1])
        risk_scores = np.array([5, 4, 3, 2, 1])  # Higher risk = shorter time
        
        c_index = concordance_index(times, -risk_scores, events)
        assert c_index == 1.0  # Perfect concordance
    
    def test_c_index_random(self):
        """Test C-index with random predictions."""
        from lifelines.utils import concordance_index
        
        np.random.seed(42)
        n = 100
        times = np.random.exponential(1, n)
        events = np.ones(n)
        risk_scores = np.random.randn(n)
        
        c_index = concordance_index(times, -risk_scores, events)
        
        # Random should be around 0.5
        assert 0.3 < c_index < 0.7


class TestDataProcessing:
    """Tests for data preprocessing functions."""
    
    def test_patient_registry(self):
        """Test patient registry functionality."""
        from src.data.patient_registry import PatientRegistry
        
        registry = PatientRegistry()
        assert registry is not None
    
    def test_multimodal_dataset_init(self):
        """Test multimodal dataset initialization."""
        # This test requires actual data files, so we just check import
        try:
            from src.data.multimodal_dataset import MultiModalSurvivalDataset
            assert MultiModalSurvivalDataset is not None
        except ImportError as e:
            pytest.skip(f"Import error: {e}")


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
