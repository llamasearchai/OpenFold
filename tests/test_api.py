"""
Comprehensive API Tests

Tests for all OpenFold API endpoints including prediction, datasets, models, and AI agents.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json
import tempfile
import os
from pathlib import Path

from api.main import app
from api.config import get_settings

# Test client
client = TestClient(app)

class TestRootEndpoints:
    """Test root and health endpoints"""
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_api_info_endpoint(self):
        """Test the API info endpoint"""
        response = client.get("/api/info")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "OpenFold API"
        assert data["author"] == "Nik Jois"
        assert "features" in data
        assert "ai_capabilities" in data

class TestPredictionAPI:
    """Test prediction endpoints"""
    
    def test_submit_prediction(self):
        """Test submitting a prediction job"""
        payload = {
            "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "model_type": "alphafold3",
            "confidence_threshold": 0.7
        }
        
        response = client.post("/api/prediction/submit", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "submitted"
    
    def test_submit_prediction_invalid_sequence(self):
        """Test submitting prediction with invalid sequence"""
        payload = {
            "sequence": "INVALID123",
            "model_type": "alphafold3"
        }
        
        response = client.post("/api/prediction/submit", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_get_job_status(self):
        """Test getting job status"""
        # First submit a job
        payload = {
            "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "model_type": "esm2"
        }
        
        submit_response = client.post("/api/prediction/submit", json=payload)
        job_id = submit_response.json()["job_id"]
        
        # Check status
        response = client.get(f"/api/prediction/status/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
    
    def test_get_job_status_not_found(self):
        """Test getting status for non-existent job"""
        response = client.get("/api/prediction/status/nonexistent-job-id")
        assert response.status_code == 404
    
    def test_list_models(self):
        """Test listing available models"""
        response = client.get("/api/prediction/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0
    
    def test_batch_prediction(self):
        """Test batch prediction submission"""
        payload = {
            "sequences": [
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGG"
            ],
            "model_type": "esm2",
            "batch_name": "test_batch"
        }
        
        response = client.post("/api/prediction/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "batch_id" in data
        assert data["total_jobs"] == 2

class TestDatasetsAPI:
    """Test datasets endpoints"""
    
    def test_create_dataset(self):
        """Test creating a new dataset"""
        payload = {
            "name": "Test Dataset",
            "description": "A test dataset for unit testing",
            "type": "protein",
            "format": "fasta",
            "is_public": False,
            "tags": ["test", "protein"]
        }
        
        response = client.post("/api/datasets/create", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Dataset"
        assert data["type"] == "protein"
        assert data["format"] == "fasta"
    
    def test_list_datasets(self):
        """Test listing datasets"""
        response = client.get("/api/datasets/")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
    
    def test_list_datasets_with_filters(self):
        """Test listing datasets with filters"""
        params = {
            "type": "protein",
            "format": "fasta",
            "limit": 10
        }
        
        response = client.get("/api/datasets/", params=params)
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10
    
    def test_get_dataset(self):
        """Test getting a specific dataset"""
        # First create a dataset
        create_payload = {
            "name": "Test Dataset for Get",
            "type": "protein",
            "format": "fasta"
        }
        
        create_response = client.post("/api/datasets/create", json=create_payload)
        dataset_id = create_response.json()["id"]
        
        # Get the dataset
        response = client.get(f"/api/datasets/{dataset_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == dataset_id
        assert data["name"] == "Test Dataset for Get"
    
    def test_update_dataset(self):
        """Test updating dataset metadata"""
        # First create a dataset
        create_payload = {
            "name": "Dataset to Update",
            "type": "protein",
            "format": "fasta"
        }
        
        create_response = client.post("/api/datasets/create", json=create_payload)
        dataset_id = create_response.json()["id"]
        
        # Update the dataset
        update_payload = {
            "name": "Updated Dataset Name",
            "description": "Updated description",
            "tags": ["updated", "test"]
        }
        
        response = client.put(f"/api/datasets/{dataset_id}", json=update_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Dataset Name"
        assert data["description"] == "Updated description"
    
    def test_delete_dataset(self):
        """Test deleting a dataset"""
        # First create a dataset
        create_payload = {
            "name": "Dataset to Delete",
            "type": "protein",
            "format": "fasta"
        }
        
        create_response = client.post("/api/datasets/create", json=create_payload)
        dataset_id = create_response.json()["id"]
        
        # Delete the dataset
        response = client.delete(f"/api/datasets/{dataset_id}")
        assert response.status_code == 200
        
        # Verify it's deleted
        get_response = client.get(f"/api/datasets/{dataset_id}")
        assert get_response.status_code == 404

class TestModelsAPI:
    """Test models endpoints"""
    
    def test_list_models(self):
        """Test listing models"""
        response = client.get("/api/models/")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total" in data
        # Should have sample models initialized
        assert data["total"] >= 2
    
    def test_get_model(self):
        """Test getting a specific model"""
        # First get list of models
        list_response = client.get("/api/models/")
        models = list_response.json()["models"]
        
        if models:
            model_id = models[0]["id"]
            response = client.get(f"/api/models/{model_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == model_id
    
    def test_create_model(self):
        """Test creating a new model"""
        payload = {
            "name": "Test Model",
            "description": "A test model for unit testing",
            "type": "custom",
            "framework": "pytorch",
            "is_public": False,
            "tags": ["test", "custom"],
            "license": "MIT"
        }
        
        response = client.post("/api/models/create", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Model"
        assert data["type"] == "custom"
        assert data["framework"] == "pytorch"
    
    def test_get_model_types(self):
        """Test getting available model types"""
        response = client.get("/api/models/types")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert "alphafold3" in data
        assert "esm2" in data
    
    def test_get_model_frameworks(self):
        """Test getting available model frameworks"""
        response = client.get("/api/models/frameworks")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert "pytorch" in data
        assert "jax" in data

class TestAgentsAPI:
    """Test AI agents endpoints"""
    
    def test_analyze_structure(self):
        """Test structure analysis with AI agent"""
        payload = {
            "structure_data": "HEADER    TEST STRUCTURE\nATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 20.00           C\nEND",
            "analysis_type": "comprehensive",
            "include_binding_sites": True,
            "include_druggability": True
        }
        
        response = client.post("/api/agents/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "insights" in data
        assert "binding_sites" in data
        assert "druggability_assessment" in data
    
    def test_compare_structures(self):
        """Test structure comparison with AI agent"""
        payload = {
            "structure1": "HEADER    STRUCTURE 1\nATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 20.00           C\nEND",
            "structure2": "HEADER    STRUCTURE 2\nATOM      1  CA  ALA A   1      1.000   1.000   1.000  1.00 20.00           C\nEND",
            "comparison_type": "detailed"
        }
        
        response = client.post("/api/agents/compare", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "structural_similarity" in data
        assert "functional_differences" in data
        assert "rmsd" in data
    
    def test_generate_hypothesis(self):
        """Test research hypothesis generation"""
        payload = {
            "structure_data": "HEADER    TEST STRUCTURE\nATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 20.00           C\nEND",
            "research_context": "drug discovery",
            "target_disease": "cancer",
            "num_hypotheses": 3
        }
        
        response = client.post("/api/agents/hypothesis", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "hypotheses" in data
        assert len(data["hypotheses"]) <= 3
    
    def test_suggest_experiments(self):
        """Test experiment suggestion"""
        payload = {
            "research_question": "How does this protein interact with small molecules?",
            "structure_data": "HEADER    TEST STRUCTURE\nATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 20.00           C\nEND",
            "budget_level": "medium",
            "timeframe": "6 months"
        }
        
        response = client.post("/api/agents/experiments", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "experiments" in data
        assert "estimated_cost" in data
        assert "timeline" in data
    
    def test_get_ai_capabilities(self):
        """Test getting AI capabilities"""
        response = client.get("/api/agents/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert "analysis_types" in data
        assert "supported_formats" in data

class TestFileUpload:
    """Test file upload functionality"""
    
    def test_upload_fasta_file(self):
        """Test uploading a FASTA file"""
        # Create a temporary FASTA file
        fasta_content = ">test_protein\nMKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            temp_file_path = f.name
        
        try:
            # First create a dataset
            create_payload = {
                "name": "Upload Test Dataset",
                "type": "protein",
                "format": "fasta"
            }
            
            create_response = client.post("/api/datasets/create", json=create_payload)
            dataset_id = create_response.json()["id"]
            
            # Upload file
            with open(temp_file_path, 'rb') as f:
                response = client.post(
                    f"/api/datasets/{dataset_id}/upload",
                    files={"file": ("test.fasta", f, "text/plain")}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "File uploaded successfully"
            assert data["dataset_id"] == dataset_id
            
        finally:
            # Clean up
            os.unlink(temp_file_path)

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_endpoint(self):
        """Test accessing invalid endpoint"""
        response = client.get("/api/invalid/endpoint")
        assert response.status_code == 404
    
    def test_invalid_json_payload(self):
        """Test sending invalid JSON"""
        response = client.post(
            "/api/prediction/submit",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test missing required fields in request"""
        payload = {
            "model_type": "alphafold3"
            # Missing required 'sequence' field
        }
        
        response = client.post("/api/prediction/submit", json=payload)
        assert response.status_code == 422
    
    def test_rate_limiting_headers(self):
        """Test that rate limiting headers are present"""
        response = client.get("/")
        # Check for common rate limiting headers
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers

class TestAuthentication:
    """Test authentication and authorization (if implemented)"""
    
    def test_public_endpoints_accessible(self):
        """Test that public endpoints are accessible without auth"""
        public_endpoints = [
            "/",
            "/health",
            "/api/info",
            "/api/models/types",
            "/api/models/frameworks"
        ]
        
        for endpoint in public_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200

@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test asynchronous endpoint behavior"""
    
    async def test_async_prediction_submission(self):
        """Test async prediction submission"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            payload = {
                "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "model_type": "esm2"
            }
            
            response = await ac.post("/api/prediction/submit", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
    
    async def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Submit multiple requests concurrently
            tasks = []
            for i in range(5):
                payload = {
                    "sequence": f"MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG{i}",
                    "model_type": "esm2"
                }
                task = ac.post("/api/prediction/submit", json=payload)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert "job_id" in data

class TestPerformance:
    """Test performance characteristics"""
    
    def test_response_time_root_endpoint(self):
        """Test response time for root endpoint"""
        import time
        
        start_time = time.time()
        response = client.get("/")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second
    
    def test_large_sequence_handling(self):
        """Test handling of large sequences"""
        # Create a large sequence (but within limits)
        large_sequence = "A" * 1000
        
        payload = {
            "sequence": large_sequence,
            "model_type": "esm2"
        }
        
        response = client.post("/api/prediction/submit", json=payload)
        assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 