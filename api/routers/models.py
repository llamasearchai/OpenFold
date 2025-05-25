"""
Models Router

Handles ML model management, registry, versioning, deployment, and monitoring.
Supports multiple model types including AlphaFold, ESM, OpenFold, and custom models.
"""

from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from enum import Enum
import asyncio
import json
from datetime import datetime, timedelta
import uuid
import os

from ..config import get_settings

router = APIRouter()
settings = get_settings()

class ModelType(str, Enum):
    ALPHAFOLD3 = "alphafold3"
    ESM2 = "esm2"
    OPENFOLD = "openfold"
    COLABFOLD = "colabfold"
    CHIMERAAX = "chimeraax"
    CUSTOM = "custom"

class ModelStatus(str, Enum):
    TRAINING = "training"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"

class ModelFramework(str, Enum):
    PYTORCH = "pytorch"
    JAX = "jax"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    CUSTOM = "custom"

class DeploymentStatus(str, Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"

class ModelMetrics(BaseModel):
    """Model performance metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    gdt_ts: Optional[float] = None  # Global Distance Test - Total Score
    rmsd: Optional[float] = None    # Root Mean Square Deviation
    tm_score: Optional[float] = None # Template Modeling Score
    inference_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    custom_metrics: Dict[str, float] = Field(default_factory=dict)

class ModelConfiguration(BaseModel):
    """Model configuration parameters"""
    batch_size: int = 1
    max_sequence_length: int = 1024
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    dropout_rate: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    custom_params: Dict[str, Any] = Field(default_factory=dict)

class ModelVersion(BaseModel):
    """Model version information"""
    version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    description: Optional[str] = None
    metrics: ModelMetrics = Field(default_factory=ModelMetrics)
    configuration: ModelConfiguration = Field(default_factory=ModelConfiguration)
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    is_active: bool = False

class Model(BaseModel):
    """Complete model information"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    type: ModelType
    framework: ModelFramework
    status: ModelStatus = ModelStatus.TRAINING
    versions: List[ModelVersion] = Field(default_factory=list)
    active_version: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    owner: Optional[str] = None
    is_public: bool = False
    tags: List[str] = Field(default_factory=list)
    license: Optional[str] = None
    paper_url: Optional[str] = None
    repository_url: Optional[str] = None

class ModelDeployment(BaseModel):
    """Model deployment information"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    model_version: str
    name: str
    status: DeploymentStatus = DeploymentStatus.PENDING
    endpoint_url: Optional[str] = None
    replicas: int = 1
    cpu_request: str = "1"
    memory_request: str = "2Gi"
    gpu_request: int = 0
    auto_scaling: bool = False
    min_replicas: int = 1
    max_replicas: int = 10
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    health_check_url: Optional[str] = None
    metrics_url: Optional[str] = None

class ModelCreateRequest(BaseModel):
    """Request to create a new model"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    type: ModelType
    framework: ModelFramework
    is_public: bool = False
    tags: List[str] = Field(default_factory=list)
    license: Optional[str] = None
    paper_url: Optional[str] = None
    repository_url: Optional[str] = None

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Model name cannot be empty')
        return v.strip()

class ModelUpdateRequest(BaseModel):
    """Request to update model metadata"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    status: Optional[ModelStatus] = None
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None
    license: Optional[str] = None
    paper_url: Optional[str] = None
    repository_url: Optional[str] = None

class ModelVersionCreateRequest(BaseModel):
    """Request to create a new model version"""
    version: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=500)
    configuration: ModelConfiguration = Field(default_factory=ModelConfiguration)
    metrics: Optional[ModelMetrics] = None

class ModelDeploymentRequest(BaseModel):
    """Request to deploy a model"""
    name: str = Field(..., min_length=1, max_length=255)
    model_version: str
    replicas: int = Field(default=1, ge=1, le=100)
    cpu_request: str = "1"
    memory_request: str = "2Gi"
    gpu_request: int = Field(default=0, ge=0, le=8)
    auto_scaling: bool = False
    min_replicas: int = Field(default=1, ge=1)
    max_replicas: int = Field(default=10, ge=1, le=100)

class ModelSearchRequest(BaseModel):
    """Request for searching models"""
    query: Optional[str] = None
    type: Optional[ModelType] = None
    framework: Optional[ModelFramework] = None
    status: Optional[ModelStatus] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

class ModelListResponse(BaseModel):
    """Response for model listing"""
    models: List[Model]
    total: int
    limit: int
    offset: int

class ModelBenchmarkRequest(BaseModel):
    """Request to benchmark a model"""
    model_version: str
    dataset_id: str
    benchmark_type: str = "standard"
    custom_metrics: List[str] = Field(default_factory=list)

class ModelBenchmarkResult(BaseModel):
    """Results of model benchmarking"""
    benchmark_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    model_version: str
    dataset_id: str
    metrics: ModelMetrics
    execution_time: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = Field(default_factory=dict)

# In-memory storage for demo (replace with actual database)
models_db: Dict[str, Model] = {}
deployments_db: Dict[str, ModelDeployment] = {}
benchmarks_db: Dict[str, ModelBenchmarkResult] = {}

# Initialize with some sample models
def initialize_sample_models():
    """Initialize the database with sample models"""
    if not models_db:
        # AlphaFold3 model
        alphafold3 = Model(
            name="AlphaFold3",
            description="Latest AlphaFold model for protein structure prediction",
            type=ModelType.ALPHAFOLD3,
            framework=ModelFramework.JAX,
            status=ModelStatus.READY,
            is_public=True,
            tags=["protein", "structure", "prediction", "deepmind"],
            license="Apache 2.0",
            paper_url="https://www.nature.com/articles/s41586-021-03819-2",
            repository_url="https://github.com/deepmind/alphafold"
        )
        
        # Add a version
        version = ModelVersion(
            version="3.0.0",
            description="Production release with improved accuracy",
            metrics=ModelMetrics(
                gdt_ts=92.4,
                tm_score=0.95,
                inference_time_ms=2300,
                memory_usage_mb=4200
            ),
            is_active=True
        )
        alphafold3.versions.append(version)
        alphafold3.active_version = "3.0.0"
        models_db[alphafold3.id] = alphafold3
        
        # ESM-2 model
        esm2 = Model(
            name="ESM-2",
            description="Evolutionary Scale Modeling for protein language understanding",
            type=ModelType.ESM2,
            framework=ModelFramework.PYTORCH,
            status=ModelStatus.READY,
            is_public=True,
            tags=["protein", "language", "model", "meta"],
            license="MIT",
            paper_url="https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1",
            repository_url="https://github.com/facebookresearch/esm"
        )
        
        version = ModelVersion(
            version="2.0.0",
            description="Large-scale protein language model",
            metrics=ModelMetrics(
                accuracy=0.89,
                inference_time_ms=1200,
                memory_usage_mb=8100
            ),
            is_active=True
        )
        esm2.versions.append(version)
        esm2.active_version = "2.0.0"
        models_db[esm2.id] = esm2

# Initialize sample models
initialize_sample_models()

@router.post("/create", response_model=Model)
async def create_model(request: ModelCreateRequest):
    """Create a new model"""
    model = Model(
        name=request.name,
        description=request.description,
        type=request.type,
        framework=request.framework,
        is_public=request.is_public,
        tags=request.tags,
        license=request.license,
        paper_url=request.paper_url,
        repository_url=request.repository_url
    )
    
    models_db[model.id] = model
    return model

@router.get("/", response_model=ModelListResponse)
async def list_models(
    query: Optional[str] = Query(None, description="Search query"),
    type: Optional[ModelType] = Query(None, description="Filter by model type"),
    framework: Optional[ModelFramework] = Query(None, description="Filter by framework"),
    status: Optional[ModelStatus] = Query(None, description="Filter by status"),
    is_public: Optional[bool] = Query(None, description="Filter by public status"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(default=0, ge=0, description="Number of results to skip")
):
    """List and search models"""
    filtered_models = list(models_db.values())
    
    # Apply filters
    if query:
        query_lower = query.lower()
        filtered_models = [
            m for m in filtered_models 
            if query_lower in m.name.lower() or 
               (m.description and query_lower in m.description.lower()) or
               any(query_lower in tag.lower() for tag in m.tags)
        ]
    
    if type:
        filtered_models = [m for m in filtered_models if m.type == type]
    
    if framework:
        filtered_models = [m for m in filtered_models if m.framework == framework]
    
    if status:
        filtered_models = [m for m in filtered_models if m.status == status]
    
    if is_public is not None:
        filtered_models = [m for m in filtered_models if m.is_public == is_public]
    
    # Sort by creation date (newest first)
    filtered_models.sort(key=lambda x: x.created_at, reverse=True)
    
    # Apply pagination
    total = len(filtered_models)
    paginated_models = filtered_models[offset:offset + limit]
    
    return ModelListResponse(
        models=paginated_models,
        total=total,
        limit=limit,
        offset=offset
    )

@router.get("/{model_id}", response_model=Model)
async def get_model(model_id: str):
    """Get model by ID"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return models_db[model_id]

@router.put("/{model_id}", response_model=Model)
async def update_model(model_id: str, request: ModelUpdateRequest):
    """Update model metadata"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models_db[model_id]
    
    # Update fields if provided
    if request.name is not None:
        model.name = request.name
    if request.description is not None:
        model.description = request.description
    if request.status is not None:
        model.status = request.status
    if request.is_public is not None:
        model.is_public = request.is_public
    if request.tags is not None:
        model.tags = request.tags
    if request.license is not None:
        model.license = request.license
    if request.paper_url is not None:
        model.paper_url = request.paper_url
    if request.repository_url is not None:
        model.repository_url = request.repository_url
    
    model.updated_at = datetime.utcnow()
    
    return model

@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a model"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check if model has active deployments
    active_deployments = [
        d for d in deployments_db.values() 
        if d.model_id == model_id and d.status == DeploymentStatus.RUNNING
    ]
    
    if active_deployments:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot delete model with {len(active_deployments)} active deployments"
        )
    
    del models_db[model_id]
    
    return {"message": "Model deleted successfully", "model_id": model_id}

@router.post("/{model_id}/versions", response_model=ModelVersion)
async def create_model_version(
    model_id: str, 
    request: ModelVersionCreateRequest,
    file: Optional[UploadFile] = File(None)
):
    """Create a new version of a model"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models_db[model_id]
    
    # Check if version already exists
    existing_versions = [v.version for v in model.versions]
    if request.version in existing_versions:
        raise HTTPException(status_code=400, detail="Version already exists")
    
    # Create new version
    version = ModelVersion(
        version=request.version,
        description=request.description,
        configuration=request.configuration,
        metrics=request.metrics or ModelMetrics()
    )
    
    # Handle file upload if provided
    if file:
        # In a real implementation, save to cloud storage
        version.file_path = f"models/{model_id}/versions/{request.version}/{file.filename}"
        content = await file.read()
        version.file_size = len(content)
        # Calculate checksum in real implementation
        version.checksum = f"sha256:{hash(content)}"
    
    model.versions.append(version)
    model.updated_at = datetime.utcnow()
    
    return version

@router.get("/{model_id}/versions")
async def list_model_versions(model_id: str):
    """List all versions of a model"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models_db[model_id]
    return {
        "model_id": model_id,
        "model_name": model.name,
        "versions": model.versions,
        "active_version": model.active_version
    }

@router.put("/{model_id}/versions/{version}/activate")
async def activate_model_version(model_id: str, version: str):
    """Activate a specific model version"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models_db[model_id]
    
    # Check if version exists
    version_exists = any(v.version == version for v in model.versions)
    if not version_exists:
        raise HTTPException(status_code=404, detail="Version not found")
    
    # Deactivate all versions
    for v in model.versions:
        v.is_active = False
    
    # Activate the specified version
    for v in model.versions:
        if v.version == version:
            v.is_active = True
            break
    
    model.active_version = version
    model.updated_at = datetime.utcnow()
    
    return {"message": f"Version {version} activated", "model_id": model_id, "active_version": version}

@router.post("/{model_id}/deploy", response_model=ModelDeployment)
async def deploy_model(model_id: str, request: ModelDeploymentRequest):
    """Deploy a model version"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models_db[model_id]
    
    # Check if version exists
    version_exists = any(v.version == request.model_version for v in model.versions)
    if not version_exists:
        raise HTTPException(status_code=404, detail="Model version not found")
    
    # Create deployment
    deployment = ModelDeployment(
        model_id=model_id,
        model_version=request.model_version,
        name=request.name,
        replicas=request.replicas,
        cpu_request=request.cpu_request,
        memory_request=request.memory_request,
        gpu_request=request.gpu_request,
        auto_scaling=request.auto_scaling,
        min_replicas=request.min_replicas,
        max_replicas=request.max_replicas,
        status=DeploymentStatus.DEPLOYING
    )
    
    # Simulate deployment process
    deployment.endpoint_url = f"https://api.openfold.ai/models/{deployment.id}/predict"
    deployment.health_check_url = f"https://api.openfold.ai/models/{deployment.id}/health"
    deployment.metrics_url = f"https://api.openfold.ai/models/{deployment.id}/metrics"
    
    deployments_db[deployment.id] = deployment
    
    # In a real implementation, trigger actual deployment
    # For demo, we'll mark it as running after a delay
    deployment.status = DeploymentStatus.RUNNING
    
    return deployment

@router.get("/deployments", response_model=List[ModelDeployment])
async def list_deployments(
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    status: Optional[DeploymentStatus] = Query(None, description="Filter by status")
):
    """List model deployments"""
    deployments = list(deployments_db.values())
    
    if model_id:
        deployments = [d for d in deployments if d.model_id == model_id]
    
    if status:
        deployments = [d for d in deployments if d.status == status]
    
    # Sort by creation date (newest first)
    deployments.sort(key=lambda x: x.created_at, reverse=True)
    
    return deployments

@router.get("/deployments/{deployment_id}", response_model=ModelDeployment)
async def get_deployment(deployment_id: str):
    """Get deployment by ID"""
    if deployment_id not in deployments_db:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    return deployments_db[deployment_id]

@router.put("/deployments/{deployment_id}/scale")
async def scale_deployment(
    deployment_id: str,
    replicas: int = Query(..., ge=1, le=100, description="Number of replicas")
):
    """Scale a deployment"""
    if deployment_id not in deployments_db:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    deployment = deployments_db[deployment_id]
    
    if deployment.status != DeploymentStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Can only scale running deployments")
    
    deployment.replicas = replicas
    deployment.updated_at = datetime.utcnow()
    
    return {
        "message": f"Deployment scaled to {replicas} replicas",
        "deployment_id": deployment_id,
        "replicas": replicas
    }

@router.delete("/deployments/{deployment_id}")
async def stop_deployment(deployment_id: str):
    """Stop a deployment"""
    if deployment_id not in deployments_db:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    deployment = deployments_db[deployment_id]
    deployment.status = DeploymentStatus.STOPPED
    deployment.updated_at = datetime.utcnow()
    
    return {"message": "Deployment stopped", "deployment_id": deployment_id}

@router.post("/{model_id}/benchmark", response_model=ModelBenchmarkResult)
async def benchmark_model(
    model_id: str, 
    request: ModelBenchmarkRequest,
    background_tasks: BackgroundTasks
):
    """Benchmark a model version"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models_db[model_id]
    
    # Check if version exists
    version_exists = any(v.version == request.model_version for v in model.versions)
    if not version_exists:
        raise HTTPException(status_code=404, detail="Model version not found")
    
    # Create benchmark result
    benchmark = ModelBenchmarkResult(
        model_id=model_id,
        model_version=request.model_version,
        dataset_id=request.dataset_id,
        metrics=ModelMetrics(
            # Simulate benchmark results
            gdt_ts=85.2 + (hash(model_id) % 100) / 10,
            tm_score=0.85 + (hash(model_id) % 15) / 100,
            rmsd=2.1 + (hash(model_id) % 20) / 10,
            inference_time_ms=1500 + (hash(model_id) % 1000),
            memory_usage_mb=3200 + (hash(model_id) % 2000)
        ),
        execution_time=120.5,
        details={
            "benchmark_type": request.benchmark_type,
            "custom_metrics": request.custom_metrics,
            "environment": "GPU Tesla V100",
            "batch_size": 1
        }
    )
    
    benchmarks_db[benchmark.benchmark_id] = benchmark
    
    return benchmark

@router.get("/{model_id}/benchmarks")
async def list_model_benchmarks(model_id: str):
    """List benchmarks for a model"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    benchmarks = [b for b in benchmarks_db.values() if b.model_id == model_id]
    benchmarks.sort(key=lambda x: x.created_at, reverse=True)
    
    return {
        "model_id": model_id,
        "benchmarks": benchmarks,
        "total": len(benchmarks)
    }

@router.get("/{model_id}/metrics")
async def get_model_metrics(model_id: str, version: Optional[str] = Query(None)):
    """Get model performance metrics"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models_db[model_id]
    
    if version:
        # Get metrics for specific version
        target_version = None
        for v in model.versions:
            if v.version == version:
                target_version = v
                break
        
        if not target_version:
            raise HTTPException(status_code=404, detail="Version not found")
        
        return {
            "model_id": model_id,
            "version": version,
            "metrics": target_version.metrics,
            "configuration": target_version.configuration
        }
    else:
        # Get metrics for active version
        if not model.active_version:
            raise HTTPException(status_code=400, detail="No active version")
        
        active_version = None
        for v in model.versions:
            if v.version == model.active_version:
                active_version = v
                break
        
        return {
            "model_id": model_id,
            "version": model.active_version,
            "metrics": active_version.metrics if active_version else None,
            "configuration": active_version.configuration if active_version else None
        }

@router.get("/types", response_model=List[str])
async def get_model_types():
    """Get available model types"""
    return [model_type.value for model_type in ModelType]

@router.get("/frameworks", response_model=List[str])
async def get_model_frameworks():
    """Get available model frameworks"""
    return [framework.value for framework in ModelFramework] 