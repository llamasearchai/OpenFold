"""
Prediction API Router

Handles biomolecule structure prediction requests and job management.
"""

import uuid
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from loguru import logger
import redis
import json

from ..config import get_settings
from ..models.prediction import (
    PredictionRequest, 
    PredictionResponse, 
    JobStatus, 
    JobStatusResponse,
    BatchPredictionRequest,
    ConfidenceMetrics,
    StructureQuality
)
from ..services.prediction_service import PredictionService
from ..services.job_manager import JobManager
from ..middleware.rate_limiter import rate_limit
from ..middleware.auth import get_current_user

router = APIRouter()
settings = get_settings()

# Initialize services
prediction_service = PredictionService()
job_manager = JobManager()

# Redis client for job status
redis_client = redis.from_url(settings.redis_url)


class PredictionJobCreate(BaseModel):
    """Request model for creating a prediction job"""
    sequence: str = Field(..., min_length=1, max_length=settings.max_sequence_length)
    model_type: str = Field(default="alphafold3", description="Model to use for prediction")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    include_templates: bool = Field(default=True)
    optimize_structure: bool = Field(default=False)
    generate_variants: bool = Field(default=False)
    email_notification: Optional[str] = Field(None, description="Email for job completion notification")
    
    @validator("sequence")
    def validate_sequence(cls, v):
        """Validate protein sequence"""
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c.upper() in valid_amino_acids for c in v):
            raise ValueError("Invalid amino acid sequence")
        return v.upper()


class BatchPredictionJobCreate(BaseModel):
    """Request model for batch prediction jobs"""
    sequences: List[str] = Field(..., min_items=1, max_items=100)
    model_type: str = Field(default="alphafold3")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    parallel_jobs: int = Field(default=4, ge=1, le=10)
    email_notification: Optional[str] = None


@router.post("/submit", response_model=Dict[str, str])
@rate_limit(calls=10, period=60)
async def submit_prediction(
    request: PredictionJobCreate,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Submit a structure prediction job
    
    - **sequence**: Protein amino acid sequence
    - **model_type**: Model to use (alphafold3, esm2, openfold)
    - **confidence_threshold**: Minimum confidence for predictions
    - **include_templates**: Whether to use template structures
    - **optimize_structure**: Whether to perform structure optimization
    - **generate_variants**: Whether to generate structural variants
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job record
        job_data = {
            "job_id": job_id,
            "user_id": current_user.get("user_id") if current_user else "anonymous",
            "sequence": request.sequence,
            "model_type": request.model_type,
            "confidence_threshold": request.confidence_threshold,
            "include_templates": request.include_templates,
            "optimize_structure": request.optimize_structure,
            "generate_variants": request.generate_variants,
            "email_notification": request.email_notification,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
        }
        
        # Store job in Redis
        redis_client.setex(
            f"job:{job_id}", 
            3600 * 24,  # 24 hours TTL
            json.dumps(job_data)
        )
        
        # Queue prediction task
        background_tasks.add_task(
            prediction_service.process_prediction_job,
            job_id,
            request
        )
        
        logger.info(f"Submitted prediction job {job_id} for sequence length {len(request.sequence)}")
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Prediction job submitted successfully",
            "estimated_completion": job_data["estimated_completion"]
        }
        
    except Exception as e:
        logger.error(f"Failed to submit prediction job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


@router.post("/batch", response_model=Dict[str, Any])
@rate_limit(calls=2, period=300)  # More restrictive for batch jobs
async def submit_batch_prediction(
    request: BatchPredictionJobCreate,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Submit a batch of structure prediction jobs
    
    - **sequences**: List of protein sequences to predict
    - **model_type**: Model to use for all predictions
    - **parallel_jobs**: Number of parallel jobs to run
    """
    try:
        batch_id = str(uuid.uuid4())
        job_ids = []
        
        # Create individual jobs for each sequence
        for i, sequence in enumerate(request.sequences):
            job_id = str(uuid.uuid4())
            job_ids.append(job_id)
            
            job_data = {
                "job_id": job_id,
                "batch_id": batch_id,
                "sequence_index": i,
                "user_id": current_user.get("user_id") if current_user else "anonymous",
                "sequence": sequence,
                "model_type": request.model_type,
                "confidence_threshold": request.confidence_threshold,
                "status": "queued",
                "created_at": datetime.utcnow().isoformat()
            }
            
            redis_client.setex(f"job:{job_id}", 3600 * 24, json.dumps(job_data))
        
        # Store batch metadata
        batch_data = {
            "batch_id": batch_id,
            "job_ids": job_ids,
            "total_jobs": len(job_ids),
            "completed_jobs": 0,
            "failed_jobs": 0,
            "status": "running",
            "created_at": datetime.utcnow().isoformat()
        }
        
        redis_client.setex(f"batch:{batch_id}", 3600 * 24, json.dumps(batch_data))
        
        # Queue batch processing
        background_tasks.add_task(
            prediction_service.process_batch_prediction,
            batch_id,
            job_ids,
            request
        )
        
        logger.info(f"Submitted batch prediction {batch_id} with {len(job_ids)} jobs")
        
        return {
            "batch_id": batch_id,
            "job_ids": job_ids,
            "total_jobs": len(job_ids),
            "status": "running",
            "message": "Batch prediction submitted successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to submit batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit batch: {str(e)}")


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a prediction job
    
    - **job_id**: Unique identifier for the prediction job
    """
    try:
        job_data = redis_client.get(f"job:{job_id}")
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_info = json.loads(job_data)
        
        # Calculate progress
        progress = 0
        if job_info["status"] == "queued":
            progress = 0
        elif job_info["status"] == "running":
            progress = 50
        elif job_info["status"] == "completed":
            progress = 100
        elif job_info["status"] == "failed":
            progress = 0
        
        return JobStatusResponse(
            job_id=job_id,
            status=job_info["status"],
            progress=progress,
            created_at=job_info["created_at"],
            updated_at=job_info.get("updated_at", job_info["created_at"]),
            estimated_completion=job_info.get("estimated_completion"),
            error_message=job_info.get("error_message"),
            result_url=f"/api/prediction/result/{job_id}" if job_info["status"] == "completed" else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job status")


@router.get("/result/{job_id}", response_model=PredictionResponse)
async def get_prediction_result(job_id: str):
    """
    Get the result of a completed prediction job
    
    - **job_id**: Unique identifier for the prediction job
    """
    try:
        # Check job status
        job_data = redis_client.get(f"job:{job_id}")
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_info = json.loads(job_data)
        
        if job_info["status"] != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Job is not completed. Current status: {job_info['status']}"
            )
        
        # Get result from storage
        result_data = redis_client.get(f"result:{job_id}")
        if not result_data:
            raise HTTPException(status_code=404, detail="Result not found")
        
        result = json.loads(result_data)
        
        return PredictionResponse(
            job_id=job_id,
            sequence=job_info["sequence"],
            model_type=job_info["model_type"],
            structure=result["structure"],
            confidence_scores=result["confidence_scores"],
            pdb_string=result["pdb_string"],
            quality_metrics=result.get("quality_metrics", {}),
            processing_time=result.get("processing_time", 0),
            created_at=job_info["created_at"],
            completed_at=result.get("completed_at")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get prediction result: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve result")


@router.get("/download/{job_id}")
async def download_structure(job_id: str, format: str = "pdb"):
    """
    Download predicted structure in specified format
    
    - **job_id**: Unique identifier for the prediction job
    - **format**: File format (pdb, cif, sdf)
    """
    try:
        # Validate format
        if format not in ["pdb", "cif", "sdf"]:
            raise HTTPException(status_code=400, detail="Invalid format. Use pdb, cif, or sdf")
        
        # Check if job exists and is completed
        job_data = redis_client.get(f"job:{job_id}")
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_info = json.loads(job_data)
        if job_info["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed")
        
        # Get file path
        file_path = f"./tmp/{job_id}.{format}"
        
        # Generate file if it doesn't exist
        if not os.path.exists(file_path):
            await prediction_service.generate_structure_file(job_id, format)
        
        return FileResponse(
            path=file_path,
            filename=f"structure_{job_id}.{format}",
            media_type="application/octet-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download structure: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download structure")


@router.post("/upload", response_model=Dict[str, str])
async def upload_sequence_file(
    file: UploadFile = File(...),
    model_type: str = "alphafold3",
    background_tasks: BackgroundTasks = None
):
    """
    Upload a FASTA file for structure prediction
    
    - **file**: FASTA file containing protein sequences
    - **model_type**: Model to use for prediction
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.fasta', '.fa', '.fas')):
            raise HTTPException(status_code=400, detail="Invalid file type. Use FASTA format")
        
        # Read file content
        content = await file.read()
        sequences = prediction_service.parse_fasta_content(content.decode())
        
        if not sequences:
            raise HTTPException(status_code=400, detail="No valid sequences found in file")
        
        # Create batch job
        batch_request = BatchPredictionJobCreate(
            sequences=sequences,
            model_type=model_type
        )
        
        # Submit batch prediction
        result = await submit_batch_prediction(batch_request, background_tasks)
        
        return {
            "message": f"Uploaded {len(sequences)} sequences for prediction",
            "batch_id": result["batch_id"],
            "total_jobs": len(sequences)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process uploaded file")


@router.get("/models", response_model=List[Dict[str, Any]])
async def list_available_models():
    """
    List all available prediction models
    """
    try:
        models = await prediction_service.get_available_models()
        return models
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models")


@router.get("/queue/status", response_model=Dict[str, Any])
async def get_queue_status():
    """
    Get current queue status and system metrics
    """
    try:
        status = await job_manager.get_queue_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get queue status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve queue status")


@router.delete("/job/{job_id}")
async def cancel_job(job_id: str, current_user: Optional[Dict] = Depends(get_current_user)):
    """
    Cancel a running prediction job
    
    - **job_id**: Unique identifier for the prediction job
    """
    try:
        # Check if job exists
        job_data = redis_client.get(f"job:{job_id}")
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_info = json.loads(job_data)
        
        # Check if user owns the job
        if current_user and job_info.get("user_id") != current_user.get("user_id"):
            raise HTTPException(status_code=403, detail="Not authorized to cancel this job")
        
        # Cancel job
        success = await job_manager.cancel_job(job_id)
        
        if success:
            return {"message": "Job cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cancel job") 