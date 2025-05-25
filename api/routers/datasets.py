"""
Datasets Router

Handles dataset management, upload, validation, and processing for biomolecule data.
Supports multiple formats including FASTA, PDB, CIF, and custom formats.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from enum import Enum
import asyncio
import io
import zipfile
from datetime import datetime
import uuid

from ..models.prediction import JobStatus
from ..config import get_settings

router = APIRouter()
settings = get_settings()

class DatasetFormat(str, Enum):
    FASTA = "fasta"
    PDB = "pdb"
    CIF = "cif"
    SDF = "sdf"
    MOL2 = "mol2"
    CUSTOM = "custom"

class DatasetType(str, Enum):
    PROTEIN = "protein"
    RNA = "rna"
    DNA = "dna"
    SMALL_MOLECULE = "small_molecule"
    COMPLEX = "complex"

class DatasetStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    VALIDATED = "validated"
    FAILED = "failed"
    READY = "ready"

class DatasetMetadata(BaseModel):
    """Metadata for dataset entries"""
    organism: Optional[str] = None
    resolution: Optional[float] = None
    method: Optional[str] = None
    publication_year: Optional[int] = None
    doi: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    annotations: Dict[str, Any] = Field(default_factory=dict)

class DatasetEntry(BaseModel):
    """Individual entry in a dataset"""
    id: str
    name: str
    sequence: Optional[str] = None
    structure_data: Optional[str] = None
    format: DatasetFormat
    type: DatasetType
    metadata: DatasetMetadata = Field(default_factory=DatasetMetadata)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    file_size: Optional[int] = None
    checksum: Optional[str] = None

class Dataset(BaseModel):
    """Complete dataset information"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    type: DatasetType
    format: DatasetFormat
    status: DatasetStatus = DatasetStatus.UPLOADING
    entries: List[DatasetEntry] = Field(default_factory=list)
    total_entries: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    owner: Optional[str] = None
    is_public: bool = False
    tags: List[str] = Field(default_factory=list)
    validation_results: Dict[str, Any] = Field(default_factory=dict)

class DatasetCreateRequest(BaseModel):
    """Request to create a new dataset"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    type: DatasetType
    format: DatasetFormat
    is_public: bool = False
    tags: List[str] = Field(default_factory=list)

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Dataset name cannot be empty')
        return v.strip()

class DatasetUpdateRequest(BaseModel):
    """Request to update dataset metadata"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None

class DatasetSearchRequest(BaseModel):
    """Request for searching datasets"""
    query: Optional[str] = None
    type: Optional[DatasetType] = None
    format: Optional[DatasetFormat] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

class DatasetValidationResult(BaseModel):
    """Results of dataset validation"""
    is_valid: bool
    total_entries: int
    valid_entries: int
    invalid_entries: int
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)

class DatasetListResponse(BaseModel):
    """Response for dataset listing"""
    datasets: List[Dataset]
    total: int
    limit: int
    offset: int

# In-memory storage for demo (replace with actual database)
datasets_db: Dict[str, Dataset] = {}

async def validate_dataset_file(file_content: bytes, format: DatasetFormat, type: DatasetType) -> DatasetValidationResult:
    """Validate uploaded dataset file"""
    try:
        content = file_content.decode('utf-8')
        errors = []
        warnings = []
        valid_entries = 0
        total_entries = 0
        
        if format == DatasetFormat.FASTA:
            # Basic FASTA validation
            lines = content.strip().split('\n')
            current_sequence = ""
            
            for i, line in enumerate(lines):
                if line.startswith('>'):
                    if current_sequence and len(current_sequence) < 10:
                        warnings.append(f"Short sequence at line {i}: {len(current_sequence)} residues")
                    total_entries += 1
                    current_sequence = ""
                elif line.strip():
                    current_sequence += line.strip()
                    # Validate sequence characters
                    if type == DatasetType.PROTEIN:
                        invalid_chars = set(line.upper()) - set('ACDEFGHIKLMNPQRSTVWY')
                        if invalid_chars:
                            errors.append(f"Invalid protein characters at line {i+1}: {invalid_chars}")
                    elif type == DatasetType.DNA:
                        invalid_chars = set(line.upper()) - set('ATCG')
                        if invalid_chars:
                            errors.append(f"Invalid DNA characters at line {i+1}: {invalid_chars}")
            
            if current_sequence and len(current_sequence) < 10:
                warnings.append(f"Short sequence at end: {len(current_sequence)} residues")
            
            valid_entries = total_entries - len(errors)
            
        elif format == DatasetFormat.PDB:
            # Basic PDB validation
            lines = content.split('\n')
            atom_count = 0
            has_header = False
            
            for line in lines:
                if line.startswith('HEADER'):
                    has_header = True
                elif line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_count += 1
                    if len(line) < 80:
                        warnings.append(f"Short PDB line: {line[:20]}...")
            
            if not has_header:
                warnings.append("No HEADER record found")
            
            if atom_count == 0:
                errors.append("No ATOM records found")
            
            total_entries = 1
            valid_entries = 1 if not errors else 0
            
        else:
            # Generic validation for other formats
            total_entries = len(content.split('\n'))
            valid_entries = total_entries
        
        statistics = {
            "file_size": len(file_content),
            "line_count": len(content.split('\n')),
            "character_count": len(content)
        }
        
        return DatasetValidationResult(
            is_valid=len(errors) == 0,
            total_entries=total_entries,
            valid_entries=valid_entries,
            invalid_entries=total_entries - valid_entries,
            errors=errors,
            warnings=warnings,
            statistics=statistics
        )
        
    except Exception as e:
        return DatasetValidationResult(
            is_valid=False,
            total_entries=0,
            valid_entries=0,
            invalid_entries=0,
            errors=[f"Validation error: {str(e)}"],
            warnings=[],
            statistics={}
        )

@router.post("/create", response_model=Dataset)
async def create_dataset(request: DatasetCreateRequest):
    """Create a new empty dataset"""
    dataset = Dataset(
        name=request.name,
        description=request.description,
        type=request.type,
        format=request.format,
        is_public=request.is_public,
        tags=request.tags,
        status=DatasetStatus.READY
    )
    
    datasets_db[dataset.id] = dataset
    return dataset

@router.post("/{dataset_id}/upload")
async def upload_dataset_file(
    dataset_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    validate_immediately: bool = Query(default=True)
):
    """Upload a file to an existing dataset"""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets_db[dataset_id]
    
    # Read file content
    content = await file.read()
    
    # Update dataset status
    dataset.status = DatasetStatus.PROCESSING
    dataset.updated_at = datetime.utcnow()
    
    if validate_immediately:
        # Validate the file
        validation_result = await validate_dataset_file(content, dataset.format, dataset.type)
        dataset.validation_results = validation_result.dict()
        
        if validation_result.is_valid:
            dataset.status = DatasetStatus.VALIDATED
        else:
            dataset.status = DatasetStatus.FAILED
    else:
        # Schedule background validation
        background_tasks.add_task(validate_dataset_background, dataset_id, content, dataset.format, dataset.type)
    
    return {
        "message": "File uploaded successfully",
        "dataset_id": dataset_id,
        "filename": file.filename,
        "size": len(content),
        "status": dataset.status
    }

async def validate_dataset_background(dataset_id: str, content: bytes, format: DatasetFormat, type: DatasetType):
    """Background task for dataset validation"""
    validation_result = await validate_dataset_file(content, format, type)
    
    if dataset_id in datasets_db:
        dataset = datasets_db[dataset_id]
        dataset.validation_results = validation_result.dict()
        dataset.status = DatasetStatus.VALIDATED if validation_result.is_valid else DatasetStatus.FAILED
        dataset.updated_at = datetime.utcnow()

@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    query: Optional[str] = Query(None, description="Search query"),
    type: Optional[DatasetType] = Query(None, description="Filter by dataset type"),
    format: Optional[DatasetFormat] = Query(None, description="Filter by format"),
    is_public: Optional[bool] = Query(None, description="Filter by public status"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(default=0, ge=0, description="Number of results to skip")
):
    """List and search datasets"""
    filtered_datasets = list(datasets_db.values())
    
    # Apply filters
    if query:
        query_lower = query.lower()
        filtered_datasets = [
            d for d in filtered_datasets 
            if query_lower in d.name.lower() or 
               (d.description and query_lower in d.description.lower()) or
               any(query_lower in tag.lower() for tag in d.tags)
        ]
    
    if type:
        filtered_datasets = [d for d in filtered_datasets if d.type == type]
    
    if format:
        filtered_datasets = [d for d in filtered_datasets if d.format == format]
    
    if is_public is not None:
        filtered_datasets = [d for d in filtered_datasets if d.is_public == is_public]
    
    # Sort by creation date (newest first)
    filtered_datasets.sort(key=lambda x: x.created_at, reverse=True)
    
    # Apply pagination
    total = len(filtered_datasets)
    paginated_datasets = filtered_datasets[offset:offset + limit]
    
    return DatasetListResponse(
        datasets=paginated_datasets,
        total=total,
        limit=limit,
        offset=offset
    )

@router.get("/{dataset_id}", response_model=Dataset)
async def get_dataset(dataset_id: str):
    """Get dataset by ID"""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return datasets_db[dataset_id]

@router.put("/{dataset_id}", response_model=Dataset)
async def update_dataset(dataset_id: str, request: DatasetUpdateRequest):
    """Update dataset metadata"""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets_db[dataset_id]
    
    # Update fields if provided
    if request.name is not None:
        dataset.name = request.name
    if request.description is not None:
        dataset.description = request.description
    if request.is_public is not None:
        dataset.is_public = request.is_public
    if request.tags is not None:
        dataset.tags = request.tags
    
    dataset.updated_at = datetime.utcnow()
    
    return dataset

@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    del datasets_db[dataset_id]
    
    return {"message": "Dataset deleted successfully", "dataset_id": dataset_id}

@router.get("/{dataset_id}/validate", response_model=DatasetValidationResult)
async def validate_dataset(dataset_id: str):
    """Get validation results for a dataset"""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets_db[dataset_id]
    
    if not dataset.validation_results:
        raise HTTPException(status_code=400, detail="Dataset has not been validated yet")
    
    return DatasetValidationResult(**dataset.validation_results)

@router.get("/{dataset_id}/download")
async def download_dataset(dataset_id: str, format: Optional[str] = Query(None)):
    """Download dataset in specified format"""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets_db[dataset_id]
    
    # For demo purposes, return a simple text file
    content = f"# Dataset: {dataset.name}\n"
    content += f"# Type: {dataset.type}\n"
    content += f"# Format: {dataset.format}\n"
    content += f"# Created: {dataset.created_at}\n"
    content += f"# Entries: {dataset.total_entries}\n\n"
    
    if dataset.description:
        content += f"# Description: {dataset.description}\n\n"
    
    # Add sample data based on format
    if dataset.format == DatasetFormat.FASTA:
        content += ">sample_protein_1\nMKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG\n"
        content += ">sample_protein_2\nMALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGG\n"
    elif dataset.format == DatasetFormat.PDB:
        content += "HEADER    SAMPLE STRUCTURE\n"
        content += "ATOM      1  N   ALA A   1      20.154  16.967  10.000  1.00 20.00           N\n"
        content += "ATOM      2  CA  ALA A   1      21.618  16.967  10.000  1.00 20.00           C\n"
    
    # Create response
    response = StreamingResponse(
        io.StringIO(content),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={dataset.name}.{dataset.format}"}
    )
    
    return response

@router.get("/{dataset_id}/statistics")
async def get_dataset_statistics(dataset_id: str):
    """Get detailed statistics for a dataset"""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets_db[dataset_id]
    
    stats = {
        "basic_info": {
            "id": dataset.id,
            "name": dataset.name,
            "type": dataset.type,
            "format": dataset.format,
            "status": dataset.status,
            "total_entries": dataset.total_entries,
            "created_at": dataset.created_at,
            "updated_at": dataset.updated_at
        },
        "validation_summary": dataset.validation_results if dataset.validation_results else None,
        "metadata": {
            "is_public": dataset.is_public,
            "tags": dataset.tags,
            "owner": dataset.owner
        }
    }
    
    return stats

@router.post("/{dataset_id}/export")
async def export_dataset(
    dataset_id: str,
    export_format: DatasetFormat = Query(..., description="Target export format"),
    include_metadata: bool = Query(default=True, description="Include metadata in export")
):
    """Export dataset to different format"""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets_db[dataset_id]
    
    # Create export job (in real implementation, this would be async)
    export_job_id = str(uuid.uuid4())
    
    return {
        "export_job_id": export_job_id,
        "status": "processing",
        "source_format": dataset.format,
        "target_format": export_format,
        "estimated_completion": "2-5 minutes",
        "download_url": f"/api/datasets/exports/{export_job_id}/download"
    } 