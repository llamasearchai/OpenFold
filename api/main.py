from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from loguru import logger
import time
import uuid

from .routers import prediction, datasets, models, agents
from .config import get_settings

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting OpenFold API")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    yield
    # Shutdown logic
    logger.info("Shutting down OpenFold API")

app = FastAPI(
    title="OpenFold API",
    description="Advanced Biomolecule Structure Prediction Platform with AI-Powered Insights",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1", "api.openfold.ai"]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://openfold.ai", "https://www.openfold.ai"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    logger.info(
        f"Request {request_id} - {request.method} {request.url.path} - "
        f"Status: {response.status_code} - Time: {process_time:.3f}s"
    )
    
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(f"Request {request_id} - Unhandled exception: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": request_id
        }
    )

# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id
        }
    )

# Include routers
app.include_router(prediction.router, prefix="/api/prediction", tags=["prediction"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(agents.router, prefix="/api/agents", tags=["ai-agents"])

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to OpenFold API",
        "status": "online",
        "version": "1.0.0",
        "description": "Advanced Biomolecule Structure Prediction Platform",
        "author": "Nik Jois <nikjois@llamasearch.ai>",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "endpoints": {
            "prediction": "/api/prediction",
            "datasets": "/api/datasets", 
            "models": "/api/models",
            "ai_agents": "/api/agents"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "environment": settings.environment
    }

@app.get("/api/info")
async def api_info():
    """Detailed API information"""
    return {
        "name": "OpenFold API",
        "version": "1.0.0",
        "description": "Advanced Biomolecule Structure Prediction Platform with AI-Powered Insights",
        "author": "Nik Jois",
        "email": "nikjois@llamasearch.ai",
        "license": "MIT",
        "repository": "https://github.com/llamasearchai/OpenFold",
        "features": [
            "Protein structure prediction",
            "Multi-model support (AlphaFold3, ESM-2, OpenFold, ColabFold)",
            "AI-powered structure analysis",
            "Dataset management",
            "Model registry and deployment",
            "Real-time job processing",
            "Comprehensive validation"
        ],
        "supported_formats": [
            "FASTA", "PDB", "CIF", "SDF", "MOL2"
        ],
        "ai_capabilities": [
            "Structure analysis and insights",
            "Binding site prediction",
            "Drug-target interaction analysis",
            "Research hypothesis generation",
            "Scientific abstract generation"
        ]
    } 