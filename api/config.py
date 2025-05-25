"""
Configuration management for OpenFold API
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "OpenFold"
    app_version: str = "1.0.0"
    app_description: str = "Advanced Biomolecule Structure Prediction Platform"
    debug: bool = False
    log_level: str = "INFO"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    
    # Database
    database_url: str = "postgresql://openfold:password@localhost:5432/openfold"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # Security
    secret_key: str = "your-super-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    openai_max_tokens: int = 4096
    openai_temperature: float = 0.1
    
    # LangChain
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: Optional[str] = None
    langchain_project: str = "openfold"
    
    # AWS
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-west-2"
    aws_s3_bucket: str = "openfold-data"
    
    # Azure
    azure_storage_connection_string: Optional[str] = None
    azure_container_name: str = "openfold-data"
    
    # Google Cloud
    google_application_credentials: Optional[str] = None
    gcp_project_id: Optional[str] = None
    gcp_bucket_name: str = "openfold-data"
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "openfold-experiments"
    mlflow_s3_endpoint_url: str = "http://localhost:9000"
    
    # Weights & Biases
    wandb_api_key: Optional[str] = None
    wandb_project: str = "openfold"
    wandb_entity: Optional[str] = None
    
    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_serializer: str = "json"
    celery_result_serializer: str = "json"
    celery_accept_content: List[str] = ["json"]
    celery_timezone: str = "UTC"
    
    # Model Configuration
    model_cache_dir: str = "./models"
    model_download_url: str = "https://storage.googleapis.com/openfold-models"
    default_model: str = "alphafold3"
    max_sequence_length: int = 2048
    batch_size: int = 1
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Data Processing
    max_file_size: str = "100MB"
    allowed_file_types: List[str] = [".fasta", ".pdb", ".cif", ".sdf"]
    temp_dir: str = "./tmp"
    cleanup_interval: int = 3600
    
    # Monitoring
    prometheus_port: int = 9090
    metrics_enabled: bool = True
    health_check_interval: int = 30
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    rate_limit_per_day: int = 10000
    
    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "https://openfold.llamasearch.ai"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers: List[str] = ["*"]
    
    # Email
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    email_from: str = "noreply@llamasearch.ai"
    
    # Backup
    backup_enabled: bool = True
    backup_interval: int = 24
    backup_retention_days: int = 30
    backup_s3_bucket: str = "openfold-backups"
    
    @validator("max_file_size")
    def parse_file_size(cls, v):
        """Convert file size string to bytes"""
        if isinstance(v, str):
            size_map = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}
            for unit, multiplier in size_map.items():
                if v.upper().endswith(unit):
                    return int(v[:-2]) * multiplier
            return int(v)
        return v
    
    class Config:
        env_file = "config.env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings() 