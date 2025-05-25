"""
Utility Functions and Classes

Common utilities for the OpenFold platform including logging, configuration,
file handling, and computational utilities.
"""

from .logging import (
    setup_logging,
    get_logger,
    LogLevel,
    LogConfig
)

from .config import (
    ConfigManager,
    load_config,
    save_config,
    merge_configs
)

from .file_utils import (
    ensure_dir,
    safe_filename,
    get_file_hash,
    compress_file,
    decompress_file,
    FileManager
)

from .compute import (
    get_device_info,
    estimate_memory_usage,
    optimize_batch_size,
    ComputeManager
)

from .metrics import (
    calculate_rmsd,
    calculate_gdt_ts,
    calculate_tm_score,
    StructureMetrics
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger", 
    "LogLevel",
    "LogConfig",
    
    # Configuration
    "ConfigManager",
    "load_config",
    "save_config",
    "merge_configs",
    
    # File utilities
    "ensure_dir",
    "safe_filename",
    "get_file_hash",
    "compress_file",
    "decompress_file",
    "FileManager",
    
    # Compute utilities
    "get_device_info",
    "estimate_memory_usage",
    "optimize_batch_size",
    "ComputeManager",
    
    # Metrics
    "calculate_rmsd",
    "calculate_gdt_ts",
    "calculate_tm_score",
    "StructureMetrics"
] 