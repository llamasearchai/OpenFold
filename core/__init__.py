"""
OpenFold Core Module

Advanced biomolecule structure prediction and analysis platform.
"""

from .models.predictor import (
    AdvancedPredictor,
    PredictionConfig,
    PredictionResult,
    ModelType,
    PredictionMode,
    ConfidenceScores,
    StructureQuality,
    create_predictor,
    create_fast_predictor,
    create_accurate_predictor,
    create_ensemble_predictor
)

from .agents.structure_agent import (
    StructureAnalysisAgent,
    AnalysisType,
    AnalysisResult,
    create_structure_agent
)

__version__ = "2.0.0"
__author__ = "OpenFold Team"

__all__ = [
    # Predictor classes and functions
    "AdvancedPredictor",
    "PredictionConfig", 
    "PredictionResult",
    "ModelType",
    "PredictionMode",
    "ConfidenceScores",
    "StructureQuality",
    "create_predictor",
    "create_fast_predictor",
    "create_accurate_predictor",
    "create_ensemble_predictor",
    
    # Agent classes and functions
    "StructureAnalysisAgent",
    "AnalysisType",
    "AnalysisResult",
    "create_structure_agent"
] 