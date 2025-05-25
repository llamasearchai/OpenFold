"""
AI Agents API Router

Provides AI-powered analysis and insights for protein structures.
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from ..config import get_settings
from ..models.prediction import AnalysisRequest, AnalysisResponse
from ..services.prediction_service import PredictionService
from ..middleware.rate_limiter import rate_limit
from ..middleware.auth import get_current_user
from core.agents.structure_agent import StructureAnalysisAgent

router = APIRouter()
settings = get_settings()

# Initialize services
prediction_service = PredictionService()
ai_agent = StructureAnalysisAgent(
    openai_api_key=settings.openai_api_key
) if settings.openai_api_key else None


class StructureAnalysisRequest(BaseModel):
    """Request for AI-powered structure analysis"""
    job_id: str = Field(..., description="Job ID of completed prediction")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")
    include_binding_sites: bool = Field(default=True, description="Include binding site analysis")
    include_druggability: bool = Field(default=True, description="Include druggability assessment")
    research_context: Optional[str] = Field(None, description="Additional research context")


class ComparisonRequest(BaseModel):
    """Request for structure comparison"""
    job_id_1: str = Field(..., description="First structure job ID")
    job_id_2: str = Field(..., description="Second structure job ID")
    comparison_type: str = Field(default="structural", description="Type of comparison")


class HypothesisRequest(BaseModel):
    """Request for research hypothesis generation"""
    job_id: str = Field(..., description="Structure job ID")
    research_area: str = Field(..., description="Research area of interest")
    context: Optional[str] = Field(None, description="Additional context")


class ExperimentRequest(BaseModel):
    """Request for experiment suggestions"""
    job_id: str = Field(..., description="Structure job ID")
    research_goal: str = Field(..., description="Research goal")
    budget_level: str = Field(default="medium", description="Budget level (low/medium/high)")


@router.post("/analyze", response_model=AnalysisResponse)
@rate_limit(calls=5, period=300)  # 5 calls per 5 minutes
async def analyze_structure(
    request: StructureAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Perform AI-powered analysis of a predicted structure
    
    - **job_id**: ID of completed prediction job
    - **analysis_type**: Type of analysis (comprehensive, functional, druggability)
    - **include_binding_sites**: Whether to analyze binding sites
    - **include_druggability**: Whether to assess druggability
    - **research_context**: Additional context for analysis
    """
    try:
        if not ai_agent:
            raise HTTPException(
                status_code=503, 
                detail="AI analysis service not available. OpenAI API key required."
            )
        
        # Get prediction result
        result_data = prediction_service.redis_client.get(f"result:{request.job_id}")
        if not result_data:
            raise HTTPException(status_code=404, detail="Prediction result not found")
        
        import json
        result = json.loads(result_data)
        
        # Prepare structure data for analysis
        structure_data = {
            "sequence_length": len(result["sequence"]),
            "sequence": result["sequence"],
            "overall_confidence": result["confidence_scores"]["overall_confidence"],
            "low_confidence_regions": [
                i for i, conf in enumerate(result["confidence_scores"]["per_residue_confidence"])
                if conf < 0.7
            ],
            "model_type": result["model_type"],
            "quality_metrics": result["quality_metrics"]
        }
        
        analysis_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Perform comprehensive analysis
        logger.info(f"Starting AI analysis {analysis_id} for job {request.job_id}")
        
        analysis_results = {}
        
        # Core structure analysis
        if request.analysis_type in ["comprehensive", "functional"]:
            insights = await ai_agent.analyze_structure_prediction(structure_data)
            analysis_results["structure_insights"] = insights
        
        # Binding site analysis
        if request.include_binding_sites and result.get("binding_sites"):
            binding_analysis = await ai_agent.analyze_binding_sites(
                result["binding_sites"], 
                result["sequence"]
            )
            analysis_results["binding_site_analysis"] = binding_analysis
        
        # Druggability assessment
        if request.include_druggability:
            druggability = await ai_agent.assess_druggability(
                structure_data,
                result.get("binding_sites", [])
            )
            analysis_results["druggability_assessment"] = druggability
        
        # Research hypothesis generation if context provided
        if request.research_context:
            hypotheses = await ai_agent.generate_research_hypothesis(
                structure_data,
                request.research_context
            )
            analysis_results["research_hypotheses"] = hypotheses
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        response = AnalysisResponse(
            analysis_id=analysis_id,
            analysis_type=request.analysis_type,
            results=analysis_results,
            created_at=start_time,
            processing_time=processing_time
        )
        
        logger.success(f"Completed AI analysis {analysis_id} in {processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/compare", response_model=AnalysisResponse)
@rate_limit(calls=3, period=300)
async def compare_structures(
    request: ComparisonRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Compare two predicted structures using AI analysis
    
    - **job_id_1**: First structure job ID
    - **job_id_2**: Second structure job ID
    - **comparison_type**: Type of comparison to perform
    """
    try:
        if not ai_agent:
            raise HTTPException(
                status_code=503, 
                detail="AI analysis service not available"
            )
        
        # Get both prediction results
        import json
        
        result_data_1 = prediction_service.redis_client.get(f"result:{request.job_id_1}")
        result_data_2 = prediction_service.redis_client.get(f"result:{request.job_id_2}")
        
        if not result_data_1 or not result_data_2:
            raise HTTPException(status_code=404, detail="One or both prediction results not found")
        
        result_1 = json.loads(result_data_1)
        result_2 = json.loads(result_data_2)
        
        # Prepare structure data
        structure_1 = {
            "sequence_length": len(result_1["sequence"]),
            "sequence": result_1["sequence"],
            "overall_confidence": result_1["confidence_scores"]["overall_confidence"],
            "model_type": result_1["model_type"]
        }
        
        structure_2 = {
            "sequence_length": len(result_2["sequence"]),
            "sequence": result_2["sequence"],
            "overall_confidence": result_2["confidence_scores"]["overall_confidence"],
            "model_type": result_2["model_type"]
        }
        
        analysis_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Perform comparison
        comparison_results = await ai_agent.compare_structures(structure_1, structure_2)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        response = AnalysisResponse(
            analysis_id=analysis_id,
            analysis_type="comparison",
            results={"comparison": comparison_results},
            created_at=start_time,
            processing_time=processing_time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Structure comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.post("/hypotheses", response_model=Dict[str, Any])
@rate_limit(calls=3, period=600)  # 3 calls per 10 minutes
async def generate_hypotheses(
    request: HypothesisRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Generate research hypotheses based on structure analysis
    
    - **job_id**: Structure job ID
    - **research_area**: Area of research interest
    - **context**: Additional research context
    """
    try:
        if not ai_agent:
            raise HTTPException(
                status_code=503, 
                detail="AI analysis service not available"
            )
        
        # Get prediction result
        import json
        result_data = prediction_service.redis_client.get(f"result:{request.job_id}")
        if not result_data:
            raise HTTPException(status_code=404, detail="Prediction result not found")
        
        result = json.loads(result_data)
        
        structure_data = {
            "sequence_length": len(result["sequence"]),
            "sequence": result["sequence"],
            "overall_confidence": result["confidence_scores"]["overall_confidence"],
            "research_area": request.research_area,
            "binding_sites": result.get("binding_sites", []),
            "quality_metrics": result["quality_metrics"]
        }
        
        # Generate hypotheses
        hypotheses = await ai_agent.generate_research_hypothesis(
            structure_data,
            request.context or ""
        )
        
        return {
            "job_id": request.job_id,
            "research_area": request.research_area,
            "hypotheses": hypotheses,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hypothesis generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hypothesis generation failed: {str(e)}")


@router.post("/experiments", response_model=Dict[str, Any])
@rate_limit(calls=3, period=600)
async def suggest_experiments(
    request: ExperimentRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Suggest experimental approaches based on structure analysis
    
    - **job_id**: Structure job ID
    - **research_goal**: Research goal or objective
    - **budget_level**: Budget level for experiments
    """
    try:
        if not ai_agent:
            raise HTTPException(
                status_code=503, 
                detail="AI analysis service not available"
            )
        
        # Get prediction result
        import json
        result_data = prediction_service.redis_client.get(f"result:{request.job_id}")
        if not result_data:
            raise HTTPException(status_code=404, detail="Prediction result not found")
        
        result = json.loads(result_data)
        
        structure_data = {
            "sequence_length": len(result["sequence"]),
            "sequence": result["sequence"],
            "overall_confidence": result["confidence_scores"]["overall_confidence"],
            "binding_sites": result.get("binding_sites", []),
            "quality_metrics": result["quality_metrics"],
            "budget_level": request.budget_level
        }
        
        # Generate experiment suggestions
        experiments = await ai_agent.suggest_experiments(
            structure_data,
            request.research_goal
        )
        
        return {
            "job_id": request.job_id,
            "research_goal": request.research_goal,
            "budget_level": request.budget_level,
            "experiments": experiments,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Experiment suggestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Experiment suggestion failed: {str(e)}")


@router.post("/abstract", response_model=Dict[str, str])
@rate_limit(calls=2, period=600)
async def generate_abstract(
    analysis_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Generate a scientific abstract based on analysis results
    
    - **analysis_id**: ID of completed analysis
    """
    try:
        if not ai_agent:
            raise HTTPException(
                status_code=503, 
                detail="AI analysis service not available"
            )
        
        # This would typically retrieve stored analysis results
        # For now, we'll use a placeholder
        analysis_results = {
            "analysis_id": analysis_id,
            "placeholder": "This would contain actual analysis results"
        }
        
        # Generate abstract
        abstract = await ai_agent.generate_publication_abstract(analysis_results)
        
        return {
            "analysis_id": analysis_id,
            "abstract": abstract,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Abstract generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Abstract generation failed: {str(e)}")


@router.get("/capabilities", response_model=Dict[str, Any])
async def get_ai_capabilities():
    """
    Get information about available AI analysis capabilities
    """
    try:
        capabilities = {
            "ai_agent_available": ai_agent is not None,
            "analysis_types": [
                "comprehensive",
                "functional",
                "druggability",
                "comparison"
            ],
            "features": {
                "structure_analysis": True,
                "binding_site_analysis": True,
                "druggability_assessment": True,
                "structure_comparison": True,
                "hypothesis_generation": True,
                "experiment_suggestions": True,
                "abstract_generation": True
            },
            "rate_limits": {
                "analysis": "5 calls per 5 minutes",
                "comparison": "3 calls per 5 minutes",
                "hypotheses": "3 calls per 10 minutes",
                "experiments": "3 calls per 10 minutes",
                "abstract": "2 calls per 10 minutes"
            },
            "requirements": {
                "openai_api_key": "Required for AI features",
                "completed_prediction": "Structure prediction must be completed"
            }
        }
        
        return capabilities
        
    except Exception as e:
        logger.error(f"Failed to get AI capabilities: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve capabilities")


@router.get("/models", response_model=Dict[str, Any])
async def get_ai_models():
    """
    Get information about AI models used for analysis
    """
    try:
        models_info = {
            "primary_model": settings.openai_model if ai_agent else None,
            "model_capabilities": {
                "structure_analysis": "GPT-4 Turbo with specialized prompts",
                "binding_site_analysis": "Medicinal chemistry expertise",
                "druggability_assessment": "Drug discovery knowledge",
                "hypothesis_generation": "Research methodology expertise",
                "experiment_design": "Experimental biology knowledge"
            },
            "model_parameters": {
                "temperature": 0.1,
                "max_tokens": 4096,
                "model_version": settings.openai_model
            } if ai_agent else None,
            "specialized_prompts": {
                "structure_analysis": "Expert computational biologist persona",
                "drug_discovery": "Medicinal chemist persona",
                "research_design": "Experimental biologist persona",
                "scientific_writing": "Scientific writer persona"
            }
        }
        
        return models_info
        
    except Exception as e:
        logger.error(f"Failed to get AI models info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models info") 