"""
Prediction Service

Core service for handling biomolecule structure predictions with AI integration.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import torch
from loguru import logger
import redis
from Bio import SeqIO
from io import StringIO

from ..config import get_settings
from ..models.prediction import (
    PredictionRequest, 
    PredictionResponse, 
    StructureData,
    ConfidenceMetrics,
    StructureQuality,
    SecondaryStructure,
    BindingSite,
    ModelInfo
)
from core.models.predictor import OpenFoldPredictor
from core.agents.structure_agent import StructureAnalysisAgent
from data.preprocessing.processor import BioMoleculeProcessor


class PredictionService:
    """Service for handling structure prediction requests"""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = redis.from_url(self.settings.redis_url)
        self.predictor = OpenFoldPredictor(self._get_predictor_config())
        self.processor = BioMoleculeProcessor(self._get_processor_config())
        self.ai_agent = StructureAnalysisAgent(
            openai_api_key=self.settings.openai_api_key
        ) if self.settings.openai_api_key else None
        
        # Model registry
        self.available_models = {
            "alphafold3": {
                "name": "AlphaFold 3",
                "description": "Latest AlphaFold model with improved accuracy",
                "version": "3.0.0",
                "max_sequence_length": 2048,
                "supports_templates": True,
                "supports_msa": True,
                "gpu_required": True,
                "memory_requirement": "16GB",
                "accuracy_metrics": {"gdt_ts": 92.4, "rmsd": 1.2},
                "estimated_runtime": {"100": 2.3, "500": 8.1, "1000": 24.5}
            },
            "esm2": {
                "name": "ESM-2",
                "description": "Meta's evolutionary scale modeling",
                "version": "2.0.0",
                "max_sequence_length": 1024,
                "supports_templates": False,
                "supports_msa": False,
                "gpu_required": True,
                "memory_requirement": "8GB",
                "accuracy_metrics": {"gdt_ts": 87.2, "rmsd": 1.8},
                "estimated_runtime": {"100": 1.5, "500": 4.2, "1000": 12.1}
            },
            "openfold": {
                "name": "OpenFold",
                "description": "Open source AlphaFold implementation",
                "version": "1.0.1",
                "max_sequence_length": 1536,
                "supports_templates": True,
                "supports_msa": True,
                "gpu_required": True,
                "memory_requirement": "12GB",
                "accuracy_metrics": {"gdt_ts": 89.1, "rmsd": 1.5},
                "estimated_runtime": {"100": 3.1, "500": 9.8, "1000": 28.2}
            },
            "colabfold": {
                "name": "ColabFold",
                "description": "Fast protein folding using MMseqs2",
                "version": "1.5.5",
                "max_sequence_length": 1000,
                "supports_templates": True,
                "supports_msa": True,
                "gpu_required": False,
                "memory_requirement": "4GB",
                "accuracy_metrics": {"gdt_ts": 85.7, "rmsd": 2.1},
                "estimated_runtime": {"100": 0.8, "500": 2.1, "1000": 6.5}
            }
        }
        
        logger.info("PredictionService initialized with AI agent support")
    
    def _get_predictor_config(self) -> Dict[str, Any]:
        """Get configuration for the predictor"""
        return {
            "model_cache_dir": self.settings.model_cache_dir,
            "use_gpu": self.settings.use_gpu,
            "gpu_memory_fraction": self.settings.gpu_memory_fraction,
            "batch_size": self.settings.batch_size,
            "max_sequence_length": self.settings.max_sequence_length
        }
    
    def _get_processor_config(self) -> Dict[str, Any]:
        """Get configuration for the data processor"""
        return {
            "temp_dir": self.settings.temp_dir,
            "max_file_size": self.settings.max_file_size
        }
    
    async def process_prediction_job(self, job_id: str, request: Any) -> None:
        """Process a single prediction job"""
        try:
            logger.info(f"Starting prediction job {job_id}")
            start_time = time.time()
            
            # Update job status to running
            await self._update_job_status(job_id, "running", progress=10)
            
            # Validate sequence
            if len(request.sequence) > self.settings.max_sequence_length:
                raise ValueError(f"Sequence too long: {len(request.sequence)} > {self.settings.max_sequence_length}")
            
            # Process sequence data
            await self._update_job_status(job_id, "running", progress=20)
            processed_data = self.processor.process_sequence(request.sequence)
            
            # Load and configure model
            await self._update_job_status(job_id, "running", progress=30)
            model_config = self.available_models.get(request.model_type, self.available_models["alphafold3"])
            
            # Perform structure prediction
            await self._update_job_status(job_id, "running", progress=50)
            prediction_result = await self._predict_structure(
                request.sequence, 
                request.model_type,
                processed_data,
                request
            )
            
            # Post-process results
            await self._update_job_status(job_id, "running", progress=70)
            
            # Generate confidence scores
            confidence_scores = await self._calculate_confidence_scores(
                prediction_result, 
                request.sequence
            )
            
            # Assess structure quality
            quality_metrics = await self._assess_structure_quality(prediction_result)
            
            # Predict secondary structure
            secondary_structure = await self._predict_secondary_structure(
                prediction_result, 
                request.sequence
            )
            
            # Predict binding sites if requested
            binding_sites = None
            if request.predict_binding_sites:
                binding_sites = await self._predict_binding_sites(
                    prediction_result, 
                    request.sequence
                )
            
            # Generate PDB string
            pdb_string = await self._generate_pdb_string(prediction_result, request.sequence)
            
            # AI-powered analysis if available
            ai_insights = None
            if self.ai_agent and request.model_type == "alphafold3":
                await self._update_job_status(job_id, "running", progress=85)
                ai_insights = await self._generate_ai_insights(
                    prediction_result, 
                    request.sequence,
                    confidence_scores
                )
            
            # Structure optimization if requested
            if request.optimize_structure:
                await self._update_job_status(job_id, "running", progress=90)
                prediction_result = await self._optimize_structure(prediction_result)
            
            # Prepare final result
            processing_time = time.time() - start_time
            
            result = {
                "job_id": job_id,
                "sequence": request.sequence,
                "model_type": request.model_type,
                "structure": prediction_result,
                "confidence_scores": confidence_scores,
                "quality_metrics": quality_metrics,
                "secondary_structure": secondary_structure,
                "binding_sites": binding_sites,
                "pdb_string": pdb_string,
                "processing_time": processing_time,
                "model_version": model_config["version"],
                "completed_at": datetime.utcnow().isoformat(),
                "ai_insights": ai_insights
            }
            
            # Store result
            self.redis_client.setex(
                f"result:{job_id}",
                3600 * 24 * 7,  # 7 days TTL
                json.dumps(result, default=str)
            )
            
            # Update job status to completed
            await self._update_job_status(job_id, "completed", progress=100)
            
            # Send notification if email provided
            if hasattr(request, 'email_notification') and request.email_notification:
                await self._send_completion_notification(job_id, request.email_notification)
            
            logger.success(f"Completed prediction job {job_id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to process prediction job {job_id}: {str(e)}")
            await self._update_job_status(
                job_id, 
                "failed", 
                error_message=str(e)
            )
            raise
    
    async def process_batch_prediction(
        self, 
        batch_id: str, 
        job_ids: List[str], 
        request: Any
    ) -> None:
        """Process a batch of prediction jobs"""
        try:
            logger.info(f"Starting batch prediction {batch_id} with {len(job_ids)} jobs")
            
            # Process jobs in parallel with limited concurrency
            semaphore = asyncio.Semaphore(request.parallel_jobs)
            
            async def process_single_job(job_id: str, sequence: str):
                async with semaphore:
                    # Create individual request
                    job_request = type('Request', (), {
                        'sequence': sequence,
                        'model_type': request.model_type,
                        'confidence_threshold': request.confidence_threshold,
                        'include_templates': True,
                        'optimize_structure': False,
                        'predict_binding_sites': False,
                        'generate_variants': False,
                        'email_notification': None
                    })()
                    
                    await self.process_prediction_job(job_id, job_request)
            
            # Get sequences for each job
            tasks = []
            for i, job_id in enumerate(job_ids):
                sequence = request.sequences[i]
                task = process_single_job(job_id, sequence)
                tasks.append(task)
            
            # Execute all jobs
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update batch status
            await self._update_batch_status(batch_id)
            
            logger.success(f"Completed batch prediction {batch_id}")
            
        except Exception as e:
            logger.error(f"Failed to process batch prediction {batch_id}: {str(e)}")
            raise
    
    async def _predict_structure(
        self, 
        sequence: str, 
        model_type: str,
        processed_data: Dict[str, Any],
        request: Any
    ) -> Dict[str, Any]:
        """Perform the actual structure prediction"""
        try:
            # Mock prediction for demonstration
            # In production, this would call the actual ML models
            
            seq_len = len(sequence)
            
            # Generate mock coordinates (alpha carbons in a helix)
            coordinates = []
            for i in range(seq_len):
                # Simple helix coordinates
                angle = i * 100 * np.pi / 180  # 100 degrees per residue
                x = 1.5 * np.cos(angle)
                y = 1.5 * np.sin(angle)
                z = i * 1.5  # 1.5 Å rise per residue
                coordinates.append([x, y, z])
            
            # Generate atom names, residue info
            atom_names = ["CA"] * seq_len
            residue_names = [self._aa_three_letter(aa) for aa in sequence]
            residue_numbers = list(range(1, seq_len + 1))
            chain_ids = ["A"] * seq_len
            
            # Generate B-factors (confidence-based)
            b_factors = np.random.uniform(20, 80, seq_len).tolist()
            occupancies = [1.0] * seq_len
            
            structure_data = {
                "coordinates": coordinates,
                "atom_names": atom_names,
                "residue_names": residue_names,
                "residue_numbers": residue_numbers,
                "chain_ids": chain_ids,
                "b_factors": b_factors,
                "occupancies": occupancies
            }
            
            return structure_data
            
        except Exception as e:
            logger.error(f"Structure prediction failed: {str(e)}")
            raise
    
    async def _calculate_confidence_scores(
        self, 
        structure: Dict[str, Any], 
        sequence: str
    ) -> Dict[str, Any]:
        """Calculate confidence scores for the prediction"""
        try:
            seq_len = len(sequence)
            
            # Generate mock confidence scores
            per_residue_confidence = np.random.uniform(0.6, 0.95, seq_len).tolist()
            overall_confidence = np.mean(per_residue_confidence)
            
            # Domain confidence (mock)
            domain_confidence = {
                "N_terminal": np.random.uniform(0.7, 0.9),
                "C_terminal": np.random.uniform(0.6, 0.85),
                "core": np.random.uniform(0.8, 0.95)
            }
            
            # Secondary structure confidence
            ss_confidence = {
                "helix": np.random.uniform(0.8, 0.95),
                "sheet": np.random.uniform(0.7, 0.9),
                "loop": np.random.uniform(0.5, 0.8)
            }
            
            return {
                "overall_confidence": overall_confidence,
                "per_residue_confidence": per_residue_confidence,
                "domain_confidence": domain_confidence,
                "secondary_structure_confidence": ss_confidence
            }
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            raise
    
    async def _assess_structure_quality(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of the predicted structure"""
        try:
            # Mock quality metrics
            quality_metrics = {
                "ramachandran_favored": np.random.uniform(85, 98),
                "ramachandran_outliers": np.random.uniform(0, 5),
                "clash_score": np.random.uniform(0, 10),
                "molprobity_score": np.random.uniform(1.0, 2.5),
                "resolution_estimate": np.random.uniform(1.5, 3.0)
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            raise
    
    async def _predict_secondary_structure(
        self, 
        structure: Dict[str, Any], 
        sequence: str
    ) -> Dict[str, Any]:
        """Predict secondary structure elements"""
        try:
            seq_len = len(sequence)
            
            # Mock secondary structure prediction
            helix_regions = [
                {"start": 10, "end": 25},
                {"start": 40, "end": 55}
            ]
            
            sheet_regions = [
                {"start": 30, "end": 35},
                {"start": 60, "end": 65}
            ]
            
            loop_regions = [
                {"start": 1, "end": 9},
                {"start": 26, "end": 29},
                {"start": 36, "end": 39},
                {"start": 56, "end": 59},
                {"start": 66, "end": seq_len}
            ]
            
            # Generate DSSP-like assignment
            dssp_assignment = "".join([
                "H" if any(r["start"] <= i+1 <= r["end"] for r in helix_regions) else
                "E" if any(r["start"] <= i+1 <= r["end"] for r in sheet_regions) else
                "L" for i in range(seq_len)
            ])
            
            return {
                "helix_regions": helix_regions,
                "sheet_regions": sheet_regions,
                "loop_regions": loop_regions,
                "dssp_assignment": dssp_assignment
            }
            
        except Exception as e:
            logger.error(f"Secondary structure prediction failed: {str(e)}")
            raise
    
    async def _predict_binding_sites(
        self, 
        structure: Dict[str, Any], 
        sequence: str
    ) -> List[Dict[str, Any]]:
        """Predict potential binding sites"""
        try:
            # Mock binding site prediction
            binding_sites = [
                {
                    "site_id": "BS1",
                    "residues": [15, 16, 17, 20, 21],
                    "confidence": 0.85,
                    "pocket_volume": 245.6,
                    "druggability_score": 0.72
                },
                {
                    "site_id": "BS2",
                    "residues": [45, 46, 49, 50, 53],
                    "confidence": 0.78,
                    "pocket_volume": 189.3,
                    "druggability_score": 0.65
                }
            ]
            
            return binding_sites
            
        except Exception as e:
            logger.error(f"Binding site prediction failed: {str(e)}")
            raise
    
    async def _generate_pdb_string(
        self, 
        structure: Dict[str, Any], 
        sequence: str
    ) -> str:
        """Generate PDB format string from structure data"""
        try:
            pdb_lines = []
            
            # Header
            pdb_lines.append("HEADER    PROTEIN STRUCTURE PREDICTION")
            pdb_lines.append("TITLE     OPENFOLD PREDICTION")
            pdb_lines.append("MODEL        1")
            
            # Atoms
            for i, (coord, atom_name, res_name, res_num, chain_id, b_factor, occupancy) in enumerate(zip(
                structure["coordinates"],
                structure["atom_names"],
                structure["residue_names"],
                structure["residue_numbers"],
                structure["chain_ids"],
                structure["b_factors"],
                structure["occupancies"]
            )):
                pdb_line = (
                    f"ATOM  {i+1:5d}  {atom_name:<4s}{res_name:>3s} {chain_id}{res_num:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{occupancy:6.2f}{b_factor:6.2f}"
                    f"           C"
                )
                pdb_lines.append(pdb_line)
            
            pdb_lines.append("ENDMDL")
            pdb_lines.append("END")
            
            return "\n".join(pdb_lines)
            
        except Exception as e:
            logger.error(f"PDB generation failed: {str(e)}")
            raise
    
    async def _generate_ai_insights(
        self, 
        structure: Dict[str, Any], 
        sequence: str,
        confidence_scores: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate AI-powered insights about the structure"""
        try:
            if not self.ai_agent:
                return None
            
            # Prepare structure summary for AI analysis
            structure_summary = {
                "sequence_length": len(sequence),
                "overall_confidence": confidence_scores["overall_confidence"],
                "low_confidence_regions": [
                    i for i, conf in enumerate(confidence_scores["per_residue_confidence"])
                    if conf < 0.7
                ],
                "sequence": sequence[:100] + "..." if len(sequence) > 100 else sequence
            }
            
            # Get AI insights
            insights = await self.ai_agent.analyze_structure_prediction(structure_summary)
            
            return {
                "functional_analysis": insights.get("functional_analysis"),
                "structural_features": insights.get("structural_features"),
                "confidence_interpretation": insights.get("confidence_interpretation"),
                "recommendations": insights.get("recommendations")
            }
            
        except Exception as e:
            logger.warning(f"AI insights generation failed: {str(e)}")
            return None
    
    async def _optimize_structure(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize the predicted structure using physics-based methods"""
        try:
            # Mock structure optimization
            # In production, this would use molecular dynamics or energy minimization
            
            # Slightly adjust coordinates to simulate optimization
            optimized_coords = []
            for coord in structure["coordinates"]:
                noise = np.random.normal(0, 0.1, 3)  # Small random adjustment
                optimized_coord = [c + n for c, n in zip(coord, noise)]
                optimized_coords.append(optimized_coord)
            
            structure["coordinates"] = optimized_coords
            
            # Update B-factors to reflect optimization
            structure["b_factors"] = [b * 0.9 for b in structure["b_factors"]]
            
            return structure
            
        except Exception as e:
            logger.error(f"Structure optimization failed: {str(e)}")
            raise
    
    async def _update_job_status(
        self, 
        job_id: str, 
        status: str, 
        progress: int = None,
        error_message: str = None
    ) -> None:
        """Update job status in Redis"""
        try:
            job_data = self.redis_client.get(f"job:{job_id}")
            if job_data:
                job_info = json.loads(job_data)
                job_info["status"] = status
                job_info["updated_at"] = datetime.utcnow().isoformat()
                
                if progress is not None:
                    job_info["progress"] = progress
                
                if error_message:
                    job_info["error_message"] = error_message
                
                self.redis_client.setex(
                    f"job:{job_id}",
                    3600 * 24,
                    json.dumps(job_info)
                )
                
        except Exception as e:
            logger.error(f"Failed to update job status: {str(e)}")
    
    async def _update_batch_status(self, batch_id: str) -> None:
        """Update batch status based on individual job statuses"""
        try:
            batch_data = self.redis_client.get(f"batch:{batch_id}")
            if not batch_data:
                return
            
            batch_info = json.loads(batch_data)
            job_ids = batch_info["job_ids"]
            
            # Count job statuses
            completed = 0
            failed = 0
            running = 0
            queued = 0
            
            for job_id in job_ids:
                job_data = self.redis_client.get(f"job:{job_id}")
                if job_data:
                    job_info = json.loads(job_data)
                    status = job_info["status"]
                    
                    if status == "completed":
                        completed += 1
                    elif status == "failed":
                        failed += 1
                    elif status == "running":
                        running += 1
                    elif status == "queued":
                        queued += 1
            
            # Update batch info
            batch_info["completed_jobs"] = completed
            batch_info["failed_jobs"] = failed
            batch_info["running_jobs"] = running
            batch_info["queued_jobs"] = queued
            batch_info["progress"] = (completed / len(job_ids)) * 100
            
            if completed == len(job_ids):
                batch_info["status"] = "completed"
            elif failed > 0 and (completed + failed) == len(job_ids):
                batch_info["status"] = "completed_with_errors"
            else:
                batch_info["status"] = "running"
            
            self.redis_client.setex(
                f"batch:{batch_id}",
                3600 * 24,
                json.dumps(batch_info)
            )
            
        except Exception as e:
            logger.error(f"Failed to update batch status: {str(e)}")
    
    async def _send_completion_notification(self, job_id: str, email: str) -> None:
        """Send email notification when job completes"""
        try:
            # Mock email notification
            logger.info(f"Sending completion notification for job {job_id} to {email}")
            # In production, integrate with email service
            
        except Exception as e:
            logger.error(f"Failed to send notification: {str(e)}")
    
    def parse_fasta_content(self, content: str) -> List[str]:
        """Parse FASTA content and extract sequences"""
        try:
            sequences = []
            fasta_io = StringIO(content)
            
            for record in SeqIO.parse(fasta_io, "fasta"):
                sequences.append(str(record.seq))
            
            return sequences
            
        except Exception as e:
            logger.error(f"Failed to parse FASTA content: {str(e)}")
            return []
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available prediction models"""
        try:
            models = []
            for model_id, model_info in self.available_models.items():
                model_data = {
                    "model_id": model_id,
                    **model_info
                }
                models.append(model_data)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            return []
    
    async def generate_structure_file(self, job_id: str, format: str) -> str:
        """Generate structure file in specified format"""
        try:
            # Get result data
            result_data = self.redis_client.get(f"result:{job_id}")
            if not result_data:
                raise ValueError("Result not found")
            
            result = json.loads(result_data)
            
            # Generate file based on format
            file_path = f"./tmp/{job_id}.{format}"
            
            if format == "pdb":
                with open(file_path, "w") as f:
                    f.write(result["pdb_string"])
            elif format == "cif":
                # Convert to mmCIF format (mock implementation)
                cif_content = self._convert_to_cif(result)
                with open(file_path, "w") as f:
                    f.write(cif_content)
            elif format == "sdf":
                # Convert to SDF format (mock implementation)
                sdf_content = self._convert_to_sdf(result)
                with open(file_path, "w") as f:
                    f.write(sdf_content)
            
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to generate structure file: {str(e)}")
            raise
    
    def _convert_to_cif(self, result: Dict[str, Any]) -> str:
        """Convert structure to mmCIF format"""
        # Mock implementation
        return "# Mock mmCIF file\ndata_structure\n"
    
    def _convert_to_sdf(self, result: Dict[str, Any]) -> str:
        """Convert structure to SDF format"""
        # Mock implementation
        return "Mock SDF file\n$$$$\n"
    
    def _aa_three_letter(self, aa: str) -> str:
        """Convert single letter amino acid to three letter code"""
        aa_map = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }
        return aa_map.get(aa.upper(), 'UNK') 