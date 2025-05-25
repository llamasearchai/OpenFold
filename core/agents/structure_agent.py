"""
Structure Analysis AI Agent

OpenAI-powered agent for intelligent protein structure analysis and insights.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class StructureInsights(BaseModel):
    """Structured output for structure analysis insights"""
    functional_analysis: str = Field(description="Functional analysis of the protein")
    structural_features: List[str] = Field(description="Key structural features identified")
    confidence_interpretation: str = Field(description="Interpretation of confidence scores")
    recommendations: List[str] = Field(description="Recommendations for further analysis")
    potential_functions: List[str] = Field(description="Potential biological functions")
    drug_targets: Optional[List[str]] = Field(description="Potential drug target sites")
    evolutionary_insights: Optional[str] = Field(description="Evolutionary context")


class BindingSiteAnalysis(BaseModel):
    """Analysis of binding sites"""
    site_description: str = Field(description="Description of the binding site")
    druggability_assessment: str = Field(description="Assessment of druggability")
    potential_ligands: List[str] = Field(description="Potential binding ligands")
    therapeutic_relevance: str = Field(description="Therapeutic relevance")


class StructureComparison(BaseModel):
    """Comparison between structures"""
    similarity_analysis: str = Field(description="Analysis of structural similarities")
    key_differences: List[str] = Field(description="Key structural differences")
    functional_implications: str = Field(description="Functional implications of differences")


class StructureAnalysisAgent:
    """AI agent for comprehensive protein structure analysis"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4-turbo-preview"):
        """Initialize the structure analysis agent"""
        self.openai_api_key = openai_api_key
        self.model = model
        
        # Initialize OpenAI client
        openai.api_key = openai_api_key
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model,
            temperature=0.1,
            max_tokens=4096
        )
        
        # Output parsers
        self.insights_parser = PydanticOutputParser(pydantic_object=StructureInsights)
        self.binding_parser = PydanticOutputParser(pydantic_object=BindingSiteAnalysis)
        self.comparison_parser = PydanticOutputParser(pydantic_object=StructureComparison)
        
        logger.info(f"StructureAnalysisAgent initialized with model {model}")
    
    async def analyze_structure_prediction(self, structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a predicted protein structure and provide comprehensive insights
        """
        try:
            logger.info("Starting AI-powered structure analysis")
            
            # Prepare the analysis prompt
            system_prompt = """You are an expert computational biologist and structural biologist with deep knowledge of protein structure, function, and drug discovery. 

Your task is to analyze protein structure prediction data and provide comprehensive insights that would be valuable for researchers, pharmaceutical companies, and biotechnology organizations.

Focus on:
1. Functional analysis based on sequence and structural features
2. Identification of key structural elements
3. Interpretation of confidence scores and their implications
4. Recommendations for experimental validation
5. Potential biological functions and pathways
6. Drug discovery opportunities
7. Evolutionary context when relevant

Provide scientifically accurate, detailed, and actionable insights."""

            human_prompt = f"""
Please analyze this protein structure prediction data:

Sequence Length: {structure_data.get('sequence_length', 'Unknown')}
Overall Confidence: {structure_data.get('overall_confidence', 'Unknown'):.3f}
Low Confidence Regions: {len(structure_data.get('low_confidence_regions', []))} regions
Sequence (first 100 residues): {structure_data.get('sequence', 'Not provided')}

Based on this information, provide a comprehensive analysis including:
1. Functional predictions based on sequence analysis
2. Structural feature identification
3. Confidence score interpretation
4. Recommendations for further analysis
5. Potential biological functions
6. Drug target assessment
7. Evolutionary insights if applicable

{self.insights_parser.get_format_instructions()}
"""

            # Create the prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            # Generate the analysis
            response = await self._call_llm_async(prompt, structure_data)
            
            # Parse the response
            try:
                insights = self.insights_parser.parse(response.content)
                return insights.dict()
            except Exception as parse_error:
                logger.warning(f"Failed to parse structured output: {parse_error}")
                # Fallback to raw response
                return {"raw_analysis": response.content}
                
        except Exception as e:
            logger.error(f"Structure analysis failed: {str(e)}")
            return {"error": str(e)}
    
    async def analyze_binding_sites(self, binding_sites: List[Dict[str, Any]], sequence: str) -> List[Dict[str, Any]]:
        """
        Analyze predicted binding sites for drug discovery potential
        """
        try:
            logger.info(f"Analyzing {len(binding_sites)} binding sites")
            
            analyses = []
            
            for site in binding_sites:
                system_prompt = """You are a medicinal chemist and drug discovery expert specializing in binding site analysis and druggability assessment.

Analyze the provided binding site data and provide insights for drug discovery, including druggability assessment, potential ligand types, and therapeutic relevance."""

                human_prompt = f"""
Analyze this binding site for drug discovery potential:

Site ID: {site.get('site_id')}
Residues: {site.get('residues')}
Confidence: {site.get('confidence', 0):.3f}
Pocket Volume: {site.get('pocket_volume', 'Unknown')} Ų
Druggability Score: {site.get('druggability_score', 'Unknown')}

Surrounding sequence context: {self._get_sequence_context(sequence, site.get('residues', []))}

Provide a detailed analysis including:
1. Druggability assessment
2. Potential ligand types
3. Therapeutic relevance
4. Design recommendations

{self.binding_parser.get_format_instructions()}
"""

                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", human_prompt)
                ])
                
                response = await self._call_llm_async(prompt, site)
                
                try:
                    analysis = self.binding_parser.parse(response.content)
                    analyses.append({
                        "site_id": site.get('site_id'),
                        "analysis": analysis.dict()
                    })
                except Exception as parse_error:
                    logger.warning(f"Failed to parse binding site analysis: {parse_error}")
                    analyses.append({
                        "site_id": site.get('site_id'),
                        "analysis": {"raw_analysis": response.content}
                    })
            
            return analyses
            
        except Exception as e:
            logger.error(f"Binding site analysis failed: {str(e)}")
            return []
    
    async def compare_structures(self, structure1: Dict[str, Any], structure2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two protein structures and analyze differences
        """
        try:
            logger.info("Comparing protein structures")
            
            system_prompt = """You are a structural biologist expert in protein structure comparison and analysis.

Compare the provided protein structures and analyze their similarities, differences, and functional implications. Focus on structural features, evolutionary relationships, and functional consequences of structural differences."""

            human_prompt = f"""
Compare these two protein structures:

Structure 1:
- Sequence Length: {structure1.get('sequence_length', 'Unknown')}
- Confidence: {structure1.get('overall_confidence', 'Unknown')}
- Sequence: {structure1.get('sequence', 'Not provided')[:100]}...

Structure 2:
- Sequence Length: {structure2.get('sequence_length', 'Unknown')}
- Confidence: {structure2.get('overall_confidence', 'Unknown')}
- Sequence: {structure2.get('sequence', 'Not provided')[:100]}...

Provide a comprehensive comparison including:
1. Structural similarity analysis
2. Key differences
3. Functional implications
4. Evolutionary insights

{self.comparison_parser.get_format_instructions()}
"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            response = await self._call_llm_async(prompt, {"struct1": structure1, "struct2": structure2})
            
            try:
                comparison = self.comparison_parser.parse(response.content)
                return comparison.dict()
            except Exception as parse_error:
                logger.warning(f"Failed to parse comparison output: {parse_error}")
                return {"raw_comparison": response.content}
                
        except Exception as e:
            logger.error(f"Structure comparison failed: {str(e)}")
            return {"error": str(e)}
    
    async def generate_research_hypothesis(self, structure_data: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """
        Generate research hypotheses based on structure analysis
        """
        try:
            logger.info("Generating research hypotheses")
            
            system_prompt = """You are a leading researcher in structural biology and biochemistry with expertise in generating novel research hypotheses.

Based on protein structure data, generate innovative and testable research hypotheses that could lead to significant scientific discoveries or therapeutic applications."""

            human_prompt = f"""
Based on this protein structure data, generate research hypotheses:

{json.dumps(structure_data, indent=2)}

Additional Context: {context}

Generate 3-5 innovative research hypotheses that are:
1. Scientifically sound and testable
2. Potentially impactful for the field
3. Based on structural insights
4. Relevant for drug discovery or basic research

For each hypothesis, provide:
- The hypothesis statement
- Scientific rationale
- Experimental approaches to test it
- Potential impact
"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = await self._call_llm_async_messages(messages)
            
            return {"hypotheses": response.content}
            
        except Exception as e:
            logger.error(f"Hypothesis generation failed: {str(e)}")
            return {"error": str(e)}
    
    async def suggest_experiments(self, structure_data: Dict[str, Any], research_goal: str) -> Dict[str, Any]:
        """
        Suggest experimental approaches based on structure analysis
        """
        try:
            logger.info(f"Suggesting experiments for goal: {research_goal}")
            
            system_prompt = """You are an experimental biologist and biochemist with extensive experience in protein characterization, drug discovery, and structural biology experiments.

Suggest specific, practical experimental approaches based on protein structure predictions and research goals."""

            human_prompt = f"""
Based on this protein structure data and research goal, suggest experimental approaches:

Structure Data:
{json.dumps(structure_data, indent=2)}

Research Goal: {research_goal}

Suggest 5-7 specific experiments including:
1. Structural validation experiments
2. Functional characterization assays
3. Drug screening approaches (if relevant)
4. Biophysical characterization methods
5. Cellular/in vivo studies

For each experiment, provide:
- Experimental method
- Expected outcomes
- Required resources
- Timeline estimate
- Potential challenges
"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = await self._call_llm_async_messages(messages)
            
            return {"experimental_suggestions": response.content}
            
        except Exception as e:
            logger.error(f"Experiment suggestion failed: {str(e)}")
            return {"error": str(e)}
    
    async def assess_druggability(self, structure_data: Dict[str, Any], binding_sites: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess overall druggability of the protein target
        """
        try:
            logger.info("Assessing protein druggability")
            
            system_prompt = """You are a medicinal chemist and drug discovery expert specializing in target assessment and druggability evaluation.

Assess the druggability of protein targets based on structural features, binding sites, and known drug discovery principles."""

            human_prompt = f"""
Assess the druggability of this protein target:

Structure Data:
{json.dumps(structure_data, indent=2)}

Binding Sites:
{json.dumps(binding_sites, indent=2)}

Provide a comprehensive druggability assessment including:
1. Overall druggability score (1-10)
2. Key druggable features
3. Challenges for drug development
4. Recommended drug modalities (small molecules, biologics, etc.)
5. Similar successful drug targets
6. Development strategy recommendations
"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = await self._call_llm_async_messages(messages)
            
            return {"druggability_assessment": response.content}
            
        except Exception as e:
            logger.error(f"Druggability assessment failed: {str(e)}")
            return {"error": str(e)}
    
    async def _call_llm_async(self, prompt: ChatPromptTemplate, data: Dict[str, Any]) -> Any:
        """Async wrapper for LLM calls"""
        try:
            # Format the prompt
            formatted_prompt = prompt.format(**data)
            
            # Call the LLM
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.llm.invoke([HumanMessage(content=formatted_prompt)])
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise
    
    async def _call_llm_async_messages(self, messages: List) -> Any:
        """Async wrapper for LLM calls with message list"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.llm.invoke(messages)
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise
    
    def _get_sequence_context(self, sequence: str, residue_numbers: List[int], context_size: int = 10) -> str:
        """Get sequence context around binding site residues"""
        try:
            if not residue_numbers:
                return "No residues provided"
            
            min_res = min(residue_numbers) - 1  # Convert to 0-based indexing
            max_res = max(residue_numbers) - 1
            
            start = max(0, min_res - context_size)
            end = min(len(sequence), max_res + context_size + 1)
            
            context = sequence[start:end]
            
            # Mark binding site residues
            marked_context = ""
            for i, aa in enumerate(context):
                pos = start + i + 1  # Convert back to 1-based
                if pos in residue_numbers:
                    marked_context += f"[{aa}]"
                else:
                    marked_context += aa
            
            return marked_context
            
        except Exception as e:
            logger.error(f"Failed to get sequence context: {str(e)}")
            return "Context unavailable"
    
    async def generate_publication_abstract(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a scientific abstract based on analysis results
        """
        try:
            logger.info("Generating publication abstract")
            
            system_prompt = """You are a scientific writer specializing in structural biology and computational biology publications.

Write a professional scientific abstract based on the provided analysis results. The abstract should be suitable for a high-impact journal and follow standard scientific writing conventions."""

            human_prompt = f"""
Write a scientific abstract based on these analysis results:

{json.dumps(analysis_results, indent=2)}

The abstract should include:
1. Background and motivation
2. Methods used
3. Key findings
4. Significance and implications
5. Potential applications

Keep it concise (250-300 words) and use appropriate scientific terminology.
"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = await self._call_llm_async_messages(messages)
            
            return response.content
            
        except Exception as e:
            logger.error(f"Abstract generation failed: {str(e)}")
            return f"Error generating abstract: {str(e)}" 