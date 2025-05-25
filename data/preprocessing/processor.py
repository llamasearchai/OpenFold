"""
Advanced Biomolecule Data Preprocessing Module

Handles preprocessing of protein, RNA, DNA, and small molecule data with comprehensive
validation, normalization, and feature extraction capabilities.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import hashlib
import json
from datetime import datetime

# Biomolecule processing libraries
from Bio import SeqIO, PDB
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.DSSP import DSSP
import biotite.structure as struc
import biotite.structure.io as strucio
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import mdtraj as md

logger = logging.getLogger(__name__)

class MoleculeType(str, Enum):
    PROTEIN = "protein"
    RNA = "rna"
    DNA = "dna"
    SMALL_MOLECULE = "small_molecule"
    COMPLEX = "complex"

class DataFormat(str, Enum):
    FASTA = "fasta"
    PDB = "pdb"
    CIF = "cif"
    SDF = "sdf"
    MOL2 = "mol2"
    SMILES = "smiles"

@dataclass
class ProcessingResult:
    """Result of data preprocessing"""
    success: bool
    data: Optional[Any] = None
    metadata: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    processing_time: float = 0.0
    checksum: Optional[str] = None

@dataclass
class SequenceFeatures:
    """Extracted sequence features"""
    length: int
    composition: Dict[str, float]
    molecular_weight: Optional[float] = None
    isoelectric_point: Optional[float] = None
    hydrophobicity: Optional[float] = None
    secondary_structure: Optional[Dict[str, float]] = None
    disorder_regions: Optional[List[Tuple[int, int]]] = None

@dataclass
class StructureFeatures:
    """Extracted structure features"""
    num_atoms: int
    num_residues: int
    resolution: Optional[float] = None
    r_factor: Optional[float] = None
    space_group: Optional[str] = None
    unit_cell: Optional[List[float]] = None
    secondary_structure: Optional[Dict[str, float]] = None
    surface_area: Optional[float] = None
    volume: Optional[float] = None
    binding_sites: Optional[List[Dict]] = None

@dataclass
class SmallMoleculeFeatures:
    """Extracted small molecule features"""
    molecular_weight: float
    logp: float
    hbd: int  # Hydrogen bond donors
    hba: int  # Hydrogen bond acceptors
    tpsa: float  # Topological polar surface area
    rotatable_bonds: int
    aromatic_rings: int
    lipinski_violations: int
    qed: float  # Quantitative Estimate of Drug-likeness
    smiles: str
    inchi: Optional[str] = None

class BiomoleculeProcessor:
    """Advanced biomolecule data processor"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        self.dna_bases = set('ATCG')
        self.rna_bases = set('AUCG')
        
        # Amino acid properties
        self.aa_properties = {
            'hydrophobic': set('AILMFPWV'),
            'polar': set('NQST'),
            'charged': set('DEKR'),
            'aromatic': set('FWY'),
            'small': set('AGCS'),
            'large': set('FWYR')
        }
        
        # Initialize structure parser
        self.pdb_parser = PDBParser(QUIET=True)
        
    def process_sequence(self, sequence: str, molecule_type: MoleculeType) -> ProcessingResult:
        """Process and validate sequence data"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            # Clean and validate sequence
            cleaned_seq = self._clean_sequence(sequence)
            
            # Validate sequence characters
            if molecule_type == MoleculeType.PROTEIN:
                invalid_chars = set(cleaned_seq.upper()) - self.amino_acids
                if invalid_chars:
                    errors.append(f"Invalid amino acid characters: {invalid_chars}")
            elif molecule_type == MoleculeType.DNA:
                invalid_chars = set(cleaned_seq.upper()) - self.dna_bases
                if invalid_chars:
                    errors.append(f"Invalid DNA base characters: {invalid_chars}")
            elif molecule_type == MoleculeType.RNA:
                invalid_chars = set(cleaned_seq.upper()) - self.rna_bases
                if invalid_chars:
                    errors.append(f"Invalid RNA base characters: {invalid_chars}")
            
            if errors:
                return ProcessingResult(
                    success=False,
                    errors=errors,
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Extract features
            features = self._extract_sequence_features(cleaned_seq, molecule_type)
            
            # Generate metadata
            metadata = {
                'original_length': len(sequence),
                'processed_length': len(cleaned_seq),
                'molecule_type': molecule_type.value,
                'features': features.__dict__,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Calculate checksum
            checksum = hashlib.sha256(cleaned_seq.encode()).hexdigest()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                data=cleaned_seq,
                metadata=metadata,
                warnings=warnings,
                processing_time=processing_time,
                checksum=checksum
            )
            
        except Exception as e:
            logger.error(f"Error processing sequence: {str(e)}")
            return ProcessingResult(
                success=False,
                errors=[f"Processing error: {str(e)}"],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def process_structure(self, structure_data: Union[str, Path], format: DataFormat) -> ProcessingResult:
        """Process and validate structure data"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            if format == DataFormat.PDB:
                result = self._process_pdb_structure(structure_data)
            elif format == DataFormat.CIF:
                result = self._process_cif_structure(structure_data)
            else:
                return ProcessingResult(
                    success=False,
                    errors=[f"Unsupported structure format: {format}"],
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            if not result.success:
                return result
            
            # Extract structural features
            features = self._extract_structure_features(result.data)
            
            # Update metadata
            result.metadata.update({
                'features': features.__dict__,
                'processing_timestamp': datetime.now().isoformat()
            })
            
            result.processing_time = (datetime.now() - start_time).total_seconds()
            return result
            
        except Exception as e:
            logger.error(f"Error processing structure: {str(e)}")
            return ProcessingResult(
                success=False,
                errors=[f"Processing error: {str(e)}"],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def process_small_molecule(self, molecule_data: str, format: DataFormat) -> ProcessingResult:
        """Process and validate small molecule data"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            if format == DataFormat.SMILES:
                mol = Chem.MolFromSmiles(molecule_data)
            elif format == DataFormat.SDF:
                mol = Chem.MolFromMolBlock(molecule_data)
            else:
                return ProcessingResult(
                    success=False,
                    errors=[f"Unsupported molecule format: {format}"],
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            if mol is None:
                return ProcessingResult(
                    success=False,
                    errors=["Invalid molecule structure"],
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Sanitize molecule
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                warnings.append(f"Molecule sanitization warning: {str(e)}")
            
            # Extract features
            features = self._extract_small_molecule_features(mol)
            
            # Generate canonical SMILES
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            
            # Generate metadata
            metadata = {
                'original_format': format.value,
                'canonical_smiles': canonical_smiles,
                'features': features.__dict__,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Calculate checksum
            checksum = hashlib.sha256(canonical_smiles.encode()).hexdigest()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                data=mol,
                metadata=metadata,
                warnings=warnings,
                processing_time=processing_time,
                checksum=checksum
            )
            
        except Exception as e:
            logger.error(f"Error processing small molecule: {str(e)}")
            return ProcessingResult(
                success=False,
                errors=[f"Processing error: {str(e)}"],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _clean_sequence(self, sequence: str) -> str:
        """Clean and normalize sequence"""
        # Remove whitespace and convert to uppercase
        cleaned = re.sub(r'\s+', '', sequence.upper())
        
        # Remove common sequence formatting characters
        cleaned = re.sub(r'[^A-Z]', '', cleaned)
        
        return cleaned
    
    def _extract_sequence_features(self, sequence: str, molecule_type: MoleculeType) -> SequenceFeatures:
        """Extract features from sequence"""
        length = len(sequence)
        
        # Calculate composition
        composition = {}
        for char in set(sequence):
            composition[char] = sequence.count(char) / length
        
        features = SequenceFeatures(
            length=length,
            composition=composition
        )
        
        if molecule_type == MoleculeType.PROTEIN:
            # Calculate protein-specific features
            features.molecular_weight = self._calculate_protein_mw(sequence)
            features.hydrophobicity = self._calculate_hydrophobicity(sequence)
            features.secondary_structure = self._predict_secondary_structure(sequence)
        
        return features
    
    def _extract_structure_features(self, structure) -> StructureFeatures:
        """Extract features from 3D structure"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated structure analysis
        
        num_atoms = 0
        num_residues = 0
        
        # Count atoms and residues
        for model in structure:
            for chain in model:
                num_residues += len(chain)
                for residue in chain:
                    num_atoms += len(residue)
        
        return StructureFeatures(
            num_atoms=num_atoms,
            num_residues=num_residues
        )
    
    def _extract_small_molecule_features(self, mol) -> SmallMoleculeFeatures:
        """Extract features from small molecule"""
        return SmallMoleculeFeatures(
            molecular_weight=Descriptors.MolWt(mol),
            logp=Crippen.MolLogP(mol),
            hbd=Descriptors.NumHDonors(mol),
            hba=Descriptors.NumHAcceptors(mol),
            tpsa=Descriptors.TPSA(mol),
            rotatable_bonds=Descriptors.NumRotatableBonds(mol),
            aromatic_rings=Descriptors.NumAromaticRings(mol),
            lipinski_violations=self._count_lipinski_violations(mol),
            qed=self._calculate_qed(mol),
            smiles=Chem.MolToSmiles(mol, canonical=True)
        )
    
    def _process_pdb_structure(self, pdb_data: Union[str, Path]) -> ProcessingResult:
        """Process PDB structure data"""
        try:
            if isinstance(pdb_data, Path):
                structure = self.pdb_parser.get_structure('structure', pdb_data)
            else:
                # Handle string data (would need temporary file in real implementation)
                structure = self.pdb_parser.get_structure('structure', pdb_data)
            
            metadata = {
                'format': 'pdb',
                'num_models': len(structure),
                'chains': [chain.id for model in structure for chain in model]
            }
            
            return ProcessingResult(
                success=True,
                data=structure,
                metadata=metadata
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                errors=[f"PDB parsing error: {str(e)}"]
            )
    
    def _process_cif_structure(self, cif_data: Union[str, Path]) -> ProcessingResult:
        """Process CIF structure data"""
        try:
            # Use biotite for CIF parsing
            if isinstance(cif_data, Path):
                structure = strucio.load_structure(cif_data)
            else:
                # Handle string data
                structure = strucio.load_structure(cif_data)
            
            metadata = {
                'format': 'cif',
                'num_atoms': len(structure),
                'chains': list(set(structure.chain_id))
            }
            
            return ProcessingResult(
                success=True,
                data=structure,
                metadata=metadata
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                errors=[f"CIF parsing error: {str(e)}"]
            )
    
    def _calculate_protein_mw(self, sequence: str) -> float:
        """Calculate protein molecular weight"""
        # Simplified MW calculation (average amino acid weight)
        aa_weights = {
            'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.0,
            'Q': 146.1, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
            'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
            'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
        }
        
        total_weight = sum(aa_weights.get(aa, 110.0) for aa in sequence)
        # Subtract water molecules (n-1 peptide bonds)
        total_weight -= (len(sequence) - 1) * 18.015
        
        return total_weight
    
    def _calculate_hydrophobicity(self, sequence: str) -> float:
        """Calculate sequence hydrophobicity"""
        hydrophobic_count = sum(1 for aa in sequence if aa in self.aa_properties['hydrophobic'])
        return hydrophobic_count / len(sequence)
    
    def _predict_secondary_structure(self, sequence: str) -> Dict[str, float]:
        """Simple secondary structure prediction"""
        # This is a very simplified prediction
        # In practice, you'd use sophisticated algorithms like DSSP or neural networks
        
        helix_propensity = {'A': 1.42, 'E': 1.51, 'L': 1.21, 'M': 1.45}
        sheet_propensity = {'V': 1.70, 'I': 1.60, 'Y': 1.47, 'F': 1.38}
        
        helix_score = sum(helix_propensity.get(aa, 1.0) for aa in sequence) / len(sequence)
        sheet_score = sum(sheet_propensity.get(aa, 1.0) for aa in sequence) / len(sequence)
        
        # Normalize scores
        total = helix_score + sheet_score + 1.0  # +1 for coil
        
        return {
            'helix': helix_score / total,
            'sheet': sheet_score / total,
            'coil': 1.0 / total
        }
    
    def _count_lipinski_violations(self, mol) -> int:
        """Count Lipinski rule of five violations"""
        violations = 0
        
        if Descriptors.MolWt(mol) > 500:
            violations += 1
        if Crippen.MolLogP(mol) > 5:
            violations += 1
        if Descriptors.NumHDonors(mol) > 5:
            violations += 1
        if Descriptors.NumHAcceptors(mol) > 10:
            violations += 1
            
        return violations
    
    def _calculate_qed(self, mol) -> float:
        """Calculate Quantitative Estimate of Drug-likeness"""
        # Simplified QED calculation
        # In practice, you'd use the full QED algorithm
        
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        # Simple scoring based on drug-like ranges
        mw_score = 1.0 if 150 <= mw <= 500 else 0.5
        logp_score = 1.0 if -2 <= logp <= 5 else 0.5
        hbd_score = 1.0 if hbd <= 5 else 0.5
        hba_score = 1.0 if hba <= 10 else 0.5
        
        return (mw_score + logp_score + hbd_score + hba_score) / 4.0

class BatchProcessor:
    """Batch processing for multiple biomolecules"""
    
    def __init__(self, processor: BiomoleculeProcessor):
        self.processor = processor
    
    def process_fasta_file(self, file_path: Path, molecule_type: MoleculeType) -> List[ProcessingResult]:
        """Process FASTA file with multiple sequences"""
        results = []
        
        try:
            for record in SeqIO.parse(file_path, "fasta"):
                result = self.processor.process_sequence(str(record.seq), molecule_type)
                result.metadata['sequence_id'] = record.id
                result.metadata['description'] = record.description
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error processing FASTA file: {str(e)}")
            results.append(ProcessingResult(
                success=False,
                errors=[f"File processing error: {str(e)}"]
            ))
        
        return results
    
    def process_sdf_file(self, file_path: Path) -> List[ProcessingResult]:
        """Process SDF file with multiple molecules"""
        results = []
        
        try:
            supplier = Chem.SDMolSupplier(str(file_path))
            for i, mol in enumerate(supplier):
                if mol is not None:
                    # Convert to SDF block for processing
                    sdf_block = Chem.MolToMolBlock(mol)
                    result = self.processor.process_small_molecule(sdf_block, DataFormat.SDF)
                    result.metadata['molecule_index'] = i
                    if mol.HasProp('_Name'):
                        result.metadata['molecule_name'] = mol.GetProp('_Name')
                    results.append(result)
                else:
                    results.append(ProcessingResult(
                        success=False,
                        errors=[f"Invalid molecule at index {i}"]
                    ))
                    
        except Exception as e:
            logger.error(f"Error processing SDF file: {str(e)}")
            results.append(ProcessingResult(
                success=False,
                errors=[f"File processing error: {str(e)}"]
            ))
        
        return results

# Factory function for easy instantiation
def create_processor(config: Optional[Dict] = None) -> BiomoleculeProcessor:
    """Create a biomolecule processor with optional configuration"""
    return BiomoleculeProcessor(config)

def create_batch_processor(config: Optional[Dict] = None) -> BatchProcessor:
    """Create a batch processor with optional configuration"""
    processor = create_processor(config)
    return BatchProcessor(processor) 