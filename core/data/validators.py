"""
Data Validation Module

Comprehensive validation for sequences, structures, and prediction inputs.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import molecular_weight, IsoelectricPoint
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import warnings

logger = logging.getLogger(__name__)

class ValidationLevel(str, Enum):
    """Validation severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationIssue:
    """Individual validation issue"""
    level: ValidationLevel
    code: str
    message: str
    position: Optional[int] = None
    suggestion: Optional[str] = None

@dataclass
class ValidationResult:
    """Complete validation result"""
    valid: bool
    issues: List[ValidationIssue]
    score: float  # Overall quality score (0-100)
    metadata: Dict[str, Any]
    
    @property
    def errors(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == ValidationLevel.ERROR]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == ValidationLevel.WARNING]
    
    @property
    def infos(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == ValidationLevel.INFO]

class ValidationError(Exception):
    """Custom validation error"""
    def __init__(self, message: str, validation_result: Optional[ValidationResult] = None):
        super().__init__(message)
        self.validation_result = validation_result

class SequenceValidator:
    """Comprehensive sequence validation"""
    
    # Standard amino acids
    STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Extended amino acids (including ambiguous)
    EXTENDED_AA = set('ACDEFGHIKLMNPQRSTVWYXBZJUO*')
    
    # Amino acid properties
    HYDROPHOBIC_AA = set('AILMFPWYV')
    POLAR_AA = set('NQST')
    CHARGED_AA = set('DEKR')
    AROMATIC_AA = set('FWY')
    
    # Secondary structure propensities
    HELIX_PRONE_AA = set('AEHKLMQR')
    SHEET_PRONE_AA = set('CFILTVY')
    TURN_PRONE_AA = set('DGHNPST')
    
    def __init__(self, 
                 min_length: int = 10,
                 max_length: int = 2048,
                 allow_ambiguous: bool = True,
                 max_ambiguous_fraction: float = 0.1):
        self.min_length = min_length
        self.max_length = max_length
        self.allow_ambiguous = allow_ambiguous
        self.max_ambiguous_fraction = max_ambiguous_fraction
    
    def validate(self, sequence: str, sequence_id: str = "query") -> ValidationResult:
        """Comprehensive sequence validation"""
        issues = []
        metadata = {}
        
        try:
            # Clean sequence
            cleaned_seq = self._clean_sequence(sequence)
            metadata['original_length'] = len(sequence)
            metadata['cleaned_length'] = len(cleaned_seq)
            metadata['sequence_id'] = sequence_id
            
            # Basic validation
            issues.extend(self._validate_length(cleaned_seq))
            issues.extend(self._validate_composition(cleaned_seq))
            issues.extend(self._validate_patterns(cleaned_seq))
            
            # Advanced validation
            issues.extend(self._validate_physicochemical_properties(cleaned_seq))
            issues.extend(self._validate_structural_features(cleaned_seq))
            issues.extend(self._validate_biological_plausibility(cleaned_seq))
            
            # Calculate quality score
            score = self._calculate_quality_score(cleaned_seq, issues)
            metadata['quality_score'] = score
            
            # Add sequence analysis
            metadata.update(self._analyze_sequence(cleaned_seq))
            
            # Determine if valid (no errors)
            valid = not any(issue.level == ValidationLevel.ERROR for issue in issues)
            
            return ValidationResult(
                valid=valid,
                issues=issues,
                score=score,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Sequence validation failed: {str(e)}")
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                code="VALIDATION_ERROR",
                message=f"Validation failed: {str(e)}"
            ))
            
            return ValidationResult(
                valid=False,
                issues=issues,
                score=0.0,
                metadata=metadata
            )
    
    def _clean_sequence(self, sequence: str) -> str:
        """Clean and normalize sequence"""
        # Remove whitespace and convert to uppercase
        cleaned = re.sub(r'\s+', '', sequence.upper())
        
        # Remove non-amino acid characters except standard ones
        cleaned = re.sub(r'[^ACDEFGHIKLMNPQRSTVWYXBZJUO*-]', '', cleaned)
        
        return cleaned
    
    def _validate_length(self, sequence: str) -> List[ValidationIssue]:
        """Validate sequence length"""
        issues = []
        length = len(sequence)
        
        if length < self.min_length:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                code="SEQ_TOO_SHORT",
                message=f"Sequence too short: {length} < {self.min_length} residues",
                suggestion=f"Provide a sequence with at least {self.min_length} residues"
            ))
        
        if length > self.max_length:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                code="SEQ_TOO_LONG",
                message=f"Sequence too long: {length} > {self.max_length} residues",
                suggestion=f"Truncate sequence to maximum {self.max_length} residues"
            ))
        
        # Length warnings
        if length < 30:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                code="SEQ_VERY_SHORT",
                message=f"Very short sequence: {length} residues",
                suggestion="Short sequences may have limited structural information"
            ))
        
        if length > 1000:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                code="SEQ_VERY_LONG",
                message=f"Very long sequence: {length} residues",
                suggestion="Long sequences may require more computational resources"
            ))
        
        return issues
    
    def _validate_composition(self, sequence: str) -> List[ValidationIssue]:
        """Validate amino acid composition"""
        issues = []
        
        # Check for invalid characters
        invalid_chars = set(sequence) - self.EXTENDED_AA
        if invalid_chars:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                code="INVALID_CHARACTERS",
                message=f"Invalid characters found: {sorted(invalid_chars)}",
                suggestion="Remove or replace invalid characters with standard amino acids"
            ))
        
        # Check ambiguous amino acids
        ambiguous_chars = set(sequence) - self.STANDARD_AA
        ambiguous_chars.discard('-')  # Gaps are handled separately
        
        if ambiguous_chars:
            ambiguous_count = sum(sequence.count(char) for char in ambiguous_chars)
            ambiguous_fraction = ambiguous_count / len(sequence)
            
            if not self.allow_ambiguous:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    code="AMBIGUOUS_NOT_ALLOWED",
                    message=f"Ambiguous amino acids not allowed: {sorted(ambiguous_chars)}",
                    suggestion="Replace ambiguous amino acids with specific ones"
                ))
            elif ambiguous_fraction > self.max_ambiguous_fraction:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="HIGH_AMBIGUOUS_CONTENT",
                    message=f"High ambiguous content: {ambiguous_fraction:.1%} > {self.max_ambiguous_fraction:.1%}",
                    suggestion="Consider reducing ambiguous amino acids for better predictions"
                ))
        
        # Check for gaps
        if '-' in sequence:
            gap_count = sequence.count('-')
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                code="GAPS_PRESENT",
                message=f"Sequence contains {gap_count} gaps",
                suggestion="Remove gaps for structure prediction"
            ))
        
        # Check for stop codons
        if '*' in sequence:
            stop_count = sequence.count('*')
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                code="STOP_CODONS",
                message=f"Sequence contains {stop_count} stop codons",
                suggestion="Remove stop codons for structure prediction"
            ))
        
        return issues
    
    def _validate_patterns(self, sequence: str) -> List[ValidationIssue]:
        """Validate sequence patterns"""
        issues = []
        
        # Check for repetitive sequences
        for repeat_length in [2, 3, 4, 5]:
            max_repeats = self._find_max_repeats(sequence, repeat_length)
            if max_repeats > 10:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="REPETITIVE_SEQUENCE",
                    message=f"Highly repetitive sequence: {max_repeats} consecutive {repeat_length}-mers",
                    suggestion="Repetitive sequences may have unusual structures"
                ))
        
        # Check for low complexity regions
        complexity_score = self._calculate_complexity(sequence)
        if complexity_score < 0.3:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                code="LOW_COMPLEXITY",
                message=f"Low sequence complexity: {complexity_score:.2f}",
                suggestion="Low complexity regions may be disordered"
            ))
        
        # Check for unusual amino acid frequencies
        aa_frequencies = {aa: sequence.count(aa) / len(sequence) for aa in self.STANDARD_AA}
        
        for aa, freq in aa_frequencies.items():
            if freq > 0.3:  # More than 30% of any single amino acid
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="HIGH_AA_FREQUENCY",
                    message=f"High frequency of {aa}: {freq:.1%}",
                    suggestion="Unusual amino acid composition may affect structure"
                ))
        
        return issues
    
    def _validate_physicochemical_properties(self, sequence: str) -> List[ValidationIssue]:
        """Validate physicochemical properties"""
        issues = []
        
        try:
            # Calculate properties
            hydrophobic_fraction = sum(sequence.count(aa) for aa in self.HYDROPHOBIC_AA) / len(sequence)
            charged_fraction = sum(sequence.count(aa) for aa in self.CHARGED_AA) / len(sequence)
            
            # Check extreme hydrophobicity
            if hydrophobic_fraction > 0.7:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="HIGHLY_HYDROPHOBIC",
                    message=f"Highly hydrophobic sequence: {hydrophobic_fraction:.1%}",
                    suggestion="May be membrane protein or have folding issues"
                ))
            elif hydrophobic_fraction < 0.1:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="HIGHLY_HYDROPHILIC",
                    message=f"Highly hydrophilic sequence: {hydrophobic_fraction:.1%}",
                    suggestion="May be intrinsically disordered"
                ))
            
            # Check charge distribution
            if charged_fraction > 0.4:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="HIGHLY_CHARGED",
                    message=f"Highly charged sequence: {charged_fraction:.1%}",
                    suggestion="May have unusual electrostatic properties"
                ))
            
            # Calculate molecular weight
            try:
                analysis = ProteinAnalysis(sequence)
                mw = analysis.molecular_weight()
                
                if mw > 100000:  # > 100 kDa
                    issues.append(ValidationIssue(
                        level=ValidationLevel.INFO,
                        code="LARGE_PROTEIN",
                        message=f"Large protein: {mw/1000:.1f} kDa",
                        suggestion="Large proteins may require domain-based prediction"
                    ))
                
            except Exception as e:
                logger.warning(f"Failed to calculate molecular weight: {str(e)}")
        
        except Exception as e:
            logger.warning(f"Physicochemical validation failed: {str(e)}")
        
        return issues
    
    def _validate_structural_features(self, sequence: str) -> List[ValidationIssue]:
        """Validate structural features"""
        issues = []
        
        # Check cysteine content for disulfide bonds
        cys_count = sequence.count('C')
        if cys_count > 0:
            if cys_count % 2 == 1:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="ODD_CYSTEINE_COUNT",
                    message=f"Odd number of cysteines: {cys_count}",
                    suggestion="May have unpaired cysteine residues"
                ))
            
            cys_fraction = cys_count / len(sequence)
            if cys_fraction > 0.1:
                issues.append(ValidationIssue(
                    level=ValidationLevel.INFO,
                    code="HIGH_CYSTEINE_CONTENT",
                    message=f"High cysteine content: {cys_fraction:.1%}",
                    suggestion="Protein may have multiple disulfide bonds"
                ))
        
        # Check proline content
        pro_count = sequence.count('P')
        pro_fraction = pro_count / len(sequence)
        if pro_fraction > 0.15:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                code="HIGH_PROLINE_CONTENT",
                message=f"High proline content: {pro_fraction:.1%}",
                suggestion="May have unusual secondary structure"
            ))
        
        # Check glycine content
        gly_count = sequence.count('G')
        gly_fraction = gly_count / len(sequence)
        if gly_fraction > 0.2:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                code="HIGH_GLYCINE_CONTENT",
                message=f"High glycine content: {gly_fraction:.1%}",
                suggestion="May be highly flexible or disordered"
            ))
        
        return issues
    
    def _validate_biological_plausibility(self, sequence: str) -> List[ValidationIssue]:
        """Validate biological plausibility"""
        issues = []
        
        # Check for signal peptides (simplified)
        if len(sequence) > 20:
            n_terminal = sequence[:20]
            hydrophobic_start = sum(aa in self.HYDROPHOBIC_AA for aa in n_terminal[:10])
            
            if hydrophobic_start > 7:
                issues.append(ValidationIssue(
                    level=ValidationLevel.INFO,
                    code="POSSIBLE_SIGNAL_PEPTIDE",
                    message="Possible signal peptide detected at N-terminus",
                    suggestion="May be cleaved in mature protein"
                ))
        
        # Check for transmembrane regions (very simplified)
        window_size = 20
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            hydrophobic_count = sum(aa in self.HYDROPHOBIC_AA for aa in window)
            
            if hydrophobic_count > 15:  # > 75% hydrophobic
                issues.append(ValidationIssue(
                    level=ValidationLevel.INFO,
                    code="POSSIBLE_TRANSMEMBRANE",
                    message=f"Possible transmembrane region at position {i+1}-{i+window_size}",
                    suggestion="May be membrane protein"
                ))
                break  # Only report first one
        
        return issues
    
    def _find_max_repeats(self, sequence: str, repeat_length: int) -> int:
        """Find maximum consecutive repeats of given length"""
        max_repeats = 0
        current_repeats = 0
        
        for i in range(len(sequence) - repeat_length + 1):
            if i + 2 * repeat_length <= len(sequence):
                current_motif = sequence[i:i + repeat_length]
                next_motif = sequence[i + repeat_length:i + 2 * repeat_length]
                
                if current_motif == next_motif:
                    current_repeats += 1
                else:
                    max_repeats = max(max_repeats, current_repeats)
                    current_repeats = 0
        
        return max(max_repeats, current_repeats)
    
    def _calculate_complexity(self, sequence: str) -> float:
        """Calculate sequence complexity (Shannon entropy)"""
        if not sequence:
            return 0.0
        
        # Count amino acid frequencies
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        # Calculate Shannon entropy
        length = len(sequence)
        entropy = 0.0
        
        for count in aa_counts.values():
            if count > 0:
                p = count / length
                entropy -= p * np.log2(p)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(min(20, len(set(sequence))))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_quality_score(self, sequence: str, issues: List[ValidationIssue]) -> float:
        """Calculate overall quality score"""
        base_score = 100.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.level == ValidationLevel.ERROR:
                base_score -= 30
            elif issue.level == ValidationLevel.WARNING:
                base_score -= 10
            else:  # INFO
                base_score -= 2
        
        # Bonus for good length
        length = len(sequence)
        if 50 <= length <= 500:
            base_score += 5
        
        # Bonus for good complexity
        complexity = self._calculate_complexity(sequence)
        if complexity > 0.7:
            base_score += 5
        
        return max(0.0, min(100.0, base_score))
    
    def _analyze_sequence(self, sequence: str) -> Dict[str, Any]:
        """Analyze sequence properties"""
        analysis = {}
        
        try:
            # Basic composition
            aa_counts = {aa: sequence.count(aa) for aa in self.STANDARD_AA}
            aa_frequencies = {aa: count / len(sequence) for aa, count in aa_counts.items()}
            
            analysis['amino_acid_composition'] = aa_frequencies
            analysis['length'] = len(sequence)
            
            # Physicochemical properties
            hydrophobic_fraction = sum(sequence.count(aa) for aa in self.HYDROPHOBIC_AA) / len(sequence)
            polar_fraction = sum(sequence.count(aa) for aa in self.POLAR_AA) / len(sequence)
            charged_fraction = sum(sequence.count(aa) for aa in self.CHARGED_AA) / len(sequence)
            aromatic_fraction = sum(sequence.count(aa) for aa in self.AROMATIC_AA) / len(sequence)
            
            analysis['hydrophobic_fraction'] = hydrophobic_fraction
            analysis['polar_fraction'] = polar_fraction
            analysis['charged_fraction'] = charged_fraction
            analysis['aromatic_fraction'] = aromatic_fraction
            
            # Secondary structure propensities
            helix_propensity = sum(sequence.count(aa) for aa in self.HELIX_PRONE_AA) / len(sequence)
            sheet_propensity = sum(sequence.count(aa) for aa in self.SHEET_PRONE_AA) / len(sequence)
            turn_propensity = sum(sequence.count(aa) for aa in self.TURN_PRONE_AA) / len(sequence)
            
            analysis['helix_propensity'] = helix_propensity
            analysis['sheet_propensity'] = sheet_propensity
            analysis['turn_propensity'] = turn_propensity
            
            # Complexity
            analysis['complexity'] = self._calculate_complexity(sequence)
            
            # Use ProteinAnalysis if available
            try:
                protein_analysis = ProteinAnalysis(sequence)
                analysis['molecular_weight'] = protein_analysis.molecular_weight()
                analysis['isoelectric_point'] = protein_analysis.isoelectric_point()
                analysis['instability_index'] = protein_analysis.instability_index()
                analysis['gravy'] = protein_analysis.gravy()  # Grand average of hydropathy
            except Exception as e:
                logger.warning(f"ProteinAnalysis failed: {str(e)}")
        
        except Exception as e:
            logger.warning(f"Sequence analysis failed: {str(e)}")
        
        return analysis

class StructureValidator:
    """Validate protein structures"""
    
    def __init__(self):
        pass
    
    def validate_pdb(self, pdb_content: str, structure_id: str = "structure") -> ValidationResult:
        """Validate PDB structure"""
        issues = []
        metadata = {}
        
        try:
            # Basic PDB format validation
            lines = pdb_content.strip().split('\n')
            
            # Check for required records
            has_header = any(line.startswith('HEADER') for line in lines)
            has_atom = any(line.startswith('ATOM') for line in lines)
            has_end = any(line.startswith('END') for line in lines)
            
            if not has_header:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="NO_HEADER",
                    message="PDB file missing HEADER record"
                ))
            
            if not has_atom:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    code="NO_ATOMS",
                    message="PDB file contains no ATOM records"
                ))
            
            if not has_end:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="NO_END",
                    message="PDB file missing END record"
                ))
            
            # Count atoms and residues
            atom_count = sum(1 for line in lines if line.startswith('ATOM'))
            residue_numbers = set()
            
            for line in lines:
                if line.startswith('ATOM') and len(line) >= 26:
                    try:
                        res_num = int(line[22:26].strip())
                        residue_numbers.add(res_num)
                    except ValueError:
                        pass
            
            metadata['atom_count'] = atom_count
            metadata['residue_count'] = len(residue_numbers)
            metadata['structure_id'] = structure_id
            
            # Validate structure quality
            if atom_count < 10:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="FEW_ATOMS",
                    message=f"Very few atoms: {atom_count}",
                    suggestion="Structure may be incomplete"
                ))
            
            # Calculate quality score
            score = self._calculate_structure_quality_score(issues, metadata)
            metadata['quality_score'] = score
            
            valid = not any(issue.level == ValidationLevel.ERROR for issue in issues)
            
            return ValidationResult(
                valid=valid,
                issues=issues,
                score=score,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Structure validation failed: {str(e)}")
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                code="VALIDATION_ERROR",
                message=f"Structure validation failed: {str(e)}"
            ))
            
            return ValidationResult(
                valid=False,
                issues=issues,
                score=0.0,
                metadata=metadata
            )
    
    def _calculate_structure_quality_score(self, issues: List[ValidationIssue], metadata: Dict) -> float:
        """Calculate structure quality score"""
        base_score = 100.0
        
        # Deduct for issues
        for issue in issues:
            if issue.level == ValidationLevel.ERROR:
                base_score -= 40
            elif issue.level == ValidationLevel.WARNING:
                base_score -= 15
            else:
                base_score -= 3
        
        # Bonus for reasonable size
        atom_count = metadata.get('atom_count', 0)
        if atom_count > 100:
            base_score += 10
        
        return max(0.0, min(100.0, base_score))

# Factory functions
def create_sequence_validator(**kwargs) -> SequenceValidator:
    """Create sequence validator with custom parameters"""
    return SequenceValidator(**kwargs)

def create_structure_validator() -> StructureValidator:
    """Create structure validator"""
    return StructureValidator()

def validate_sequence(sequence: str, **kwargs) -> ValidationResult:
    """Quick sequence validation"""
    validator = create_sequence_validator(**kwargs)
    return validator.validate(sequence)

def validate_structure(pdb_content: str, **kwargs) -> ValidationResult:
    """Quick structure validation"""
    validator = create_structure_validator()
    return validator.validate_pdb(pdb_content, **kwargs) 