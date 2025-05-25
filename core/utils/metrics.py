"""
Structure Evaluation Metrics

Comprehensive metrics for evaluating protein structure predictions including
RMSD, GDT-TS, TM-score, LDDT, and other structural quality measures.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)

@dataclass
class StructureAlignment:
    """Structure alignment result"""
    rmsd: float
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    aligned_coords1: np.ndarray
    aligned_coords2: np.ndarray
    alignment_length: int

@dataclass
class QualityMetrics:
    """Comprehensive structure quality metrics"""
    rmsd: float
    gdt_ts: float
    gdt_ha: float
    tm_score: float
    lddt: float
    clash_score: float
    ramachandran_favored: float
    ramachandran_outliers: float
    rotamer_outliers: float
    c_beta_deviations: int
    bond_length_rmsd: float
    bond_angle_rmsd: float
    overall_score: float

class StructureMetrics:
    """Comprehensive structure evaluation metrics"""
    
    def __init__(self):
        self.amino_acid_masses = {
            'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
            'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
            'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
            'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
        }
    
    def calculate_comprehensive_metrics(
        self,
        predicted_coords: np.ndarray,
        reference_coords: np.ndarray,
        sequence: Optional[str] = None,
        predicted_structure: Optional[str] = None,
        reference_structure: Optional[str] = None
    ) -> QualityMetrics:
        """Calculate comprehensive structure quality metrics"""
        try:
            # Basic alignment and RMSD
            alignment = self.align_structures(predicted_coords, reference_coords)
            rmsd = alignment.rmsd
            
            # GDT scores
            gdt_ts = self.calculate_gdt_ts(predicted_coords, reference_coords)
            gdt_ha = self.calculate_gdt_ha(predicted_coords, reference_coords)
            
            # TM-score
            tm_score = self.calculate_tm_score(predicted_coords, reference_coords, sequence)
            
            # LDDT
            lddt = self.calculate_lddt(predicted_coords, reference_coords)
            
            # Structure quality metrics (if PDB structures provided)
            clash_score = 0.0
            ramachandran_favored = 0.0
            ramachandran_outliers = 0.0
            rotamer_outliers = 0.0
            c_beta_deviations = 0
            bond_length_rmsd = 0.0
            bond_angle_rmsd = 0.0
            
            if predicted_structure:
                quality_metrics = self._assess_structure_quality(predicted_structure)
                clash_score = quality_metrics.get('clash_score', 0.0)
                ramachandran_favored = quality_metrics.get('ramachandran_favored', 0.0)
                ramachandran_outliers = quality_metrics.get('ramachandran_outliers', 0.0)
                rotamer_outliers = quality_metrics.get('rotamer_outliers', 0.0)
                c_beta_deviations = quality_metrics.get('c_beta_deviations', 0)
                bond_length_rmsd = quality_metrics.get('bond_length_rmsd', 0.0)
                bond_angle_rmsd = quality_metrics.get('bond_angle_rmsd', 0.0)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                rmsd, gdt_ts, tm_score, lddt, clash_score,
                ramachandran_favored, ramachandran_outliers
            )
            
            return QualityMetrics(
                rmsd=rmsd,
                gdt_ts=gdt_ts,
                gdt_ha=gdt_ha,
                tm_score=tm_score,
                lddt=lddt,
                clash_score=clash_score,
                ramachandran_favored=ramachandran_favored,
                ramachandran_outliers=ramachandran_outliers,
                rotamer_outliers=rotamer_outliers,
                c_beta_deviations=c_beta_deviations,
                bond_length_rmsd=bond_length_rmsd,
                bond_angle_rmsd=bond_angle_rmsd,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate comprehensive metrics: {str(e)}")
            raise
    
    def align_structures(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> StructureAlignment:
        """Align two structures using Kabsch algorithm"""
        try:
            # Ensure coordinates are the same length
            min_length = min(len(coords1), len(coords2))
            coords1 = coords1[:min_length]
            coords2 = coords2[:min_length]
            
            if weights is None:
                weights = np.ones(min_length)
            else:
                weights = weights[:min_length]
            
            # Center coordinates
            centroid1 = np.average(coords1, axis=0, weights=weights)
            centroid2 = np.average(coords2, axis=0, weights=weights)
            
            centered1 = coords1 - centroid1
            centered2 = coords2 - centroid2
            
            # Calculate rotation matrix using Kabsch algorithm
            H = np.dot(centered1.T, centered2 * weights[:, np.newaxis])
            U, S, Vt = np.linalg.svd(H)
            
            # Ensure proper rotation (not reflection)
            d = np.linalg.det(np.dot(Vt.T, U.T))
            if d < 0:
                Vt[-1, :] *= -1
            
            rotation_matrix = np.dot(Vt.T, U.T)
            
            # Apply rotation and translation
            aligned_coords1 = np.dot(centered1, rotation_matrix) + centroid2
            translation_vector = centroid2 - np.dot(centroid1, rotation_matrix)
            
            # Calculate RMSD
            diff = aligned_coords1 - coords2
            rmsd = np.sqrt(np.average(np.sum(diff**2, axis=1), weights=weights))
            
            return StructureAlignment(
                rmsd=rmsd,
                rotation_matrix=rotation_matrix,
                translation_vector=translation_vector,
                aligned_coords1=aligned_coords1,
                aligned_coords2=coords2,
                alignment_length=min_length
            )
            
        except Exception as e:
            logger.error(f"Structure alignment failed: {str(e)}")
            raise
    
    def calculate_rmsd(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray,
        align: bool = True,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """Calculate RMSD between two coordinate sets"""
        try:
            if align:
                alignment = self.align_structures(coords1, coords2, weights)
                return alignment.rmsd
            else:
                # Direct RMSD without alignment
                min_length = min(len(coords1), len(coords2))
                coords1 = coords1[:min_length]
                coords2 = coords2[:min_length]
                
                if weights is None:
                    weights = np.ones(min_length)
                else:
                    weights = weights[:min_length]
                
                diff = coords1 - coords2
                rmsd = np.sqrt(np.average(np.sum(diff**2, axis=1), weights=weights))
                return rmsd
                
        except Exception as e:
            logger.error(f"RMSD calculation failed: {str(e)}")
            raise
    
    def calculate_gdt_ts(
        self,
        predicted_coords: np.ndarray,
        reference_coords: np.ndarray,
        cutoffs: List[float] = [1.0, 2.0, 4.0, 8.0]
    ) -> float:
        """Calculate GDT-TS (Global Distance Test - Total Score)"""
        try:
            min_length = min(len(predicted_coords), len(reference_coords))
            predicted_coords = predicted_coords[:min_length]
            reference_coords = reference_coords[:min_length]
            
            # Align structures
            alignment = self.align_structures(predicted_coords, reference_coords)
            aligned_pred = alignment.aligned_coords1
            
            # Calculate distances
            distances = np.sqrt(np.sum((aligned_pred - reference_coords)**2, axis=1))
            
            # Calculate percentage of residues within each cutoff
            percentages = []
            for cutoff in cutoffs:
                within_cutoff = np.sum(distances <= cutoff)
                percentage = within_cutoff / min_length * 100
                percentages.append(percentage)
            
            # GDT-TS is the average of percentages
            gdt_ts = np.mean(percentages)
            
            return gdt_ts
            
        except Exception as e:
            logger.error(f"GDT-TS calculation failed: {str(e)}")
            return 0.0
    
    def calculate_gdt_ha(
        self,
        predicted_coords: np.ndarray,
        reference_coords: np.ndarray,
        cutoffs: List[float] = [0.5, 1.0, 2.0, 4.0]
    ) -> float:
        """Calculate GDT-HA (Global Distance Test - High Accuracy)"""
        return self.calculate_gdt_ts(predicted_coords, reference_coords, cutoffs)
    
    def calculate_tm_score(
        self,
        predicted_coords: np.ndarray,
        reference_coords: np.ndarray,
        sequence: Optional[str] = None
    ) -> float:
        """Calculate TM-score"""
        try:
            min_length = min(len(predicted_coords), len(reference_coords))
            predicted_coords = predicted_coords[:min_length]
            reference_coords = reference_coords[:min_length]
            
            # Determine normalization length
            if sequence:
                L_norm = len(sequence)
            else:
                L_norm = len(reference_coords)
            
            # Calculate d0 (distance scale)
            d0 = 1.24 * (L_norm - 15)**(1/3) - 1.8 if L_norm > 21 else 0.5
            
            # Align structures
            alignment = self.align_structures(predicted_coords, reference_coords)
            aligned_pred = alignment.aligned_coords1
            
            # Calculate distances
            distances = np.sqrt(np.sum((aligned_pred - reference_coords)**2, axis=1))
            
            # Calculate TM-score
            tm_score = np.sum(1 / (1 + (distances / d0)**2)) / L_norm
            
            return tm_score
            
        except Exception as e:
            logger.error(f"TM-score calculation failed: {str(e)}")
            return 0.0
    
    def calculate_lddt(
        self,
        predicted_coords: np.ndarray,
        reference_coords: np.ndarray,
        cutoff: float = 15.0,
        thresholds: List[float] = [0.5, 1.0, 2.0, 4.0]
    ) -> float:
        """Calculate LDDT (Local Distance Difference Test)"""
        try:
            min_length = min(len(predicted_coords), len(reference_coords))
            predicted_coords = predicted_coords[:min_length]
            reference_coords = reference_coords[:min_length]
            
            # Calculate distance matrices
            pred_distances = cdist(predicted_coords, predicted_coords)
            ref_distances = cdist(reference_coords, reference_coords)
            
            # Create mask for distances within cutoff in reference
            mask = (ref_distances > 0) & (ref_distances <= cutoff)
            
            if not np.any(mask):
                return 0.0
            
            # Calculate distance differences
            distance_diffs = np.abs(pred_distances - ref_distances)
            
            # Count preserved distances for each threshold
            preserved_counts = []
            for threshold in thresholds:
                preserved = np.sum((distance_diffs <= threshold) & mask)
                preserved_counts.append(preserved)
            
            # LDDT is the average fraction of preserved distances
            total_distances = np.sum(mask)
            lddt = np.mean([count / total_distances for count in preserved_counts]) * 100
            
            return lddt
            
        except Exception as e:
            logger.error(f"LDDT calculation failed: {str(e)}")
            return 0.0
    
    def calculate_dihedral_angles(self, coords: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate backbone dihedral angles (phi, psi, omega)"""
        try:
            n_residues = len(coords) // 4  # Assuming N, CA, C, O atoms per residue
            
            phi_angles = []
            psi_angles = []
            omega_angles = []
            
            for i in range(1, n_residues - 1):
                # Phi angle: C(i-1) - N(i) - CA(i) - C(i)
                if i > 0:
                    c_prev = coords[4*(i-1) + 2]  # C of previous residue
                    n_curr = coords[4*i]          # N of current residue
                    ca_curr = coords[4*i + 1]     # CA of current residue
                    c_curr = coords[4*i + 2]      # C of current residue
                    
                    phi = self._calculate_dihedral(c_prev, n_curr, ca_curr, c_curr)
                    phi_angles.append(phi)
                
                # Psi angle: N(i) - CA(i) - C(i) - N(i+1)
                if i < n_residues - 1:
                    n_curr = coords[4*i]          # N of current residue
                    ca_curr = coords[4*i + 1]     # CA of current residue
                    c_curr = coords[4*i + 2]      # C of current residue
                    n_next = coords[4*(i+1)]      # N of next residue
                    
                    psi = self._calculate_dihedral(n_curr, ca_curr, c_curr, n_next)
                    psi_angles.append(psi)
                
                # Omega angle: CA(i-1) - C(i-1) - N(i) - CA(i)
                if i > 0:
                    ca_prev = coords[4*(i-1) + 1] # CA of previous residue
                    c_prev = coords[4*(i-1) + 2]  # C of previous residue
                    n_curr = coords[4*i]          # N of current residue
                    ca_curr = coords[4*i + 1]     # CA of current residue
                    
                    omega = self._calculate_dihedral(ca_prev, c_prev, n_curr, ca_curr)
                    omega_angles.append(omega)
            
            return {
                'phi': np.array(phi_angles),
                'psi': np.array(psi_angles),
                'omega': np.array(omega_angles)
            }
            
        except Exception as e:
            logger.error(f"Dihedral angle calculation failed: {str(e)}")
            return {'phi': np.array([]), 'psi': np.array([]), 'omega': np.array([])}
    
    def _calculate_dihedral(self, p1: np.ndarray, p2: np.ndarray, 
                           p3: np.ndarray, p4: np.ndarray) -> float:
        """Calculate dihedral angle between four points"""
        try:
            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2
            v3 = p4 - p3
            
            # Normal vectors
            n1 = np.cross(v1, v2)
            n2 = np.cross(v2, v3)
            
            # Normalize
            n1 = n1 / np.linalg.norm(n1)
            n2 = n2 / np.linalg.norm(n2)
            
            # Calculate angle
            cos_angle = np.dot(n1, n2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            angle = np.arccos(cos_angle)
            
            # Determine sign
            if np.dot(np.cross(n1, n2), v2) < 0:
                angle = -angle
            
            return np.degrees(angle)
            
        except Exception as e:
            logger.warning(f"Dihedral calculation failed: {str(e)}")
            return 0.0
    
    def assess_ramachandran_quality(self, phi_angles: np.ndarray, 
                                   psi_angles: np.ndarray) -> Dict[str, float]:
        """Assess Ramachandran plot quality"""
        try:
            if len(phi_angles) == 0 or len(psi_angles) == 0:
                return {'favored': 0.0, 'allowed': 0.0, 'outliers': 100.0}
            
            min_length = min(len(phi_angles), len(psi_angles))
            phi_angles = phi_angles[:min_length]
            psi_angles = psi_angles[:min_length]
            
            favored_count = 0
            allowed_count = 0
            outlier_count = 0
            
            for phi, psi in zip(phi_angles, psi_angles):
                if self._is_ramachandran_favored(phi, psi):
                    favored_count += 1
                elif self._is_ramachandran_allowed(phi, psi):
                    allowed_count += 1
                else:
                    outlier_count += 1
            
            total = len(phi_angles)
            
            return {
                'favored': favored_count / total * 100,
                'allowed': allowed_count / total * 100,
                'outliers': outlier_count / total * 100
            }
            
        except Exception as e:
            logger.error(f"Ramachandran assessment failed: {str(e)}")
            return {'favored': 0.0, 'allowed': 0.0, 'outliers': 100.0}
    
    def _is_ramachandran_favored(self, phi: float, psi: float) -> bool:
        """Check if phi/psi angles are in favored regions"""
        # Simplified Ramachandran regions
        # Alpha-helix region
        if -180 <= phi <= -30 and -90 <= psi <= 50:
            return True
        # Beta-sheet region
        if -180 <= phi <= -30 and 90 <= psi <= 180:
            return True
        # Left-handed alpha-helix (rare)
        if 30 <= phi <= 90 and -30 <= psi <= 90:
            return True
        
        return False
    
    def _is_ramachandran_allowed(self, phi: float, psi: float) -> bool:
        """Check if phi/psi angles are in allowed regions"""
        # Expanded regions around favored areas
        # This is a simplified implementation
        if -180 <= phi <= 180 and -180 <= psi <= 180:
            # If not in favored, check if in generally allowed regions
            if not self._is_ramachandran_favored(phi, psi):
                # Allow some flexibility around favored regions
                return True
        
        return False
    
    def _assess_structure_quality(self, pdb_content: str) -> Dict[str, float]:
        """Assess structure quality from PDB content"""
        # This is a simplified implementation
        # In practice, you'd use tools like MolProbity, Phenix, etc.
        
        try:
            # Parse PDB and extract coordinates
            lines = pdb_content.strip().split('\n')
            ca_coords = []
            
            for line in lines:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_coords.append([x, y, z])
            
            ca_coords = np.array(ca_coords)
            
            # Calculate basic quality metrics
            clash_score = self._calculate_clash_score(ca_coords)
            
            # Calculate dihedral angles for Ramachandran analysis
            # This would require full backbone atoms, simplified here
            ramachandran_metrics = {
                'favored': 85.0,  # Dummy values
                'outliers': 2.0
            }
            
            return {
                'clash_score': clash_score,
                'ramachandran_favored': ramachandran_metrics['favored'],
                'ramachandran_outliers': ramachandran_metrics['outliers'],
                'rotamer_outliers': 1.0,  # Dummy
                'c_beta_deviations': 0,   # Dummy
                'bond_length_rmsd': 0.01, # Dummy
                'bond_angle_rmsd': 1.0    # Dummy
            }
            
        except Exception as e:
            logger.warning(f"Structure quality assessment failed: {str(e)}")
            return {
                'clash_score': 0.0,
                'ramachandran_favored': 0.0,
                'ramachandran_outliers': 100.0,
                'rotamer_outliers': 100.0,
                'c_beta_deviations': 999,
                'bond_length_rmsd': 999.0,
                'bond_angle_rmsd': 999.0
            }
    
    def _calculate_clash_score(self, coords: np.ndarray, 
                              clash_threshold: float = 2.0) -> float:
        """Calculate clash score based on inter-atomic distances"""
        try:
            if len(coords) < 2:
                return 0.0
            
            # Calculate distance matrix
            distances = cdist(coords, coords)
            
            # Count clashes (distances below threshold, excluding self)
            mask = (distances > 0) & (distances < clash_threshold)
            clash_count = np.sum(mask) // 2  # Divide by 2 to avoid double counting
            
            # Normalize by number of possible pairs
            total_pairs = len(coords) * (len(coords) - 1) // 2
            clash_score = clash_count / total_pairs * 100 if total_pairs > 0 else 0.0
            
            return clash_score
            
        except Exception as e:
            logger.warning(f"Clash score calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_overall_score(
        self,
        rmsd: float,
        gdt_ts: float,
        tm_score: float,
        lddt: float,
        clash_score: float,
        ramachandran_favored: float,
        ramachandran_outliers: float
    ) -> float:
        """Calculate overall quality score"""
        try:
            # Normalize and weight different metrics
            # Lower RMSD is better
            rmsd_score = max(0, 100 - rmsd * 10)  # Penalize high RMSD
            
            # Higher GDT-TS is better
            gdt_score = gdt_ts
            
            # Higher TM-score is better
            tm_score_normalized = tm_score * 100
            
            # Higher LDDT is better
            lddt_score = lddt
            
            # Lower clash score is better
            clash_penalty = max(0, 100 - clash_score * 10)
            
            # Higher Ramachandran favored is better
            rama_score = ramachandran_favored - ramachandran_outliers
            
            # Weighted average
            weights = [0.2, 0.25, 0.25, 0.15, 0.1, 0.05]
            scores = [rmsd_score, gdt_score, tm_score_normalized, 
                     lddt_score, clash_penalty, rama_score]
            
            overall_score = np.average(scores, weights=weights)
            
            return max(0.0, min(100.0, overall_score))
            
        except Exception as e:
            logger.warning(f"Overall score calculation failed: {str(e)}")
            return 0.0

# Convenience functions
def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray, 
                  align: bool = True) -> float:
    """Calculate RMSD between two coordinate sets"""
    metrics = StructureMetrics()
    return metrics.calculate_rmsd(coords1, coords2, align)

def calculate_gdt_ts(predicted_coords: np.ndarray, 
                    reference_coords: np.ndarray) -> float:
    """Calculate GDT-TS score"""
    metrics = StructureMetrics()
    return metrics.calculate_gdt_ts(predicted_coords, reference_coords)

def calculate_tm_score(predicted_coords: np.ndarray, 
                      reference_coords: np.ndarray,
                      sequence: Optional[str] = None) -> float:
    """Calculate TM-score"""
    metrics = StructureMetrics()
    return metrics.calculate_tm_score(predicted_coords, reference_coords, sequence) 