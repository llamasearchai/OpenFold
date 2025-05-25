"""
Comprehensive Tests for Core Modules

Tests for predictors, data processing, validation, metrics, and utilities.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import asyncio

# Core imports
from core.models.predictor import (
    AdvancedPredictor, PredictionConfig, ModelType, PredictionMode,
    create_predictor, create_fast_predictor
)
from core.data.processor import (
    DataPipeline, ProcessingConfig, SequenceProcessor, MSAGenerator,
    TemplateSearcher, FeatureExtractor
)
from core.data.validators import (
    SequenceValidator, StructureValidator, ValidationLevel,
    validate_sequence, validate_structure
)
from core.data.loaders import (
    DataLoader, FASTALoader, PDBLoader, MSALoader,
    LoaderConfig, FileFormat, load_fasta
)
from core.utils.metrics import (
    StructureMetrics, calculate_rmsd, calculate_gdt_ts, calculate_tm_score
)

class TestAdvancedPredictor:
    """Test the AdvancedPredictor class"""
    
    def test_predictor_initialization(self):
        """Test predictor initialization"""
        config = PredictionConfig(
            model_type=ModelType.ALPHAFOLD3,
            mode=PredictionMode.FAST
        )
        predictor = AdvancedPredictor(config)
        
        assert predictor.config.model_type == ModelType.ALPHAFOLD3
        assert predictor.config.mode == PredictionMode.FAST
        assert predictor.predictor is None  # Not initialized yet
    
    def test_predictor_factory_functions(self):
        """Test predictor factory functions"""
        # Test create_predictor
        predictor = create_predictor(PredictionConfig())
        assert isinstance(predictor, AdvancedPredictor)
        
        # Test create_fast_predictor
        fast_predictor = create_fast_predictor()
        assert fast_predictor.config.mode == PredictionMode.FAST
        assert fast_predictor.config.model_type == ModelType.ESM2
    
    @pytest.mark.asyncio
    async def test_async_prediction(self):
        """Test async prediction"""
        config = PredictionConfig(model_type=ModelType.ESM2)
        predictor = AdvancedPredictor(config)
        
        # Mock the predictor
        with patch.object(predictor, 'initialize') as mock_init:
            with patch.object(predictor, 'predict') as mock_predict:
                mock_predict.return_value = Mock(success=True)
                
                sequence = "MKLLVLGLGAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"
                
                result = await predictor.predict_async(sequence)
                
                mock_init.assert_called_once()
                mock_predict.assert_called_once()
                assert result.success

class TestSequenceProcessor:
    """Test the SequenceProcessor class"""
    
    def test_sequence_cleaning(self):
        """Test sequence cleaning"""
        config = ProcessingConfig()
        processor = SequenceProcessor(config)
        
        # Test basic cleaning
        dirty_seq = "  M K L L V L  \n\t"
        cleaned = processor._clean_sequence(dirty_seq)
        assert cleaned == "MKLLVL"
        
        # Test ambiguous amino acid replacement
        ambiguous_seq = "MKBZJUO"
        cleaned = processor._clean_sequence(ambiguous_seq)
        assert cleaned == "MKNQLK"
    
    def test_sequence_validation(self):
        """Test sequence validation"""
        config = ProcessingConfig(min_length=5, max_length=100)
        processor = SequenceProcessor(config)
        
        # Valid sequence
        valid_seq = "MKLLVLGLGAGVGK"
        validation = processor._validate_sequence(valid_seq)
        assert validation['valid'] == True
        assert len(validation['errors']) == 0
        
        # Too short sequence
        short_seq = "MKL"
        validation = processor._validate_sequence(short_seq)
        assert validation['valid'] == False
        assert any("too short" in error for error in validation['errors'])
    
    def test_sequence_features(self):
        """Test sequence feature extraction"""
        config = ProcessingConfig()
        processor = SequenceProcessor(config)
        
        sequence = "MKLLVLGLGAGVGK"
        features = processor._extract_sequence_features(sequence)
        
        assert 'aa_composition' in features
        assert 'hydrophobic_fraction' in features
        assert 'molecular_weight' in features
        assert features['hydrophobic_fraction'] > 0
        assert features['molecular_weight'] > 0

class TestDataPipeline:
    """Test the DataPipeline class"""
    
    @pytest.mark.asyncio
    async def test_pipeline_processing(self):
        """Test complete data pipeline"""
        config = ProcessingConfig(use_precomputed_msas=False, use_templates=False)
        pipeline = DataPipeline(config)
        
        sequence = "MKLLVLGLGAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"
        
        # Mock the MSA and template components
        with patch.object(pipeline.msa_generator, 'generate_msa') as mock_msa:
            with patch.object(pipeline.template_searcher, 'search_templates') as mock_templates:
                mock_msa.return_value = Mock()
                mock_templates.return_value = []
                
                features = await pipeline.process(sequence)
                
                assert features.sequence == sequence
                assert features.aatype is not None
                assert len(features.aatype) == len(sequence)

class TestSequenceValidator:
    """Test the SequenceValidator class"""
    
    def test_basic_validation(self):
        """Test basic sequence validation"""
        validator = SequenceValidator(min_length=10, max_length=1000)
        
        # Valid sequence
        valid_seq = "MKLLVLGLGAGVGKSALTIQLIQ"
        result = validator.validate(valid_seq)
        
        assert result.valid == True
        assert result.score > 50
        assert len(result.errors) == 0
    
    def test_invalid_sequence(self):
        """Test invalid sequence validation"""
        validator = SequenceValidator(min_length=10, max_length=1000)
        
        # Too short sequence
        short_seq = "MKL"
        result = validator.validate(short_seq)
        
        assert result.valid == False
        assert len(result.errors) > 0
        assert any(error.code == "SEQ_TOO_SHORT" for error in result.errors)
    
    def test_sequence_analysis(self):
        """Test sequence analysis features"""
        validator = SequenceValidator()
        
        sequence = "MKLLVLGLGAGVGKSALTIQLIQ"
        result = validator.validate(sequence)
        
        assert 'amino_acid_composition' in result.metadata
        assert 'hydrophobic_fraction' in result.metadata
        assert 'complexity' in result.metadata

class TestDataLoader:
    """Test the DataLoader class"""
    
    def test_fasta_loading(self):
        """Test FASTA file loading"""
        # Create temporary FASTA file
        fasta_content = ">test_seq\nMKLLVLGLGAGVGKSALTIQLIQ\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            temp_path = f.name
        
        try:
            config = LoaderConfig()
            loader = FASTALoader(config)
            
            sequences = loader.load_sequences(temp_path)
            
            assert len(sequences) == 1
            assert sequences[0].id == "test_seq"
            assert str(sequences[0].seq) == "MKLLVLGLGAGVGKSALTIQLIQ"
        
        finally:
            os.unlink(temp_path)
    
    def test_format_detection(self):
        """Test file format auto-detection"""
        config = LoaderConfig()
        loader = DataLoader(config)
        
        # Test FASTA detection
        fasta_content = ">test\nMKLLVL\n"
        format_detected = loader._detect_format(fasta_content, "test.fasta")
        assert format_detected == FileFormat.FASTA
        
        # Test PDB detection
        pdb_content = "HEADER    TEST\nATOM      1  N   ALA A   1      20.154  16.967  10.000  1.00 20.00           N\n"
        format_detected = loader._detect_format(pdb_content, "test.pdb")
        assert format_detected == FileFormat.PDB

class TestStructureMetrics:
    """Test the StructureMetrics class"""
    
    def test_rmsd_calculation(self):
        """Test RMSD calculation"""
        metrics = StructureMetrics()
        
        # Create test coordinates
        coords1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        coords2 = np.array([[0.1, 0, 0], [1.1, 0, 0], [0.1, 1, 0]])
        
        rmsd = metrics.calculate_rmsd(coords1, coords2, align=False)
        
        assert rmsd > 0
        assert rmsd < 1.0  # Should be small for similar structures
    
    def test_structure_alignment(self):
        """Test structure alignment"""
        metrics = StructureMetrics()
        
        # Create test coordinates (one rotated version of the other)
        coords1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        # Rotate by 90 degrees around z-axis
        coords2 = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0], [-1, 1, 0]])
        
        alignment = metrics.align_structures(coords1, coords2)
        
        assert alignment.rmsd < 0.1  # Should align perfectly
        assert alignment.alignment_length == 4
        assert alignment.rotation_matrix.shape == (3, 3)
    
    def test_gdt_ts_calculation(self):
        """Test GDT-TS calculation"""
        metrics = StructureMetrics()
        
        # Create test coordinates
        coords1 = np.random.randn(50, 3)
        coords2 = coords1 + np.random.randn(50, 3) * 0.5  # Add small noise
        
        gdt_ts = metrics.calculate_gdt_ts(coords1, coords2)
        
        assert 0 <= gdt_ts <= 100
        assert gdt_ts > 50  # Should be high for similar structures
    
    def test_tm_score_calculation(self):
        """Test TM-score calculation"""
        metrics = StructureMetrics()
        
        # Create test coordinates
        coords1 = np.random.randn(50, 3)
        coords2 = coords1 + np.random.randn(50, 3) * 0.5
        
        tm_score = metrics.calculate_tm_score(coords1, coords2)
        
        assert 0 <= tm_score <= 1
        assert tm_score > 0.5  # Should be high for similar structures

class TestMSAGenerator:
    """Test the MSAGenerator class"""
    
    @pytest.mark.asyncio
    async def test_msa_generation(self):
        """Test MSA generation"""
        config = ProcessingConfig()
        generator = MSAGenerator(config)
        
        sequence = "MKLLVLGLGAGVGKSALTIQLIQ"
        
        # Mock external MSA tools
        with patch.object(generator, '_run_blast_search') as mock_blast:
            mock_blast.return_value = generator._generate_dummy_msa(sequence, "test")
            
            msa_result = await generator.generate_msa(sequence)
            
            assert msa_result.num_sequences >= 1
            assert msa_result.query_sequence == sequence
            assert len(msa_result.sequences) > 0

class TestTemplateSearcher:
    """Test the TemplateSearcher class"""
    
    @pytest.mark.asyncio
    async def test_template_search(self):
        """Test template structure search"""
        config = ProcessingConfig()
        searcher = TemplateSearcher(config)
        
        sequence = "MKLLVLGLGAGVGKSALTIQLIQ"
        msa_result = Mock()
        
        # Mock PDB search
        with patch.object(searcher, '_search_pdb_templates') as mock_search:
            mock_search.return_value = [{'identifier': '1ABC'}]
            
            templates = await searcher.search_templates(sequence, msa_result)
            
            assert isinstance(templates, list)

class TestFeatureExtractor:
    """Test the FeatureExtractor class"""
    
    def test_sequence_features(self):
        """Test sequence feature extraction"""
        config = ProcessingConfig()
        extractor = FeatureExtractor(config)
        
        sequence = "MKLLVLGLGAGVGKSALTIQLIQ"
        features = extractor._extract_sequence_features(sequence)
        
        assert 'aatype' in features
        assert 'residue_index' in features
        assert 'seq_length' in features
        assert len(features['aatype']) == len(sequence)
    
    def test_msa_features(self):
        """Test MSA feature extraction"""
        config = ProcessingConfig()
        extractor = FeatureExtractor(config)
        
        # Create mock MSA result
        msa_result = Mock()
        msa_result.sequences = ["MKLLVL", "MKLLAL", "MKLLIL"]
        msa_result.species = ["Species1", "Species2", "Species3"]
        
        features = extractor._extract_msa_features(msa_result)
        
        assert 'msa' in features
        assert 'deletion_matrix' in features
        assert features['msa'].shape[0] == 3  # Number of sequences

# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_prediction(self):
        """Test end-to-end prediction workflow"""
        # Create predictor
        config = PredictionConfig(model_type=ModelType.ESM2, mode=PredictionMode.FAST)
        predictor = create_predictor(config)
        
        sequence = "MKLLVLGLGAGVGKSALTIQLIQ"
        
        # Mock the actual prediction
        with patch.object(predictor, 'initialize'):
            with patch.object(predictor, 'predict') as mock_predict:
                mock_result = Mock()
                mock_result.success = True
                mock_result.structure = "MOCK_PDB_CONTENT"
                mock_result.confidence.overall = 0.8
                mock_predict.return_value = mock_result
                
                result = await predictor.predict_async(sequence)
                
                assert result.success == True
                assert result.confidence.overall == 0.8
    
    def test_validation_pipeline(self):
        """Test validation pipeline"""
        # Test sequence validation
        sequence = "MKLLVLGLGAGVGKSALTIQLIQ"
        result = validate_sequence(sequence)
        
        assert result.valid == True
        assert result.score > 0
        
        # Test structure validation
        pdb_content = """HEADER    TEST
ATOM      1  N   ALA A   1      20.154  16.967  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      21.618  16.967  10.000  1.00 20.00           C
END"""
        
        structure_result = validate_structure(pdb_content)
        assert structure_result.valid == True

# Fixtures
@pytest.fixture
def sample_sequence():
    """Sample protein sequence for testing"""
    return "MKLLVLGLGAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"

@pytest.fixture
def sample_coordinates():
    """Sample coordinates for testing"""
    return np.random.randn(100, 3)

@pytest.fixture
def temp_fasta_file():
    """Temporary FASTA file for testing"""
    content = ">test_protein\nMKLLVLGLGAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS\n"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    
    os.unlink(temp_path)

# Performance tests
class TestPerformance:
    """Performance tests for critical components"""
    
    def test_rmsd_performance(self):
        """Test RMSD calculation performance"""
        import time
        
        metrics = StructureMetrics()
        
        # Large coordinate sets
        coords1 = np.random.randn(1000, 3)
        coords2 = np.random.randn(1000, 3)
        
        start_time = time.time()
        rmsd = metrics.calculate_rmsd(coords1, coords2)
        end_time = time.time()
        
        assert rmsd > 0
        assert (end_time - start_time) < 1.0  # Should complete in under 1 second
    
    def test_sequence_validation_performance(self):
        """Test sequence validation performance"""
        import time
        
        validator = SequenceValidator()
        
        # Long sequence
        sequence = "MKLLVLGLGAGVGKSALTIQLIQ" * 100  # 2300 residues
        
        start_time = time.time()
        result = validator.validate(sequence)
        end_time = time.time()
        
        assert result.valid == True
        assert (end_time - start_time) < 1.0  # Should complete in under 1 second

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 