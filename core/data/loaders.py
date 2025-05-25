"""
Data Loading Module

Comprehensive data loaders for various biomolecule file formats including
PDB, FASTA, MSA, and other structural biology formats.
"""

import io
import gzip
import bz2
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from dataclasses import dataclass
from enum import Enum
import logging
import requests
import tempfile
import shutil
from urllib.parse import urlparse
import json

# Bioinformatics libraries
from Bio import SeqIO, PDB
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from Bio.PDB.Structure import Structure
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class FileFormat(str, Enum):
    """Supported file formats"""
    FASTA = "fasta"
    PDB = "pdb"
    MMCIF = "mmcif"
    MSA = "msa"
    A3M = "a3m"
    STOCKHOLM = "stockholm"
    CLUSTAL = "clustal"
    JSON = "json"
    CSV = "csv"
    TSV = "tsv"
    AUTO = "auto"

class CompressionType(str, Enum):
    """Supported compression types"""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    ZIP = "zip"
    AUTO = "auto"

@dataclass
class LoaderConfig:
    """Configuration for data loaders"""
    max_file_size: int = 100 * 1024 * 1024  # 100 MB
    timeout: int = 30
    cache_dir: Optional[str] = None
    validate_data: bool = True
    auto_decompress: bool = True
    encoding: str = "utf-8"

@dataclass
class LoadedData:
    """Container for loaded data"""
    data: Any
    metadata: Dict[str, Any]
    format: FileFormat
    source: str
    
class DataLoader:
    """Base data loader class"""
    
    def __init__(self, config: LoaderConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir) if config.cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self, source: Union[str, Path, io.IOBase], 
             format: FileFormat = FileFormat.AUTO,
             compression: CompressionType = CompressionType.AUTO) -> LoadedData:
        """Load data from various sources"""
        try:
            # Determine source type and get content
            content, metadata = self._get_content(source, compression)
            
            # Auto-detect format if needed
            if format == FileFormat.AUTO:
                format = self._detect_format(content, source)
            
            # Parse content based on format
            data = self._parse_content(content, format)
            
            # Validate if requested
            if self.config.validate_data:
                self._validate_data(data, format)
            
            return LoadedData(
                data=data,
                metadata=metadata,
                format=format,
                source=str(source)
            )
            
        except Exception as e:
            logger.error(f"Failed to load data from {source}: {str(e)}")
            raise
    
    def _get_content(self, source: Union[str, Path, io.IOBase], 
                    compression: CompressionType) -> Tuple[str, Dict[str, Any]]:
        """Get content from various sources"""
        metadata = {}
        
        if isinstance(source, io.IOBase):
            # File-like object
            content = source.read()
            if isinstance(content, bytes):
                content = content.decode(self.config.encoding)
            metadata['source_type'] = 'file_object'
            
        elif isinstance(source, (str, Path)):
            source_str = str(source)
            
            if source_str.startswith(('http://', 'https://')):
                # URL
                content, url_metadata = self._download_from_url(source_str)
                metadata.update(url_metadata)
                metadata['source_type'] = 'url'
                
            elif source_str.startswith(('ftp://', 'ftps://')):
                # FTP
                content = self._download_from_ftp(source_str)
                metadata['source_type'] = 'ftp'
                
            else:
                # Local file
                file_path = Path(source)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                content = self._read_file(file_path, compression)
                metadata['source_type'] = 'local_file'
                metadata['file_size'] = file_path.stat().st_size
                metadata['file_path'] = str(file_path)
        
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
        
        return content, metadata
    
    def _download_from_url(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """Download content from URL"""
        try:
            response = requests.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.config.max_file_size:
                raise ValueError(f"File too large: {content_length} bytes")
            
            content = response.text
            metadata = {
                'url': url,
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type'),
                'content_length': content_length
            }
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Failed to download from {url}: {str(e)}")
            raise
    
    def _download_from_ftp(self, url: str) -> str:
        """Download content from FTP"""
        # Simplified FTP implementation
        # In practice, you'd use ftplib
        raise NotImplementedError("FTP download not implemented")
    
    def _read_file(self, file_path: Path, compression: CompressionType) -> str:
        """Read file with optional decompression"""
        # Auto-detect compression
        if compression == CompressionType.AUTO:
            compression = self._detect_compression(file_path)
        
        # Read file based on compression
        if compression == CompressionType.GZIP:
            with gzip.open(file_path, 'rt', encoding=self.config.encoding) as f:
                return f.read()
        elif compression == CompressionType.BZIP2:
            with bz2.open(file_path, 'rt', encoding=self.config.encoding) as f:
                return f.read()
        elif compression == CompressionType.ZIP:
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Read first file in zip
                names = zf.namelist()
                if not names:
                    raise ValueError("Empty ZIP file")
                with zf.open(names[0]) as f:
                    return f.read().decode(self.config.encoding)
        else:
            # Uncompressed
            with open(file_path, 'r', encoding=self.config.encoding) as f:
                return f.read()
    
    def _detect_compression(self, file_path: Path) -> CompressionType:
        """Auto-detect compression type"""
        suffix = file_path.suffix.lower()
        
        if suffix in ['.gz', '.gzip']:
            return CompressionType.GZIP
        elif suffix in ['.bz2', '.bzip2']:
            return CompressionType.BZIP2
        elif suffix in ['.zip']:
            return CompressionType.ZIP
        else:
            return CompressionType.NONE
    
    def _detect_format(self, content: str, source: Union[str, Path, io.IOBase]) -> FileFormat:
        """Auto-detect file format"""
        # Check file extension first
        if isinstance(source, (str, Path)):
            source_str = str(source).lower()
            
            if any(ext in source_str for ext in ['.pdb', '.ent']):
                return FileFormat.PDB
            elif any(ext in source_str for ext in ['.cif', '.mmcif']):
                return FileFormat.MMCIF
            elif any(ext in source_str for ext in ['.fasta', '.fa', '.fas', '.faa']):
                return FileFormat.FASTA
            elif '.a3m' in source_str:
                return FileFormat.A3M
            elif '.sto' in source_str:
                return FileFormat.STOCKHOLM
            elif '.aln' in source_str:
                return FileFormat.CLUSTAL
            elif '.json' in source_str:
                return FileFormat.JSON
            elif '.csv' in source_str:
                return FileFormat.CSV
            elif '.tsv' in source_str:
                return FileFormat.TSV
        
        # Check content patterns
        content_lines = content.strip().split('\n')
        if not content_lines:
            raise ValueError("Empty content")
        
        first_line = content_lines[0].strip()
        
        # PDB format
        if first_line.startswith(('HEADER', 'TITLE', 'ATOM', 'HETATM')):
            return FileFormat.PDB
        
        # mmCIF format
        if first_line.startswith('data_') or 'loop_' in content:
            return FileFormat.MMCIF
        
        # FASTA format
        if first_line.startswith('>'):
            return FileFormat.FASTA
        
        # A3M format
        if first_line.startswith('#') and 'A3M' in first_line.upper():
            return FileFormat.A3M
        
        # Stockholm format
        if first_line.startswith('# STOCKHOLM'):
            return FileFormat.STOCKHOLM
        
        # Clustal format
        if 'CLUSTAL' in first_line.upper():
            return FileFormat.CLUSTAL
        
        # JSON format
        if first_line.startswith(('{', '[')):
            return FileFormat.JSON
        
        # Default to FASTA for sequence-like content
        if all(c.upper() in 'ACDEFGHIKLMNPQRSTVWYXBZJUO*-\n\r\t ' for c in content):
            return FileFormat.FASTA
        
        raise ValueError("Could not detect file format")
    
    def _parse_content(self, content: str, format: FileFormat) -> Any:
        """Parse content based on format"""
        if format == FileFormat.FASTA:
            return self._parse_fasta(content)
        elif format == FileFormat.PDB:
            return self._parse_pdb(content)
        elif format == FileFormat.MMCIF:
            return self._parse_mmcif(content)
        elif format == FileFormat.MSA:
            return self._parse_msa(content)
        elif format == FileFormat.A3M:
            return self._parse_a3m(content)
        elif format == FileFormat.STOCKHOLM:
            return self._parse_stockholm(content)
        elif format == FileFormat.CLUSTAL:
            return self._parse_clustal(content)
        elif format == FileFormat.JSON:
            return self._parse_json(content)
        elif format == FileFormat.CSV:
            return self._parse_csv(content)
        elif format == FileFormat.TSV:
            return self._parse_tsv(content)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _parse_fasta(self, content: str) -> List[SeqRecord]:
        """Parse FASTA format"""
        records = []
        with io.StringIO(content) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                records.append(record)
        return records
    
    def _parse_pdb(self, content: str) -> Structure:
        """Parse PDB format"""
        parser = PDBParser(QUIET=True)
        with io.StringIO(content) as handle:
            structure = parser.get_structure("structure", handle)
        return structure
    
    def _parse_mmcif(self, content: str) -> Structure:
        """Parse mmCIF format"""
        parser = MMCIFParser(QUIET=True)
        with io.StringIO(content) as handle:
            structure = parser.get_structure("structure", handle)
        return structure
    
    def _parse_msa(self, content: str) -> List[SeqRecord]:
        """Parse generic MSA format"""
        # Try different MSA formats
        for fmt in ["fasta", "clustal", "stockholm"]:
            try:
                with io.StringIO(content) as handle:
                    records = list(SeqIO.parse(handle, fmt))
                    if records:
                        return records
            except:
                continue
        
        raise ValueError("Could not parse MSA format")
    
    def _parse_a3m(self, content: str) -> List[SeqRecord]:
        """Parse A3M format"""
        records = []
        lines = content.strip().split('\n')
        
        current_id = None
        current_seq = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('>'):
                # Save previous record
                if current_id and current_seq:
                    seq_str = ''.join(current_seq)
                    records.append(SeqRecord(Seq(seq_str), id=current_id))
                
                # Start new record
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        
        # Save last record
        if current_id and current_seq:
            seq_str = ''.join(current_seq)
            records.append(SeqRecord(Seq(seq_str), id=current_id))
        
        return records
    
    def _parse_stockholm(self, content: str) -> List[SeqRecord]:
        """Parse Stockholm format"""
        with io.StringIO(content) as handle:
            records = list(SeqIO.parse(handle, "stockholm"))
        return records
    
    def _parse_clustal(self, content: str) -> List[SeqRecord]:
        """Parse Clustal format"""
        with io.StringIO(content) as handle:
            records = list(SeqIO.parse(handle, "clustal"))
        return records
    
    def _parse_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON format"""
        return json.loads(content)
    
    def _parse_csv(self, content: str) -> pd.DataFrame:
        """Parse CSV format"""
        with io.StringIO(content) as handle:
            return pd.read_csv(handle)
    
    def _parse_tsv(self, content: str) -> pd.DataFrame:
        """Parse TSV format"""
        with io.StringIO(content) as handle:
            return pd.read_csv(handle, sep='\t')
    
    def _validate_data(self, data: Any, format: FileFormat):
        """Validate loaded data"""
        if format == FileFormat.FASTA:
            if not isinstance(data, list) or not all(isinstance(r, SeqRecord) for r in data):
                raise ValueError("Invalid FASTA data")
        elif format in [FileFormat.PDB, FileFormat.MMCIF]:
            if not isinstance(data, Structure):
                raise ValueError("Invalid structure data")
        # Add more validation as needed

class FASTALoader(DataLoader):
    """Specialized FASTA loader"""
    
    def load_sequences(self, source: Union[str, Path, io.IOBase]) -> List[SeqRecord]:
        """Load FASTA sequences"""
        result = self.load(source, FileFormat.FASTA)
        return result.data
    
    def load_single_sequence(self, source: Union[str, Path, io.IOBase]) -> SeqRecord:
        """Load single FASTA sequence"""
        sequences = self.load_sequences(source)
        if not sequences:
            raise ValueError("No sequences found")
        if len(sequences) > 1:
            logger.warning(f"Multiple sequences found, returning first one")
        return sequences[0]

class PDBLoader(DataLoader):
    """Specialized PDB loader"""
    
    def load_structure(self, source: Union[str, Path, io.IOBase]) -> Structure:
        """Load PDB structure"""
        result = self.load(source, FileFormat.PDB)
        return result.data
    
    def load_from_pdb_id(self, pdb_id: str, chain: Optional[str] = None) -> Structure:
        """Load structure from PDB ID"""
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        structure = self.load_structure(url)
        
        if chain:
            # Filter to specific chain
            for model in structure:
                chains_to_remove = [c for c in model if c.id != chain.upper()]
                for c in chains_to_remove:
                    model.detach_child(c.id)
        
        return structure
    
    def extract_sequence(self, structure: Structure, chain_id: Optional[str] = None) -> str:
        """Extract sequence from PDB structure"""
        sequences = []
        
        for model in structure:
            for chain in model:
                if chain_id and chain.id != chain_id:
                    continue
                
                sequence = ""
                for residue in chain:
                    if residue.id[0] == ' ':  # Standard residue
                        resname = residue.resname
                        # Convert 3-letter to 1-letter code
                        aa_code = self._three_to_one(resname)
                        sequence += aa_code
                
                if sequence:
                    sequences.append(sequence)
        
        return sequences[0] if sequences else ""
    
    def _three_to_one(self, three_letter: str) -> str:
        """Convert 3-letter amino acid code to 1-letter"""
        conversion = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        return conversion.get(three_letter, 'X')

class MSALoader(DataLoader):
    """Specialized MSA loader"""
    
    def load_alignment(self, source: Union[str, Path, io.IOBase], 
                      format: FileFormat = FileFormat.AUTO) -> List[SeqRecord]:
        """Load multiple sequence alignment"""
        if format == FileFormat.AUTO:
            # Try to detect MSA format
            content, _ = self._get_content(source, CompressionType.AUTO)
            format = self._detect_msa_format(content)
        
        result = self.load(source, format)
        return result.data
    
    def _detect_msa_format(self, content: str) -> FileFormat:
        """Detect MSA format"""
        first_line = content.strip().split('\n')[0]
        
        if first_line.startswith('# STOCKHOLM'):
            return FileFormat.STOCKHOLM
        elif 'CLUSTAL' in first_line.upper():
            return FileFormat.CLUSTAL
        elif first_line.startswith('#') and 'A3M' in first_line.upper():
            return FileFormat.A3M
        elif first_line.startswith('>'):
            return FileFormat.FASTA
        else:
            return FileFormat.FASTA  # Default

# Factory functions
def create_loader_config(**kwargs) -> LoaderConfig:
    """Create loader configuration"""
    return LoaderConfig(**kwargs)

def create_data_loader(config: Optional[LoaderConfig] = None) -> DataLoader:
    """Create generic data loader"""
    if config is None:
        config = LoaderConfig()
    return DataLoader(config)

def create_fasta_loader(config: Optional[LoaderConfig] = None) -> FASTALoader:
    """Create FASTA loader"""
    if config is None:
        config = LoaderConfig()
    return FASTALoader(config)

def create_pdb_loader(config: Optional[LoaderConfig] = None) -> PDBLoader:
    """Create PDB loader"""
    if config is None:
        config = LoaderConfig()
    return PDBLoader(config)

def create_msa_loader(config: Optional[LoaderConfig] = None) -> MSALoader:
    """Create MSA loader"""
    if config is None:
        config = LoaderConfig()
    return MSALoader(config)

# Convenience functions
def load_fasta(source: Union[str, Path, io.IOBase]) -> List[SeqRecord]:
    """Quick FASTA loading"""
    loader = create_fasta_loader()
    return loader.load_sequences(source)

def load_pdb(source: Union[str, Path, io.IOBase]) -> Structure:
    """Quick PDB loading"""
    loader = create_pdb_loader()
    return loader.load_structure(source)

def load_pdb_from_id(pdb_id: str, chain: Optional[str] = None) -> Structure:
    """Quick PDB loading from ID"""
    loader = create_pdb_loader()
    return loader.load_from_pdb_id(pdb_id, chain)

def load_msa(source: Union[str, Path, io.IOBase]) -> List[SeqRecord]:
    """Quick MSA loading"""
    loader = create_msa_loader()
    return loader.load_alignment(source) 