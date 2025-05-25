#!/usr/bin/env python3
"""
OpenFold Complete System Runner

Comprehensive startup script for the OpenFold biomolecule structure prediction platform.
Supports multiple execution modes: API server, CLI prediction, batch processing, and interactive mode.
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Core imports
from core import (
    create_predictor, create_fast_predictor, create_accurate_predictor,
    PredictionConfig, ModelType, PredictionMode
)
from core.data import (
    create_data_pipeline, ProcessingConfig,
    validate_sequence, load_fasta
)
from core.agents import create_structure_agent
from api.main import app
from api.config import get_settings

# Utilities
import uvicorn
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

console = Console()

class OpenFoldRunner:
    """Main runner for the OpenFold system"""
    
    def __init__(self):
        self.setup_logging()
        self.settings = get_settings()
        self.predictor = None
        self.data_pipeline = None
        self.structure_agent = None
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('openfold.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_components(self, mode: str = "fast"):
        """Initialize core components"""
        console.print("[bold blue]Initializing OpenFold components...[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Initialize predictor
            task1 = progress.add_task("Loading prediction models...", total=None)
            if mode == "fast":
                self.predictor = create_fast_predictor()
            elif mode == "accurate":
                self.predictor = create_accurate_predictor()
            else:
                config = PredictionConfig(model_type=ModelType.ESM2)
                self.predictor = create_predictor(config)
            progress.update(task1, completed=True)
            
            # Initialize data pipeline
            task2 = progress.add_task("Setting up data pipeline...", total=None)
            processing_config = ProcessingConfig()
            self.data_pipeline = create_data_pipeline(processing_config)
            progress.update(task2, completed=True)
            
            # Initialize AI agent
            task3 = progress.add_task("Loading AI agents...", total=None)
            self.structure_agent = create_structure_agent()
            progress.update(task3, completed=True)
        
        console.print("[bold green]All components initialized successfully![/bold green]")
    
    async def predict_structure(self, sequence: str, sequence_id: str = "query") -> Dict[str, Any]:
        """Predict structure for a single sequence"""
        try:
            # Validate sequence
            validation_result = validate_sequence(sequence)
            if not validation_result.valid:
                raise ValueError(f"Invalid sequence: {validation_result.errors}")
            
            # Process data
            features = await self.data_pipeline.process(sequence, sequence_id)
            
            # Predict structure
            result = await self.predictor.predict_async(sequence)
            
            # Analyze with AI agent
            if result.success and self.structure_agent:
                analysis = await self.structure_agent.analyze_structure_async(
                    result.structure, sequence
                )
                result.metadata['ai_analysis'] = analysis
            
            return {
                'success': result.success,
                'sequence_id': sequence_id,
                'sequence': sequence,
                'structure': result.structure,
                'confidence': {
                    'overall': result.confidence.overall,
                    'per_residue': result.confidence.per_residue
                },
                'quality': {
                    'clash_score': result.quality.clash_score,
                    'ramachandran_favored': result.quality.ramachandran_favored
                },
                'processing_time': result.processing_time,
                'model_used': result.model_used.value,
                'metadata': result.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {sequence_id}: {str(e)}")
            return {
                'success': False,
                'sequence_id': sequence_id,
                'error': str(e)
            }
    
    async def batch_predict(self, input_file: str, output_dir: str) -> List[Dict[str, Any]]:
        """Batch prediction from FASTA file"""
        console.print(f"[bold blue]Processing batch file: {input_file}[/bold blue]")
        
        # Load sequences
        sequences = load_fasta(input_file)
        results = []
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with Progress(console=console) as progress:
            task = progress.add_task("Predicting structures...", total=len(sequences))
            
            for seq_record in sequences:
                sequence = str(seq_record.seq)
                sequence_id = seq_record.id
                
                # Predict structure
                result = await self.predict_structure(sequence, sequence_id)
                results.append(result)
                
                # Save individual result
                if result['success']:
                    # Save PDB file
                    pdb_file = output_path / f"{sequence_id}.pdb"
                    with open(pdb_file, 'w') as f:
                        f.write(result['structure'])
                    
                    # Save metadata
                    json_file = output_path / f"{sequence_id}.json"
                    with open(json_file, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                
                progress.update(task, advance=1)
        
        # Save summary
        summary_file = output_path / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        console.print(f"[bold green]Batch processing complete! Results saved to {output_dir}[/bold green]")
        return results
    
    def run_api_server(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Run the FastAPI server"""
        console.print(Panel.fit(
            "[bold blue]Starting OpenFold API Server[/bold blue]\n\n"
            f"Server URL: http://{host}:{port}\n"
            f"API Documentation: http://{host}:{port}/docs\n"
            f"Auto-reload: {'Enabled' if reload else 'Disabled'}",
            title="OpenFold API",
            border_style="blue"
        ))
        
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    
    def run_interactive_mode(self):
        """Run interactive CLI mode"""
        console.print(Panel.fit(
            "[bold green]OpenFold Interactive Mode[/bold green]\n\n"
            "Enter protein sequences for structure prediction.\n"
            "Type 'help' for commands, 'quit' to exit.",
            title="Interactive Mode",
            border_style="green"
        ))
        
        while True:
            try:
                user_input = console.input("\n[bold cyan]OpenFold>[/bold cyan] ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                
                elif user_input.lower() == 'status':
                    self.show_status()
                
                elif user_input.startswith('>'):
                    # FASTA format input
                    lines = [user_input]
                    while True:
                        line = console.input().strip()
                        if not line:
                            break
                        lines.append(line)
                    
                    # Parse FASTA
                    fasta_content = '\n'.join(lines)
                    # Process FASTA content...
                    console.print("[yellow]FASTA processing not implemented in interactive mode[/yellow]")
                
                elif len(user_input) > 10 and all(c.upper() in 'ACDEFGHIKLMNPQRSTVWYXBZJUO' for c in user_input):
                    # Direct sequence input
                    asyncio.run(self.process_interactive_sequence(user_input))
                
                else:
                    console.print("[red]Invalid input. Type 'help' for usage information.[/red]")
            
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'quit' to exit.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
    
    async def process_interactive_sequence(self, sequence: str):
        """Process a sequence in interactive mode"""
        with console.status("[bold green]Predicting structure..."):
            result = await self.predict_structure(sequence, "interactive")
        
        if result['success']:
            # Display results
            table = Table(title="Prediction Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Sequence Length", str(len(sequence)))
            table.add_row("Overall Confidence", f"{result['confidence']['overall']:.3f}")
            table.add_row("Model Used", result['model_used'])
            table.add_row("Processing Time", f"{result['processing_time']:.2f}s")
            table.add_row("Clash Score", f"{result['quality']['clash_score']:.2f}")
            table.add_row("Ramachandran Favored", f"{result['quality']['ramachandran_favored']:.1f}%")
            
            console.print(table)
            
            # Ask if user wants to save
            save = console.input("\n[cyan]Save structure to file? (y/N):[/cyan] ").strip().lower()
            if save in ['y', 'yes']:
                filename = console.input("[cyan]Enter filename (without extension):[/cyan] ").strip()
                if not filename:
                    filename = f"structure_{int(time.time())}"
                
                with open(f"{filename}.pdb", 'w') as f:
                    f.write(result['structure'])
                
                console.print(f"[green]Structure saved to {filename}.pdb[/green]")
        
        else:
            console.print(f"[red]Prediction failed: {result.get('error', 'Unknown error')}[/red]")
    
    def show_help(self):
        """Show help information"""
        help_text = """
[bold cyan]OpenFold Interactive Commands:[/bold cyan]

[green]Sequence Input:[/green]
  • Enter a protein sequence directly (e.g., MKLLVLGLGAGVGK...)
  • Paste FASTA format (starting with >header)

[green]Commands:[/green]
  • help     - Show this help message
  • status   - Show system status
  • quit     - Exit interactive mode

[green]Examples:[/green]
  MKLLVLGLGAGVGKSALTIQLIQ
  >my_protein
  MKLLVLGLGAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLL
        """
        console.print(Panel(help_text, title="Help", border_style="cyan"))
    
    def show_status(self):
        """Show system status"""
        table = Table(title="System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        table.add_row("Predictor", "Ready" if self.predictor else "Not loaded", 
                     self.predictor.config.model_type.value if self.predictor else "N/A")
        table.add_row("Data Pipeline", "Ready" if self.data_pipeline else "Not loaded", "")
        table.add_row("AI Agent", "Ready" if self.structure_agent else "Not loaded", "")
        
        console.print(table)

# CLI Interface
@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """OpenFold: Advanced Biomolecule Structure Prediction Platform"""
    ctx.ensure_object(dict)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.option('--mode', default='fast', type=click.Choice(['fast', 'accurate']), 
              help='Prediction mode')
def server(host, port, reload, mode):
    """Start the API server"""
    runner = OpenFoldRunner()
    runner.initialize_components(mode)
    runner.run_api_server(host, port, reload)

@cli.command()
@click.argument('sequence')
@click.option('--output', '-o', help='Output PDB file')
@click.option('--mode', default='fast', type=click.Choice(['fast', 'accurate']), 
              help='Prediction mode')
@click.option('--json-output', help='Output JSON metadata file')
def predict(sequence, output, mode, json_output):
    """Predict structure for a single sequence"""
    runner = OpenFoldRunner()
    runner.initialize_components(mode)
    
    async def run_prediction():
        result = await runner.predict_structure(sequence)
        
        if result['success']:
            console.print("[green]Prediction successful![/green]")
            
            # Save PDB file
            if output:
                with open(output, 'w') as f:
                    f.write(result['structure'])
                console.print(f"[green]Structure saved to {output}[/green]")
            
            # Save JSON metadata
            if json_output:
                with open(json_output, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                console.print(f"[green]Metadata saved to {json_output}[/green]")
            
            # Display summary
            console.print(f"Confidence: {result['confidence']['overall']:.3f}")
            console.print(f"Processing time: {result['processing_time']:.2f}s")
        
        else:
            console.print(f"[red]Prediction failed: {result.get('error', 'Unknown error')}[/red]")
            sys.exit(1)
    
    asyncio.run(run_prediction())

@cli.command()
@click.argument('input_file')
@click.option('--output-dir', '-o', default='./results', help='Output directory')
@click.option('--mode', default='fast', type=click.Choice(['fast', 'accurate']), 
              help='Prediction mode')
def batch(input_file, output_dir, mode):
    """Batch prediction from FASTA file"""
    runner = OpenFoldRunner()
    runner.initialize_components(mode)
    
    async def run_batch():
        results = await runner.batch_predict(input_file, output_dir)
        
        # Summary statistics
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        
        console.print(f"\n[bold green]Batch Processing Complete![/bold green]")
        console.print(f"Successful predictions: {successful}/{total}")
        console.print(f"Results saved to: {output_dir}")
    
    asyncio.run(run_batch())

@cli.command()
@click.option('--mode', default='fast', type=click.Choice(['fast', 'accurate']), 
              help='Prediction mode')
def interactive(mode):
    """Start interactive mode"""
    runner = OpenFoldRunner()
    runner.initialize_components(mode)
    runner.run_interactive_mode()

@cli.command()
def test():
    """Run system tests"""
    console.print("[bold blue]Running OpenFold tests...[/bold blue]")
    
    import subprocess
    result = subprocess.run(['python', '-m', 'pytest', 'tests/', '-v'], 
                          capture_output=True, text=True)
    
    console.print(result.stdout)
    if result.stderr:
        console.print(f"[red]{result.stderr}[/red]")
    
    if result.returncode == 0:
        console.print("[bold green]All tests passed![/bold green]")
    else:
        console.print("[bold red]Some tests failed![/bold red]")
        sys.exit(1)

@cli.command()
def info():
    """Show system information"""
    import torch
    import platform
    
    table = Table(title="OpenFold System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Version/Status", style="green")
    
    table.add_row("Platform", platform.platform())
    table.add_row("Python", platform.python_version())
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        table.add_row("CUDA Devices", str(torch.cuda.device_count()))
    
    console.print(table)

def main():
    """Main entry point"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main() 