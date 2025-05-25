# Contributing to OpenFold

We welcome contributions to OpenFold! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Docker (for containerized development)
- CUDA 11.8+ (optional, for GPU development)

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/OpenFold.git
   cd OpenFold
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/
   ```

## Contributing Process

### 1. Choose an Issue

- Look for issues labeled `good first issue` for beginners
- Check existing issues and discussions before creating new ones
- Comment on issues you'd like to work on

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Make Changes

- Follow our coding standards (see below)
- Write tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 4. Commit Changes

Use conventional commit messages:

```bash
git commit -m "feat: add new structure prediction model"
git commit -m "fix: resolve memory leak in data processing"
git commit -m "docs: update API documentation"
```

### 5. Submit Pull Request

- Push your branch to your fork
- Create a pull request with a clear description
- Link to relevant issues
- Ensure CI checks pass

## Coding Standards

### Python Style

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use type hints for all functions and methods
- Use docstrings for all public functions, classes, and modules

### Code Formatting

We use automated formatting tools:

```bash
# Format code
black .
isort .

# Check formatting
black --check .
isort --check-only .

# Lint code
flake8 .
mypy .
```

### Docstring Style

Use Google-style docstrings:

```python
def predict_structure(sequence: str, model_type: str = "alphafold3") -> PredictionResult:
    """Predict protein structure from amino acid sequence.
    
    Args:
        sequence: Amino acid sequence in single-letter code.
        model_type: Type of model to use for prediction.
        
    Returns:
        PredictionResult containing structure and confidence scores.
        
    Raises:
        ValueError: If sequence contains invalid characters.
        ModelError: If model fails to load or predict.
        
    Example:
        >>> result = predict_structure("MKLLVLGLGAGVGK")
        >>> print(result.confidence.overall)
        0.85
    """
```

### Import Organization

Organize imports in this order:

1. Standard library imports
2. Third-party imports
3. Local application imports

```python
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from core.models import AdvancedPredictor
from core.data import DataPipeline
```

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interactions
├── e2e/           # End-to-end tests for complete workflows
└── fixtures/      # Test data and fixtures
```

### Writing Tests

- Use pytest for all tests
- Aim for >90% code coverage
- Write both positive and negative test cases
- Use descriptive test names

```python
def test_sequence_validation_with_valid_protein_sequence():
    """Test that valid protein sequences pass validation."""
    validator = SequenceValidator()
    result = validator.validate("MKLLVLGLGAGVGK")
    assert result.valid is True
    assert len(result.errors) == 0

def test_sequence_validation_with_invalid_characters():
    """Test that sequences with invalid characters fail validation."""
    validator = SequenceValidator()
    result = validator.validate("MKLLVL123")
    assert result.valid is False
    assert any("invalid characters" in error.message.lower() for error in result.errors)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_predictor.py

# Run with coverage
pytest --cov=core --cov-report=html

# Run performance tests
pytest tests/performance/ -m performance
```

## Documentation

### Code Documentation

- All public APIs must have docstrings
- Include type hints for all parameters and return values
- Provide usage examples for complex functions

### API Documentation

- API endpoints are automatically documented via FastAPI
- Ensure all endpoints have proper descriptions and examples
- Update OpenAPI schema when adding new endpoints

### User Documentation

- Update README.md for user-facing changes
- Add examples to the `examples/` directory
- Update installation instructions if dependencies change

## Submitting Changes

### Pull Request Guidelines

1. **Title**: Use a clear, descriptive title
2. **Description**: Explain what changes you made and why
3. **Testing**: Describe how you tested your changes
4. **Documentation**: Note any documentation updates needed
5. **Breaking Changes**: Clearly mark any breaking changes

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## Review Process

### What We Look For

1. **Correctness**: Does the code work as intended?
2. **Style**: Does it follow our coding standards?
3. **Tests**: Are there adequate tests?
4. **Documentation**: Is it properly documented?
5. **Performance**: Are there any performance implications?
6. **Security**: Are there any security concerns?

### Review Timeline

- Initial review within 2-3 business days
- Follow-up reviews within 1-2 business days
- Merge after approval from at least one maintainer

### Addressing Feedback

- Respond to all review comments
- Make requested changes in new commits
- Mark conversations as resolved when addressed
- Request re-review when ready

## Development Guidelines

### Performance Considerations

- Profile code for performance-critical paths
- Use appropriate data structures and algorithms
- Consider memory usage for large datasets
- Optimize GPU utilization where applicable

### Security Best Practices

- Never commit API keys or secrets
- Validate all user inputs
- Use secure communication protocols
- Follow OWASP guidelines for web applications

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Log errors appropriately
- Handle edge cases gracefully

```python
try:
    result = predict_structure(sequence)
except ValidationError as e:
    logger.error(f"Sequence validation failed: {e}")
    raise ValueError(f"Invalid sequence: {e.message}")
except ModelError as e:
    logger.error(f"Model prediction failed: {e}")
    raise RuntimeError(f"Prediction failed: {e.message}")
```

## Getting Help

- **Issues**: Create an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact nikjois@llamasearch.ai for urgent matters
- **Documentation**: Check the docs at https://openfold.readthedocs.io

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Annual contributor acknowledgments

Thank you for contributing to OpenFold! 