# Fastfood Projection for Neural Network Training

This directory contains configuration files for training neural networks using **Fastfood random projection**, an efficient structured random projection method.

## What is Fastfood Projection?

Fastfood projection is a structured random matrix approach that approximates Gaussian random projections using the Fast Walsh-Hadamard Transform (FWHT). It provides:

- **Fast computation**: O(d log D) instead of O(dD) for dense random projections
- **Memory efficiency**: Stores only O(D) parameters instead of O(dD) 
- **Good approximation**: Closely approximates unstructured Gaussian random projections

The transform is: `S * H * Π * B * H * z`

where:
- **H**: Hadamard transform (via FWHT)
- **B**: Diagonal matrix with random ±1 entries
- **Π**: Random permutation
- **S**: Diagonal scaling matrix

## Key Features

- Automatically pads parameters to the next power of 2 if needed (required for FWHT)
- Pure PyTorch implementation for GPU efficiency
- Reproducible with seed control

## Usage

### Basic Example

```python
from src.weight_sharing import FastfoodProjection
from src.models.lenet import LeNetCIFAR

# Create model
model = LeNetCIFAR()

# Create fastfood projection
ws = FastfoodProjection(
    model=model,
    d=128,           # Latent dimension
    alpha=1.0,       # Scaling factor
    device='cuda',
    seed=42
)

# Use in optimization
import numpy as np
z = np.random.randn(128)
params = ws(z)  # Maps to full parameter space
```

### Running Experiments

The configuration files in this directory provide ready-to-use setups:

```bash
# Differential Evolution
python command_runner.py scripts/fastfood/ce_de.yaml

# CMA-ES
python command_runner.py scripts/fastfood/ce_cmaes.yaml

# OpenES
python command_runner.py scripts/fastfood/ce_openes.yaml
```

## Configuration Files

- **ce_de.yaml**: Experiments with Differential Evolution optimizer
- **ce_cmaes.yaml**: Experiments with CMA-ES optimizer
- **ce_openes.yaml**: Experiments with OpenES optimizer

Each configuration tests different latent dimensions (d=128, 256, 512, 1024) to explore the trade-off between compression and model capacity.

## Parameters

### Key Hyperparameters

- `--ws fastfood`: Specifies Fastfood projection
- `--d`: Latent dimension (number of parameters to optimize)
- `--alpha`: Scaling factor for the projection (default: 1.0)
- `--ws_device`: Device for weight sharing computations ('cuda' or 'cpu')

### Recommended Settings

| Model Size | Recommended d | Compression Ratio |
|------------|---------------|-------------------|
| Small (<100K params) | 256-512 | 100-400x |
| Medium (100K-1M params) | 512-2048 | 50-1000x |
| Large (>1M params) | 2048-8192 | 100-500x |

## Advantages over Dense Random Projection

1. **Speed**: Much faster for large models (O(d log D) vs O(dD))
2. **Memory**: Minimal memory footprint
3. **Quality**: Comparable or better optimization performance
4. **Scalability**: Can handle very large models

## Reference

Le, Q., Sarlós, T., & Smola, A. (2013). Fastfood-approximating kernel expansions in loglinear time. *ICML 2013*.

## Tips

1. **Power of 2**: The implementation automatically pads to the next power of 2, so you don't need to worry about model size
2. **Latent dimension**: Start with d = sqrt(D) as a baseline and adjust based on results
3. **Seed control**: Use different seeds for multiple runs to ensure robustness
4. **Device**: Use 'cuda' for large models, 'cpu' for small experiments

