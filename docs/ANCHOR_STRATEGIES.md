# Anchor Strategies for Evolutionary Neural Network Training

This document describes the various anchor strategies available in the evolutionary neural network training framework, including the new advanced strategies based on the proposing concept.

## Overview

The anchor strategy determines how the base parameters (theta_anchor) are updated after each epoch of evolutionary optimization. The proposing concept uses weight sharing to map from a low-dimensional latent space to the full parameter space, and the anchor serves as the reference point for this mapping.

## Available Anchor Strategies

### 1. Fixed Anchor (`fixed`)
- **Description**: Uses the original random initialization as the anchor throughout training
- **Use Case**: Baseline comparison, when you want to maintain a fixed reference point
- **Advantages**: Simple, stable, good baseline
- **Disadvantages**: May not adapt to optimization progress

### 2. Full Anchor (`full`)
- **Description**: Updates the anchor to the current best solution each epoch
- **Use Case**: When you want the anchor to track the optimization progress
- **Advantages**: Always uses the best solution as reference
- **Disadvantages**: Can be unstable, may lose exploration

### 3. EMA Anchor (`ema`)
- **Description**: Uses exponential moving average to update the anchor
- **Formula**: `theta_anchor = theta_anchor + lr * delta`
- **Use Case**: Smooth anchor updates with learning rate control
- **Advantages**: Smooth updates, controllable learning rate
- **Disadvantages**: May be slow to adapt

## New Advanced Anchor Strategies

### 4. Adaptive Learning Rate Anchor (`ala`)
- **Description**: Adapts the learning rate based on performance improvement
- **Formula**: 
  - If improving: `lr_adaptive = lr * improvement_factor`
  - If not improving: `lr_adaptive = lr * decay_factor`
- **Parameters**:
  - `--improvement_factor`: Factor to increase LR when improving (default: 1.2)
  - `--decay_factor`: Factor to decrease LR when not improving (default: 0.8)
- **Use Case**: When you want adaptive learning rates based on performance
- **Advantages**: Automatically adjusts learning rate, can accelerate good progress
- **Disadvantages**: May be sensitive to parameter tuning

### 5. Momentum-Based Anchor (`mba`)
- **Description**: Uses momentum to smooth anchor updates
- **Formula**: 
  - `velocity = momentum * velocity + lr * delta`
  - `theta_anchor = theta_anchor + velocity`
- **Parameters**:
  - `--momentum`: Momentum factor (default: 0.9)
- **Use Case**: When you want smooth, stable anchor updates
- **Advantages**: Reduces oscillations, smooth convergence
- **Disadvantages**: May be slow to respond to changes

### 6. Multi-Scale Anchor (`msa`)
- **Description**: Uses different learning rates for different parameter groups
- **Formula**: Different LRs for conv, batch norm, and fully connected layers
- **Use Case**: When different layers need different update rates
- **Advantages**: Layer-specific adaptation, can be more effective
- **Disadvantages**: More complex, requires understanding of layer types

### 7. Performance-Weighted Anchor (`pwa`)
- **Description**: Weights anchor updates based on performance improvement
- **Formula**: `weighted_delta = performance_weight * delta`
- **Parameters**:
  - `--performance_weight`: Weight for performance-based updates (default: 1.0)
- **Use Case**: When you want to emphasize good-performing updates
- **Advantages**: Focuses on promising directions
- **Disadvantages**: May be too aggressive in early stages

### 8. Hierarchical Anchor (`ha`)
- **Description**: Uses different update strategies for different layer types
- **Strategy**:
  - Conv layers: Full update
  - Batch norm layers: EMA update
  - Other layers: Momentum update
- **Use Case**: When different layers need different update strategies
- **Advantages**: Layer-specific strategies, can be very effective
- **Disadvantages**: Most complex, requires careful tuning

### 9. Ensemble Anchor (`ea`)
- **Description**: Maintains multiple anchors and uses ensemble voting
- **Formula**: Weighted combination of multiple anchors based on performance
- **Parameters**:
  - `--ensemble_size`: Number of anchors in ensemble (default: 3)
- **Use Case**: When you want robust anchor updates using multiple strategies
- **Advantages**: Robust to individual anchor failures, combines multiple approaches
- **Disadvantages**: Higher computational cost, more complex

### 10. Uncertainty-Aware Anchor (`uaa`)
- **Description**: Uses ES uncertainty (if available) to guide anchor updates
- **Formula**: Weight updates inversely by uncertainty
- **Use Case**: When you want to be more conservative in uncertain regions
- **Advantages**: Can avoid overconfident updates, uses ES uncertainty
- **Disadvantages**: Depends on ES implementation, fallback to performance weighting

### 11. Adaptive Proposing Space (`aps`)
- **Description**: Dynamically adjusts the proposing space dimensionality
- **Formula**: Increases/decreases `d` based on optimization progress
- **Parameters**:
  - `--max_d_ratio`: Maximum ratio for space expansion (default: 2.0)
  - `--min_d_ratio`: Minimum ratio for space contraction (default: 0.5)
- **Use Case**: When you want to adapt the search space complexity
- **Advantages**: Can find optimal search space size, adaptive complexity
- **Disadvantages**: Most complex, requires careful tuning

### 12. Multi-Objective Anchor (`moa`)
- **Description**: Considers multiple objectives when updating the anchor
- **Formula**: Weighted combination of accuracy and efficiency objectives
- **Parameters**:
  - `--objective_weights`: Weights for objectives (default: "0.7,0.3")
- **Use Case**: When you want to balance multiple objectives
- **Advantages**: Can optimize for multiple goals simultaneously
- **Disadvantages**: More complex objective definition, parameter tuning

## Usage Examples

### Basic Usage
```bash
# Use adaptive learning rate anchor
python es_trainer.py --anchor ala --improvement_factor 1.5 --decay_factor 0.7

# Use momentum-based anchor
python es_trainer.py --anchor mba --momentum 0.95

# Use multi-scale anchor
python es_trainer.py --anchor msa

# Use performance-weighted anchor
python es_trainer.py --anchor pwa --performance_weight 1.2

# Use hierarchical anchor
python es_trainer.py --anchor ha --momentum 0.9

# Use ensemble anchor
python es_trainer.py --anchor ea --ensemble_size 5

# Use uncertainty-aware anchor
python es_trainer.py --anchor uaa

# Use adaptive proposing space
python es_trainer.py --anchor aps --max_d_ratio 2.0 --min_d_ratio 0.5

# Use multi-objective anchor
python es_trainer.py --anchor moa --objective_weights 0.8,0.2
```

### Advanced Usage
```bash
# Full configuration with new anchor strategy
python es_trainer.py \
    --dataset cifar10 \
    --arch resnet32 \
    --optimizer CMA_ES \
    --d 256 \
    --criterion ce \
    --inner_steps 2 \
    --epochs 100 \
    --ws randproj \
    --normalize_projection \
    --es_std 0.1 \
    --anchor ala \
    --lr 1.0 \
    --lr_scheduler cosine \
    --improvement_factor 1.3 \
    --decay_factor 0.8
```

## Parameter Tuning Guidelines

### Adaptive Learning Rate Anchor
- **improvement_factor**: 1.1-1.5 (higher = more aggressive when improving)
- **decay_factor**: 0.7-0.9 (lower = more conservative when not improving)

### Momentum-Based Anchor
- **momentum**: 0.8-0.95 (higher = more smoothing, lower = more responsive)

### Performance-Weighted Anchor
- **performance_weight**: 0.8-1.5 (higher = more emphasis on good performance)

### Hierarchical Anchor
- **momentum**: 0.8-0.95 (for momentum-based layers)

## When to Use Each Strategy

- **Fixed**: Baseline experiments, stable reference needed
- **Full**: Quick adaptation needed, exploration is important
- **EMA**: Smooth updates, moderate adaptation
- **ALA**: Performance-based adaptation, varying optimization speed
- **MBA**: Smooth convergence, reducing oscillations
- **MSA**: Layer-specific needs, complex architectures
- **PWA**: Performance-driven optimization, good performance tracking
- **HA**: Complex architectures, different layer requirements

## Performance Considerations

- **Fixed**: Fastest, no computation overhead
- **Full**: Fast, simple computation
- **EMA**: Fast, simple computation
- **ALA**: Moderate overhead, performance evaluation needed
- **MBA**: Moderate overhead, velocity storage needed
- **MSA**: Higher overhead, layer type detection needed
- **PWA**: Moderate overhead, performance evaluation needed
- **HA**: Highest overhead, complex layer-specific logic

## Troubleshooting

### Common Issues
1. **Oscillations**: Try momentum-based anchor or reduce learning rate
2. **Slow convergence**: Try adaptive learning rate or performance-weighted anchor
3. **Instability**: Try momentum-based anchor or reduce improvement factor
4. **Poor performance**: Try hierarchical anchor or multi-scale anchor

### Debugging Tips
1. Monitor anchor update magnitudes
2. Check performance improvement trends
3. Verify parameter group assignments for multi-scale anchor
4. Monitor velocity updates for momentum-based anchor
