# mergekit-evolve-ga

`mergekit-evolve-ga` is a script that uses a Genetic Algorithm (GA) to optimize merge parameters against model metrics measured by EleutherAI's [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). 

## Key Features

- **Traditional Parameter Optimization**: Optimize merge parameters for a fixed method
- **Multi-Method Genome Evolution**: Evolve both merge methods and parameters simultaneously
- **Semantic Operations**: Method-aware crossover and constraint-aware mutations
- **15 Supported Merge Methods**: From linear to advanced methods like DARE-TIES and breadcrumbs
- **Backwards Compatibility**: Seamlessly handles both genome types
- **Enhanced GA Optimizer**: Automatic detection of genome types with appropriate optimization strategies

## Genome Types

### Traditional Genome
Fixed merge method with parameter optimization:

```yaml
genome:
  merge_method: linear
  models:
    - path/to/modelA
    - path/to/modelB
  base_model: base_model_if_needed
  layer_granularity: 8
```

### Multi-Method Genome  
Method and parameter co-evolution:

```yaml
multi_method_genome:
  models:
    - path/to/modelA
    - path/to/modelB
    - path/to/modelC
  base_model: base_model_if_needed
  layer_granularity: 8
  
  allowed_methods:
    - linear
    - slerp
    - dare_ties
    - task_arithmetic
    
  semantic_crossover:
    method_inheritance_prob: 0.6
    parameter_compatibility_check: true
    
  semantic_mutation:
    method_mutation_prob: 0.1
    parameter_constraint_enforcement: true
```

## CLI

```
mergekit-evolve-ga [OPTIONS] --storage-path PATH GENOME_CONFIG_PATH
```

### GA Hyperparameters

#### Traditional GA Parameters
- `--population-size`: population size (default 32)
- `--elite-fraction`: fraction of top individuals carried over (default 0.125)
- `--mutation-rate`: per-gene mutation probability (default 0.15)
- `--mutation-sigma`: stddev of Gaussian mutation noise (default 0.05)
- `--crossover`: `arithmetic` or `uniform` (default `arithmetic`)
- `--tournament-size`: tournament size for selection (default 4)

#### Enhanced GA Parameters (Multi-Method Genomes)
- `--semantic-crossover`: enable method-aware crossover (auto-detected)
- `--method-mutation-rate`: probability of method changes (default 0.1)
- `--constraint-enforcement`: enforce parameter constraints (default true)

### Configuration Options

YAML configuration supports both traditional and enhanced GA parameters:

#### Traditional Configuration
```yaml
genome:
  merge_method: linear
  models: [modelA, modelB, modelC]
tasks:
  - name: truthfulqa_mc
    weight: 1.0
ga:
  population_size: 32
  elite_fraction: 0.125
  mutation_rate: 0.15
  mutation_sigma: 0.05
  crossover: arithmetic
  tournament_size: 4
```

#### Multi-Method Configuration  
```yaml
multi_method_genome:
  models: [modelA, modelB, modelC]
  allowed_methods: [linear, dare_ties, slerp]
  semantic_crossover:
    method_inheritance_prob: 0.6
    parameter_compatibility_check: true
  semantic_mutation:
    method_mutation_prob: 0.1
    parameter_constraint_enforcement: true
tasks:
  - name: truthfulqa_mc
    weight: 1.0
enhanced_ga:
  population_size: 32
  semantic_operations: true
  method_evolution: true
```
## Multi-Method Evolution

### Supported Methods

The multi-method genome system supports 15 merge methods:

**Linear Family:**
- `linear` - Standard linear interpolation
- `dare_linear` - DARE with linear interpolation  
- `della_linear` - DELLA with linear weights

**SLERP Variants:**
- `slerp` - Spherical linear interpolation

**Task Arithmetic Family:**
- `task_arithmetic` - Basic task vector arithmetic
- `ties` - TIES (Trim, Elect, and Merge)  
- `dare_ties` - DARE with TIES methodology

**Advanced Methods:**
- `breadcrumbs` - Breadcrumbs method
- `breadcrumbs_ties` - Breadcrumbs with TIES
- `model_stock` - Model stock approach
- `della` - DELLA method
- `magnus` - Magnus method

**Utility Methods:**
- `copy` - Direct model copying
- `passthrough` - Parameter passthrough
- `consensus` - Consensus-based merging

### Semantic Operations

#### Method-Aware Crossover
- **Parameter Inheritance**: Child inherits parameters compatible with selected method
- **Compatibility Checking**: Prevents invalid parameter combinations
- **Method Selection**: Intelligent selection of parent methods for offspring

#### Constraint-Aware Mutations
- **Method Mutations**: Can change merge method while preserving valid parameters  
- **Parameter Constraints**: Respects mathematical bounds for each method
- **Method-Specific Ranges**: Different parameter spaces for different methods

#### Example Evolution Path
```
Generation 1: linear(w=[0.3, 0.7]) + dare_ties(density=0.8, epsilon=0.01)
     ↓ (semantic crossover)
Generation 2: dare_ties(density=0.6, epsilon=0.015) 
     ↓ (method mutation)  
Generation 3: slerp(t=0.6)
     ↓ (parameter mutation)
Generation 4: slerp(t=0.65)
```

## Enhanced GA Architecture

### Automatic Detection
The system automatically detects genome type and selects appropriate optimization strategy:

```python
# Traditional genome → Standard GA
if isinstance(genome_def, GenomeDefinition):
    optimizer = GAOptimizer(...)
    
# Multi-method genome → Enhanced GA  
elif isinstance(genome_def, MultiMethodGenomeDefinition):
    optimizer = EnhancedGAOptimizer(...)
```

### Backwards Compatibility
- Existing configurations work unchanged
- Traditional GA behavior preserved
- Gradual migration path to enhanced features

### Performance Optimizations
- **Method Compatibility Caching**: Pre-computed compatibility matrix
- **Constraint Validation**: Fast parameter bound checking
- **Semantic Operation Efficiency**: Optimized crossover and mutation operators

## Advanced Configuration

## Advanced Configuration

### Method-Specific Parameters

```yaml
multi_method_genome:
  models: [modelA, modelB, modelC]
  allowed_methods: [linear, dare_ties, slerp]
  
  # Method-specific parameter ranges
  method_parameters:
    linear:
      weight_bounds: [0.0, 1.0]
      normalize_default: true
    dare_ties:  
      density_bounds: [0.1, 0.9]
      epsilon_bounds: [0.001, 0.1]
    slerp:
      t_bounds: [0.0, 1.0]
      
  # Semantic operation tuning
  semantic_crossover:
    method_inheritance_prob: 0.7
    parameter_compatibility_check: true
    constraint_aware: true
    
  semantic_mutation:
    method_mutation_prob: 0.15
    parameter_constraint_enforcement: true  
    method_specific_ranges: true
```

### Execution Options

- `--max-fevals`: maximum evaluations before stopping
- `--timeout`: optional time budget in seconds

Shared options with `mergekit-evolve`:

- `--strategy {pool,buffered,serial}`: evaluation scheduling
- `--vllm`: evaluate with vLLM backend
- `--in-memory`: in-memory merges (pool strategy only)
- `--wandb`: enable Weights & Biases logging
- `--num-gpus`, `--batch-size`, `--reshard`, `--trust-remote-code`, etc.

## Example Workflows

### Traditional Parameter Optimization

```bash
mergekit-evolve-ga \
  --storage-path /path/to/storage \
  --strategy pool \
  --population-size 32 \
  --max-fevals 100 \
  config_traditional.yml
```

### Multi-Method Evolution

```bash
mergekit-evolve-ga \
  --storage-path /path/to/storage \
  --strategy pool \
  --population-size 48 \
  --max-fevals 200 \
  --semantic-crossover \
  --method-mutation-rate 0.15 \
  config_multimethod.yml  
```

### CPU-Only Testing

```bash
mergekit-evolve-ga \
  --strategy serial \
  --no-merge-cuda \
  --num-gpus 0 \
  --max-fevals 16 \
  --storage-path /tmp/test \
  config_minimal.yml
```

## Notes

- Writes the best-so-far configuration to `storage_path/best_config.yaml`.
- If `--save-final-model` is set (default true), saves the final best merge to `storage_path/final_model`.
- `--reshard` converts inputs to single-shard safetensors for faster merges and is enabled by default.

### CPU-only testing

You can run a full end-to-end pipeline on a CPU-only node using the serial strategy and disabling CUDA merges. Use the HuggingFace backend (no vLLM):

```
mergekit-evolve-ga \
  --strategy serial \
  --no-merge-cuda \
  --max-fevals 16 \
  --storage-path /tmp/mk-ga \
  examples/evolve_ga_tiny.yml
```

Notes:
- `--vllm` is GPU-only; omit it on CPU.
- Serial strategy automatically switches to a CPU path when no GPUs are detected.

## Installation

Install the package and GA extras (quote the extras on zsh):

```
pip install -e '.[evolve-ga]'
```

To use CMA-ES instead (or both), install:

```
pip install -e '.[evolve]'
```
