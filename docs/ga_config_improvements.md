# GA Evolution Configuration Improvements

## Current Issues & Proposed Solutions

### 1. **Parameter Space Definition**
```yaml
# Current: Limited to simple ranges
parameters:
  weight:
    min: 0.0
    max: 1.0

# Proposed: Rich parameter definitions
parameters:
  weight:
    type: continuous
    range: [0.0, 1.0]
    distribution: uniform
    constraint: sum_to_one  # For multi-model merging

  layer_weights:
    type: discrete
    choices: [0.1, 0.25, 0.5, 0.75, 0.9]
    per_layer: true
```

### 2. **Multi-Objective Optimization**
```yaml
objectives:
  - name: hellaswag
    weight: 0.4
    direction: maximize
  - name: model_size
    weight: 0.3
    direction: minimize
  - name: inference_speed
    weight: 0.3
    direction: maximize

optimization:
  method: nsga2  # Non-dominated Sorting GA
  pareto_archive: true
```

### 3. **Adaptive Parameters**
```yaml
ga:
  population_size:
    initial: 20
    adaptive: true
    min: 10
    max: 50

  mutation_rate:
    initial: 0.2
    decay: 0.95
    schedule: exponential
```

### 4. **Resource Management**
```yaml
resources:
  max_concurrent_evaluations: 4
  memory_limit: "16GB"
  timeout_per_evaluation: "10m"
  disk_cache: true
  checkpoint_frequency: 10  # generations
```
