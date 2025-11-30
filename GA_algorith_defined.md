# MergeKit GA Algorithm (Current Design)

## High-Level Goals
- Discover merge configurations that maximize task-specific evaluation scores.
- Support both traditional linear merges and multi-method genomes without sacrificing reliability.
- Guarantee that at least one known-good linear merge is evaluated in every generation for stability.

## Key Components

### Genome Representations
- **ModelGenome**: classic continuous genome for simple linear blends.
- **MultiMethodGenome**: extended genome that encodes method selection, model participation, and method parameters per layer group.
  - Method gene chooses from the YAML `allowed_methods` list.
  - Model-selection genes weight and prune candidate source models.
  - Parameter genes map to method-specific tensors (weights, density, epsilon, gamma, etc.).
   - NuSLERP automatically backfills missing non-base sources so merges never fail when the GA temporarily zeroes out alternative models.

### Evaluation Strategies
- `SerialEvaluationStrategy`: merges and evaluates individuals one at a time using the CPU helper.
- `PoolEvaluationStrategy`: distributes work to Ray actors for higher throughput.
- Both strategies delegate merging to `merge_model` (which now emits complete parameter dictionaries for GTA methods) and scoring to `evaluate_model_cpu`.

## Population Initialisation
1. Fetch the baseline genotype from `genome.initial_genotype(random=False)`.
2. Seed the population with:
   - The baseline genotype (pure equal-weight linear merge).
   - One-hot variants for each source model (if using non-random init).
   - Random or perturbed variants to fill the remaining slots.
3. Cache the baseline vector in `self._baseline_genotype` for reuse.

## Generation Loop (Per `GAOptimizer.run`)

```
+-------------------------+
| Start Generation g      |
+------------+------------+
             |
             v
  Clamp pop[0] := baseline
             |
             v
  Evaluate population -> scores
             |
             v
  Update best_x / best_score
             |
             v
  Apply elitism (top n_elite)
             |
             v
  Breed + mutate children
             |
             v
  Inject immigrants (optional)
             |
             v
  Assemble next_pop array
             |
             v
  Restore baseline: next_pop[0] := baseline
             |
             v
+------------+------------+
| Loop or exit (max fevals)|
+-------------------------+
```

### Detailed Flow
1. **Generation Start Hook** (`on_generation_start`):
   - Reports progress, ETA, and current best before evaluation begins.
2. **Baseline Reinsertion**:
   - `pop[0]` is overwritten with `self._baseline_genotype` right before evaluation, ensuring a linear merge is always tested first.
3. **Population Evaluation**:
   - Individuals hashed for caching to skip re-evaluations.
   - Each genotype merged and scored; failed merges return `score=None` and are treated as `-inf`.
4. **Result Logging** (`on_population_evaluated`):
   - Writes generation statistics to `ga_history.csv` (best, mean, std, mutation sigma, cache hits, etc.).
5. **Global Best Tracking**:
   - If current generation beats `best_score`, emit `on_new_best` and copy genotype.
   - Otherwise increment `no_improve`; optionally decay `mutation_sigma` after `patience` stagnant generations.
6. **Elitism & Breeding**:
   - Preserve top `n_elite` individuals.
   - Perform crossover (arithmetic/uniform/SBX) and Gaussian mutation to fill the population.
   - Optional immigrant fraction replaces the tail with fresh random genotypes.
7. **Baseline Restoration for Next Gen**:
   - After the new population array is built, `pop[0]` is again set to the baseline so the guarantee holds for the upcoming generation.
8. **Termination**:
   - Exit when `max_fevals` reached or optional timeout expires.

## Baseline Safeguard Rationale
- Guarantees at least one successful merge per generation.
- Provides a stable reference score to compare multi-method experiments.
- Prevents regressions when newly explored methods produce degenerate parameters.

## Logging & Artifacts
- `ga_history.csv`: chronological record of generation metrics (first row may show `best_so_far=-inf` before baseline evaluation completes).
- `best_config.yaml`: YAML for the best-scoring genotype (method-aware via `genotype_to_merge_config`).
- `baseline_results.csv`: evaluation metrics for each genome seed model.
- Pruning of temporary `merged/` artifacts is handled post-generation to conserve disk space.

## Failure Handling
- Invalid genotypes (`InvalidGenotypeError`, `MultiMethodInvalidGenotypeError`) return early.
- Merge exceptions log and skip individuals without crashing the run.
- Chat-template gaps trigger automatic fallback to raw text evaluation.

## Extensibility Hooks
- **Callbacks**: `on_generation_start`, `on_population_evaluated`, `on_new_best` for telemetry or custom behaviours.
- **Mutation Controls**: `patience`, `sigma_decay`, `min_mutation_sigma` tune exploration vs. exploitation.
- **Genome Extensions**: Additional methods or parameter constraints can be wired into `MultiMethodGenome` without modifying the GA loop.

## Summary
The current algorithm combines robust baseline safeguarding with flexible method exploration. By anchoring every generation with a reliable linear merge, we retain steady progress while still allowing the GA to discover and adopt superior multi-method configurations when they deliver tangible gains.
