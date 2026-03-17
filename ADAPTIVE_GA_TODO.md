# Adaptive GA Redesign TODO

Status: phase 1 in progress
Branch: `codex/ga-evolve/adaptive-ga-phase-1`
Last updated: `2026-03-17`

This file is the review-first source of truth for the adaptive GA redesign.
It preserves the full intent of the 28-step proposal, but stages the work so
we can implement it safely and keep a clear record of what has and has not been
done on this branch.

## Working Position

- Keep the essence of the 28-step proposal.
- Do not treat the 28 steps as one atomic code change.
- Phase 1 is the core research contribution and should stay small enough to
  verify locally on tiny models.
- Phase 2 adds stronger search control and richer stability/diversity handling.
- Phase 3 adds the structural features that require deeper refactors.

## Current Code Reality

- The current genome already supports evolving merge methods, model selection,
  and method parameters.
- The current GA already supports elitism, tournament selection, and immigrants.
- MLflow already records merge-method usage and success metrics.
- The current GA now adapts operator probabilities from observed
  performance.
- The current GA now supports a built-in stage-1/stage-2 evaluation
  pipeline.
- Same-method layered configs are now emitted as real slice configs.
- Mixed-method layered genomes now execute through a hierarchical plan because
  mergekit still exposes one top-level `merge_method` per config.
- Multi-objective or Pareto ranking is not implemented yet.

## Current Progress On This Branch

- [x] Added phase-1 adaptive operator config fields to the GA schema
- [x] Added optimizer-side operator state and adaptive method probabilities
- [x] Added passthrough penalty and per-generation passthrough caps
- [x] Added method-aware parameter sampling and method projection helpers
- [x] Fixed method gene encoding to respect `allowed_methods` order
- [x] Extended GA logging and MLflow metrics for adaptive operator behavior
- [x] Added focused unit tests for the new adaptive operator slice
- [x] Added two-stage evaluation plumbing to the evaluation strategies
- [x] Added diversity-aware parent selection
- [x] Added role-separated breeding with explorer slots
- [x] Added real layer-aware slice configs for same-method layered genomes
- [x] Added hierarchical execution for layered mixed-method genomes
- [x] Added structured tiny-model fitness with explicit components
- [x] Added candidate-level GA history output
- [x] Added phase-2 behavior probes and smoke-test rejection hooks
- [x] Added phase-2 local and cloud experiment presets

## Preserved 28-Step Intent

Each item below preserves one part of the original proposal and records the
target phase.

1. Search for better merged models. Phase: `P1`
2. Prevent collapse into passthrough-only behavior. Phase: `P1`
3. Adaptively learn which merge methods work best. Phase: `P1`
4. Represent each candidate as a genome containing parents, method, and
   hyperparameters. Phase: `P1`
5. Keep model-level evolution and operator-level evolution visible in the
   design. Phase: `P1`
6. Use a size-aware allowed-method policy. Phase: `P1`
7. Add role separation inside the population. Phase: `P2`
8. Replace raw single-metric thinking with a structured fitness design.
   Phase: `P1`
9. Support multi-objective or weighted-rank evolution. Phase: `P3`
10. Track operator success statistics explicitly. Phase: `P1`
11. Update operator probabilities adaptively with smoothing. Phase: `P1`
12. Use diversity-aware parent selection. Phase: `P2`
13. Make child generation method-aware, including parameter sampling.
    Phase: `P1`
14. Keep mutation small and safe for tiny models. Phase: `P1`
15. Preserve elites. Phase: `Already present`, refine in `P2`
16. Preserve diversity deliberately. Phase: `P2`
17. Add early rejection filters for obviously bad children. Phase: `P2`
18. Add two-stage evaluation. Phase: `P1`
19. Use a tiny-model task mix that is less misleading than heavy reasoning
    tasks. Phase: `P1`
20. Control passthrough with penalties and caps instead of banning it.
    Phase: `P1`
21. Add layer-aware extension capability. Phase: `P3`
22. Formalize the generation loop around adaptive operators and staged
    evaluation. Phase: `P1`
23. Log candidate-, operator-, and population-level data. Phase: `P1`
24. Define observable success criteria for the framework. Phase: `P1`
25. Preserve the academic claims enabled by the framework. Phase: `P1`
26. Start from a practical Pythia-70M config. Phase: `P1`
27. Add a stronger 160M+ config once the tiny-model path is stable. Phase: `P2`
28. Start simple with `passthrough + linear + slerp`, staged evaluation, and
    operator logging. Phase: `P1`

## Phase Plan

## Phase 1: Core Adaptive Operator Framework

Goal: prove the adaptive-operator idea on tiny local runs without blowing up the
existing code path.

### Scope

- [x] Extend GA config schema with:
  `initial_method_probs`, `passthrough_penalty`,
  `passthrough_max_fraction`, adaptive-operator weights, and adaptive sampling
- [x] Extend GA config schema with:
  `two_stage`, `stage1_limit`, `stage2_limit`, `stage2_top_k`
- [x] Extend GA config schema with:
  `fitness_mode`, `task_mix_profile`
- [x] Define a small operator-state model:
  usage count, success count, failure count, average child fitness,
  parent-improvement rate, survival rate, next-generation probability
- [x] Add adaptive method sampling to the GA loop
- [x] Add smoothed probability updates using operator scores
- [x] Add passthrough controls:
  small penalty and per-generation fraction cap
- [x] Keep allowed methods narrow for tiny runs:
  `passthrough`, `linear`, `slerp`
- [x] Add method-aware child parameter sampling:
  alpha ranges and safe defaults by method
- [x] Keep mutation small and genotype-level only for phase 1
- [x] Add structured scalar fitness for tiny models:
  task score + language quality + stability bonus/penalty +
  diversity proxy + passthrough penalty
- [x] Add two-stage evaluation inside the GA:
  cheap screen, shortlist, then full evaluation
- [x] Reuse the deterministic sweep staging idea instead of inventing a
  separate evaluation stack
- [x] Expand MLflow logging to include adaptive probabilities and
  operator-improvement metrics
- [x] Emit workspace artifacts for operator history and per-generation
  adaptive probability history
- [x] Add a tracked local experiment config for Pythia-70M using the
  phase-1 settings

### Expected Files

- [x] [mergekit/evo/config.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/config.py)
- [x] [mergekit/evo/enhanced_ga.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/enhanced_ga.py)
- [x] [mergekit/evo/strategy.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/strategy.py)
- [x] [mergekit/evo/multi_method_genome.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/multi_method_genome.py)
- [x] [mergekit/scripts/evolve_ga.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/scripts/evolve_ga.py)
- [ ] [mergekit/evo/tracking.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/tracking.py)
- [x] [tests/](/Users/nkululekothangelane/Documents/master_research/mergekit/tests)
- [ ] [experiments/thesis/local_mac/](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac)

### Acceptance Criteria

- [ ] Method probabilities change over generations
- [ ] Passthrough no longer dominates by default
- [ ] The GA can preserve the best parent without collapsing entirely to it
- [ ] Stage-1 and stage-2 evaluation results are both recorded
- [ ] MLflow shows operator usage, survival, improvement, and probability shift
- [ ] A tiny local run completes without tokenizer-path regressions

## Proposed Reference Pseudocode

This section preserves the intended execution shape of the redesign. We should
review implementation changes against this flow, not just against individual
features.

```text
initialize_population()
evaluate_population()
initialize_operator_stats()

for generation in range(num_generations):
    elites = select_elites(population)

    children = []
    while len(children) < target_children:
        method = sample_method(adaptive_method_probs)

        parents = select_parents(population, method)

        child = merge_models(
            parents=parents,
            method=method,
            params=sample_merge_params(method)
        )

        child = maybe_mutate(child)

        if not passes_smoke_test(child):
            continue

        child.stage1_metrics = evaluate_stage1(child)

        children.append(child)

    shortlisted = select_for_stage2(children)
    evaluate_stage2(shortlisted)

    population = form_next_generation(elites, shortlisted, immigrants=True)

    compute_fitness(population)
    update_operator_stats(population)
    adaptive_method_probs = update_method_probs(operator_stats)

    log_generation(population, operator_stats, adaptive_method_probs)
```

## Pseudocode Review Mapping

This maps the proposed pseudocode to the current codebase so we can review what
already exists and what must be added or refactored.

### Already Present

- `initialize_population()`
  Current home: [mergekit/evo/enhanced_ga.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/enhanced_ga.py)
  Current function: `EnhancedGAOptimizer._init_population()`
- `evaluate_population()`
  Current home: [mergekit/evo/enhanced_ga.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/enhanced_ga.py)
  Current function: `EnhancedGAOptimizer._evaluate_population()`
- `select_elites(population)`
  Current home: [mergekit/evo/enhanced_ga.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/enhanced_ga.py)
  Current behavior: inside `EnhancedGAOptimizer.run()`
- `select_parents(population, method)`
  Partial current home: [mergekit/evo/enhanced_ga.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/enhanced_ga.py)
  Current function: `EnhancedGAOptimizer._select_parent()`
  Gap: method-aware and diversity-aware selection is not implemented
- `maybe_mutate(child)`
  Current home: [mergekit/evo/enhanced_ga.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/enhanced_ga.py)
  Current function: `EnhancedGAOptimizer._enhanced_mutate()`
- `log_generation(...)`
  Current homes:
  [mergekit/scripts/evolve_ga.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/scripts/evolve_ga.py)
  and [mergekit/evo/tracking.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/tracking.py)
  Current state: generation logging exists, adaptive-operator logging does not yet fully exist

### Needs New Logic

- `initialize_operator_stats()`
  Needed in: [mergekit/evo/enhanced_ga.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/enhanced_ga.py)
- `sample_method(adaptive_method_probs)`
  Needed in: [mergekit/evo/enhanced_ga.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/enhanced_ga.py)
- `sample_merge_params(method)`
  Needed in: [mergekit/evo/multi_method_genome.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/multi_method_genome.py)
  or a dedicated helper module
- `passes_smoke_test(child)`
  Needed in:
  [mergekit/evo/strategy.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/strategy.py)
  or a new evaluation helper layer
- `evaluate_stage1(child)`
  Needed in:
  [mergekit/evo/strategy.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/strategy.py)
- `select_for_stage2(children)`
  Needed in:
  [mergekit/evo/enhanced_ga.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/enhanced_ga.py)
- `evaluate_stage2(shortlisted)`
  Needed in:
  [mergekit/evo/strategy.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/strategy.py)
- `form_next_generation(elites, shortlisted, immigrants=True)`
  Needs refactor in:
  [mergekit/evo/enhanced_ga.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/enhanced_ga.py)
- `compute_fitness(population)`
  Current scoring exists indirectly through evaluation results
  Gap: structured phase-1 scalar fitness and stage-aware scoring policy
- `update_operator_stats(population)`
  Needed in:
  [mergekit/evo/enhanced_ga.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/enhanced_ga.py)
  with reporting in
  [mergekit/scripts/evolve_ga.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/scripts/evolve_ga.py)
- `update_method_probs(operator_stats)`
  Needed in:
  [mergekit/evo/enhanced_ga.py](/Users/nkululekothangelane/Documents/master_research/mergekit/mergekit/evo/enhanced_ga.py)

## Proposed Review Walkthrough Of The Pseudocode

This is the order I recommend for reviewing the redesign against the pseudocode.

1. Initialization:
   confirm what state belongs in population state, operator state, and config
2. Child generation:
   confirm method sampling, parent selection, and parameter sampling
3. Screening:
   confirm smoke tests and stage-1 evaluation rules
4. Promotion:
   confirm how shortlist-to-stage-2 works
5. Population update:
   confirm elites, children, and immigrants
6. Learning:
   confirm operator-score formula and smoothed probability updates
7. Logging:
   confirm candidate-level, operator-level, and population-level MLflow outputs

## Phase 2: Search Control and Diversity

Goal: improve search quality after phase 1 is stable.

### Scope

- [x] Add explicit population roles:
  elites, exploiters, explorers, immigrants
- [x] Add diversity-aware tournament selection
- [x] Add gene-diversity scoring
- [x] Add simple behavior-diversity scoring on a fixed prompt set
- [x] Add early rejection filters for repetitive or degenerate outputs
- [x] Improve stability scoring beyond merge/eval success
- [x] Broaden the tiny-to-mid-size method policy to include `ties`
- [x] Add a stronger 160M+ starter config

### Acceptance Criteria

- [x] Diversity remains measurable across generations
- [ ] Explorer slots actually use less common methods
- [ ] Search revisits fewer near-duplicate candidates
- [x] Rejection filters remove clearly broken children before full eval

## Phase 3: Structural Extensions

Goal: add the features that need deeper architectural work and stronger tests.

### Scope

- [ ] Implement real layer-aware or block-aware merge configs
- [x] Let genomes specify per-block method assignments and parameters
- [ ] Implement real layer-aware or block-aware merge configs
  Note: native same-method slice configs are implemented; mixed per-block
  methods currently execute through a hierarchical merge plan instead of a
  single native mergekit config because mergekit still exposes one top-level
  `merge_method` per config.
- [ ] Add weighted-rank or Pareto-based multi-objective optimization
- [ ] Add more advanced diversity measures where cost is justified
- [ ] Expand the method family only after the pipeline is stable:
  `ties`, `dare_linear`, `dare_ties`, and later others

### Acceptance Criteria

- [ ] Layer-aware genomes decode into real slice-based merge configs
- [ ] The optimizer can rank candidates without forcing everything into one
  raw scalar
- [ ] The extended search space remains stable under tests and smoke runs

## Detailed Phase 1 Task List

These are the concrete review items we should walk through one by one.

### A. Config and Schema

- [x] Define adaptive-GA config fields in the Pydantic schema
- [x] Decide which fields live under `ga` versus `evaluation`
- [x] Decide whether task-mix presets belong in YAML or experiment presets only
- [x] Validate mutually exclusive or invalid combinations early

### B. Operator State and Sampling

- [x] Introduce an operator-state structure with counts, rates, and probability
- [x] Initialize probabilities from config, with defaults for tiny runs
- [x] Compute operator score from child fitness, parent improvement, and survival
- [x] Apply softmax plus smoothing when updating probabilities
- [x] Persist per-generation operator state to CSV and MLflow

### C. Fitness Design

- [x] Define the phase-1 scalar fitness formula
- [x] Normalize higher-is-better and lower-is-better metrics consistently
- [x] Decide the first stability proxy
- [x] Decide the first diversity proxy
- [x] Add passthrough penalty without breaking baseline preservation
- [x] Keep `delta_vs_best_baseline` visible in reporting

### D. Two-Stage Evaluation

- [x] Define the stage-1 cheap evaluation contract
- [x] Define how many children advance to stage 2
- [ ] Decide whether elites need stage-2 reevaluation every generation
- [x] Record stage-1 and stage-2 scores separately
- [ ] Ensure failed stage-1 candidates do not poison caches incorrectly

### E. Child Generation

- [x] Narrow phase-1 methods to `passthrough`, `linear`, `slerp`
- [x] Sample parents according to method requirements
- [x] Sample alpha from safe method-specific ranges
- [x] Keep mutation small and bounded
- [x] Defer tensor-level weight mutation until there is a strong reason to add it

### F. Logging and MLflow

- [x] Log candidate-level data:
  parents, method, params, stage-1 metrics, stage-2 metrics, fitness
- [x] Log operator-level data:
  counts, success, failure, improvement, survival, next probability
- [x] Log population-level data:
  best, median, diversity proxy, passthrough fraction, method mix
- [x] Write workspace markdown linking to the MLflow run for review

### G. Experiments and Tests

- [x] Add unit tests for adaptive probability updates
- [x] Add unit tests for passthrough caps and penalties
- [x] Add unit tests for staged evaluation selection and promotion
- [x] Add unit tests for config validation
- [x] Add a tiny local smoke preset for phase 1
- [ ] Add a comparison plan:
  static operator schedule vs adaptive operator schedule

## Open Design Decisions For Review

- [ ] Use a structured scalar fitness first, or weighted-rank from day one?
- [ ] How should survival rate be defined:
  survives to next generation, or survives to stage 2, or both?
- [ ] Should diversity in phase 1 be gene-only, or gene plus cheap behavioral
  probes?
- [ ] Should stage 1 use smaller task limits, fewer tasks, or both?
- [ ] Should passthrough cap apply to the full population or children only?
- [ ] Should operator updates use generation-local statistics only, or an
  exponential moving average over generations?

## Explicit Deferrals

These ideas are kept, but intentionally deferred so phase 1 stays tractable.

- [ ] True per-layer mixed-method merging
- [ ] Full Pareto optimization
- [ ] Expensive behavior-diversity over large prompt sets
- [ ] Heavy mutation or pruning on tiny models
- [ ] Broad method families on local tiny runs before the adaptive loop is stable

## Suggested First Review Order

1. Confirm phase boundaries
2. Confirm the phase-1 fitness design
3. Confirm phase-1 adaptive operator score formula
4. Confirm passthrough controls
5. Confirm stage-1 and stage-2 evaluation behavior
6. Confirm the first Pythia-70M experiment preset
7. Only then start code changes
