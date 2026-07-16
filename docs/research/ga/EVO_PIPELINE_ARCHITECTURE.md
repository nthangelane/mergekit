# Evo Pipeline Architecture

The evo command remains `python -m mergekit.scripts.evolve_ga`, but reusable
responsibilities are separated from the Click entry point:

- `mergekit/evo/config.py`: validated run schema, including explicit optimizer
  selection and versioned fitness definitions.
- `mergekit/evo/fitness.py`: resolved fitness metadata and the immutable
  `fitness_definition.json` artifact.
- `mergekit/evo/orchestrator.py`: device policy, stop and GA parameter
  resolution, genome/strategy construction, and resume signatures.
- `mergekit/evo/optimizer_factory.py`: standard/enhanced optimizer selection and
  enhanced parameter construction.
- `mergekit/evo/baselines.py`: parent-model evaluation and version-aware
  baseline CSV reuse.
- `mergekit/evo/reporting.py`: candidate/method history, summaries, comparison
  tables, plots, stop metadata, and tracker artifacts.
- `mergekit/scripts/evolve_ga.py`: CLI options and the high-level lifecycle that
  connects these modules.

## Optimizer Selection

New experiment YAML should choose explicitly:

```yaml
optimizer: enhanced
```

`standard` is valid only for the traditional single-method genome without
enhanced GA features. `auto` remains the default solely for compatibility with
older YAML: a legacy `ga` block retains its previous enhanced-optimizer
selection, while a basic traditional config without `ga` uses the standard
optimizer.

## Compatibility Contract

Historical YAML remains valid. Configuration defaults preserve the old fitness
formula and optimizer dispatch. The run signature includes the resolved config
and GA parameters, so changing the fitness version, optimizer selection, device
policy, or evaluation settings prevents an incompatible checkpoint resume.
