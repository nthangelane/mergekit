# Versioned Fitness Definitions

Every evo run has an explicit fitness definition. The runner writes the resolved
definition to `fitness_definition.json`, includes it in tracker configuration,
and includes it in the resume signature.

## Versions

### v1: historical compatibility

```yaml
fitness:
  version: v1
```

Lower-is-better metrics use:

```text
quality(x) = 1 / (1 + max(0, x))
```

This is the implicit definition for YAML files that omit `fitness`. It preserves
all historical results exactly. For perplexities in the 20-200 range it yields
approximately 0.048-0.005, so language quality has a small numerical effect on
structured fitness.

### v2: thesis campaign definition

```yaml
fitness:
  version: v2
```

Lower-is-better metrics use:

```text
quality(x) = 1 / (1 + log1p(max(0, x)))
```

For perplexities in the 20-200 range it yields approximately 0.247-0.159. This
keeps the metric bounded while allowing the configured language-quality weight
to make a material contribution.

The version fixes the transform: `v1` requires `legacy_reciprocal` and `v2`
requires `log_reciprocal`. Mismatched labels are rejected during configuration
validation.

## Comparison Rule

Do not compare, pool, or rank fitness values across versions. GA, random search,
parent baselines, seed replications, and sensitivity runs in one comparison must
all use the same fitness version, task mix, task limits, and evaluation protocol.
Historical `v1` results remain evidence for the historical protocol; they are not
controls for a new `v2` run.

All pending local experiments, EKS presets, and the pre-AWS CPU smoke explicitly
use `v2`. Completed local presets remain unchanged and therefore resolve to
`v1`.
