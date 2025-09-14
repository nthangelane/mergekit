---
base_model:
- EleutherAI/pythia-70m-deduped
library_name: transformers
tags:
- mergekit
- merge

---
# mergedmitznm4e

This is a merge of pre-trained language models created using [mergekit](https://github.com/cg123/mergekit).

## Merge Details
### Merge Method

This model was merged using the [Linear](https://arxiv.org/abs/2203.05482) merge method.

### Models Merged

The following models were included in the merge:
* [EleutherAI/pythia-70m-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped)

### Configuration

The following YAML configuration was used to produce this model:

```yaml
dtype: bfloat16
merge_method: linear
modules:
  default:
    slices:
    - sources:
      - layer_range: [0, 6]
        model: EleutherAI/pythia-70m-deduped
        parameters:
          weight: 0.0
      - layer_range: [0, 6]
        model: EleutherAI/pythia-70m-deduped
        parameters:
          weight: 1.0
      - layer_range: [0, 6]
        model: EleutherAI/pythia-70m-deduped
        parameters:
          weight: 0.0
parameters:
  int8_mask: 1.0
  normalize: 1.0
```
