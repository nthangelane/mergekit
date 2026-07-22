"""Genome expansion helpers for memetic repaired-parent re-entry."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from mergekit.common import ModelReference
from mergekit.evo.multi_method_genome import MultiMethodGenome

LOG = logging.getLogger(__name__)


def capture_genome_geometry(genome) -> Dict[str, Any]:
    if isinstance(genome, MultiMethodGenome):
        return {
            "kind": "multi_method",
            "num_layer_groups": int(genome.num_layer_groups),
            "method_dim": int(genome.method_dim),
            "model_selection_dim": int(genome.model_selection_dim),
            "param_dim": int(genome.param_dim),
            "layer_group_dim": int(genome.layer_group_dim),
        }
    template = genome.initial_genotype(random=False)
    return {"kind": "model", "shape": tuple(int(value) for value in template.shape)}


def register_reentrant_parent(genome, checkpoint_path: str) -> int:
    """Add a checkpoint to a genome definition and return its model index."""
    reference = ModelReference.model_validate(checkpoint_path)
    models = list(genome.definition.models)
    parent_index = len(models)
    if isinstance(genome, MultiMethodGenome):
        # Multi-method genomes expose only the leading max_models entries. Insert
        # at that boundary so the new parent owns the newly padded weight slot.
        parent_index = int(genome.model_selection_dim)
        models.insert(parent_index, reference)
    else:
        models.append(reference)
    updates: Dict[str, Any] = {"models": models}
    if isinstance(genome, MultiMethodGenome):
        updates["max_models_per_layer"] = min(
            len(models), int(genome.model_selection_dim) + 1
        )
    genome.definition = genome.definition.model_copy(update=updates)
    if isinstance(genome, MultiMethodGenome):
        genome.num_models = len(models)
        genome.max_models = min(
            genome.definition.max_models_per_layer, genome.num_models
        )
        genome._calculate_genome_dimensions()
    return parent_index


def align_reentrant_parent_vocab(
    genome,
    checkpoint_path: str,
    *,
    trust_remote_code: bool = False,
) -> bool:
    """Expand a repaired child back to the source parents' padded vocabulary.

    Merge output tokenizers can contain fewer entries than the source model's
    padded embedding table (Pythia is a common example).  A repaired merge is
    therefore loadable and evaluable on its own, but cannot subsequently be
    blended with the original parents unless its embedding matrices are padded
    back to the source width.
    """
    import torch
    import transformers

    source_refs = list(genome.definition.models)
    base_model = getattr(genome.definition, "base_model", None)
    if base_model is not None:
        source_refs.append(base_model)

    source_vocab_sizes = []
    for model_ref in source_refs:
        config = model_ref.config(trust_remote_code=trust_remote_code)
        vocab_size = getattr(config, "vocab_size", None)
        if vocab_size is not None:
            source_vocab_sizes.append(int(vocab_size))
    if not source_vocab_sizes:
        return False

    target_vocab_size = max(source_vocab_sizes)
    child_config = transformers.AutoConfig.from_pretrained(
        checkpoint_path,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    child_vocab_size = getattr(child_config, "vocab_size", None)
    if child_vocab_size is None or int(child_vocab_size) == target_vocab_size:
        return False
    if int(child_vocab_size) > target_vocab_size:
        raise ValueError(
            "Re-entrant checkpoint vocabulary is larger than every source parent: "
            f"{child_vocab_size} > {target_vocab_size}"
        )

    LOG.info(
        "Expanding re-entrant checkpoint vocabulary from %d to %d: %s",
        int(child_vocab_size),
        target_vocab_size,
        checkpoint_path,
    )
    child_model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        child_model.resize_token_embeddings(target_vocab_size)
    child_model.save_pretrained(checkpoint_path, safe_serialization=True)
    del child_model
    return True


def pad_genotype_for_reentry(
    genotype: np.ndarray, geometry: Dict[str, Any]
) -> np.ndarray:
    """Insert a zero-valued model gene while preserving existing decoding."""
    values = np.asarray(genotype, dtype=np.float32)
    flat = values.reshape(-1)
    if geometry["kind"] == "model":
        old_shape = tuple(geometry["shape"])
        structured = flat.reshape(old_shape)
        padded = np.pad(structured, ((0, 0), (0, 1), (0, 0), (0, 0)))
        return padded.astype(np.float32).reshape(-1)

    groups = []
    old_group_dim = int(geometry["layer_group_dim"])
    method_dim = int(geometry["method_dim"])
    model_dim = int(geometry["model_selection_dim"])
    for group_idx in range(int(geometry["num_layer_groups"])):
        group = flat[group_idx * old_group_dim : (group_idx + 1) * old_group_dim]
        insert_at = method_dim + model_dim
        groups.append(np.insert(group, insert_at, 0.0))
    return np.concatenate(groups).astype(np.float32)
