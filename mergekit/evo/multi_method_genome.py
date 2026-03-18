# Copyright (C) 2025 Nkululeko Thangelane
# Multi-method genome for evolving both merge strategies AND parameters

import logging
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import transformers
from pydantic import BaseModel, model_validator

from mergekit.common import ModelReference
from mergekit.config import MergeConfiguration


class InvalidGenotypeError(RuntimeError):
    pass


# Available merge methods as an enum for genetic encoding
class MergeMethod(IntEnum):
    LINEAR = 0
    SLERP = 1
    NUSLERP = 2
    KARCHER = 3
    PASSTHROUGH = 4
    MODEL_STOCK = 5
    ARCEE_FUSION = 6
    TASK_ARITHMETIC = 7
    TIES = 8
    DARE_TIES = 9
    DARE_LINEAR = 10
    BREADCRUMBS = 11
    BREADCRUMBS_TIES = 12
    DELLA = 13
    DELLA_LINEAR = 14


METHOD_NAMES = {
    MergeMethod.LINEAR: "linear",
    MergeMethod.SLERP: "slerp",
    MergeMethod.NUSLERP: "nuslerp",
    MergeMethod.KARCHER: "karcher",
    MergeMethod.PASSTHROUGH: "passthrough",
    MergeMethod.MODEL_STOCK: "model_stock",
    MergeMethod.ARCEE_FUSION: "arcee_fusion",
    MergeMethod.TASK_ARITHMETIC: "task_arithmetic",
    MergeMethod.TIES: "ties",
    MergeMethod.DARE_TIES: "dare_ties",
    MergeMethod.DARE_LINEAR: "dare_linear",
    MergeMethod.BREADCRUMBS: "breadcrumbs",
    MergeMethod.BREADCRUMBS_TIES: "breadcrumbs_ties",
    MergeMethod.DELLA: "della",
    MergeMethod.DELLA_LINEAR: "della_linear",
}

# Parameter counts for each method (maximum parameters needed)
METHOD_PARAM_COUNTS = {
    MergeMethod.LINEAR: 1,  # weight
    MergeMethod.SLERP: 1,  # t
    MergeMethod.NUSLERP: 1,  # weight
    MergeMethod.KARCHER: 1,  # weight
    MergeMethod.PASSTHROUGH: 0,  # no parameters
    MergeMethod.MODEL_STOCK: 1,  # weight
    MergeMethod.ARCEE_FUSION: 0,  # no per-model parameters (dynamic thresholding)
    MergeMethod.TASK_ARITHMETIC: 1,  # weight
    MergeMethod.TIES: 2,  # weight, density
    MergeMethod.DARE_TIES: 2,  # weight, density
    MergeMethod.DARE_LINEAR: 2,  # weight, density
    MergeMethod.BREADCRUMBS: 3,  # weight, density, gamma
    MergeMethod.BREADCRUMBS_TIES: 3,  # weight, density, gamma
    MergeMethod.DELLA: 3,  # weight, density, epsilon
    MergeMethod.DELLA_LINEAR: 3,  # weight, density, epsilon
}

# Model requirements for each method
METHOD_REQUIRES_BASE = {
    MergeMethod.LINEAR: False,
    MergeMethod.SLERP: True,  # needs base model for 2-model interpolation
    MergeMethod.NUSLERP: False,  # optional base model
    MergeMethod.KARCHER: False,
    MergeMethod.PASSTHROUGH: False,
    MergeMethod.MODEL_STOCK: True,  # requires base model for task vector computation
    MergeMethod.ARCEE_FUSION: True,  # requires base model
    MergeMethod.TASK_ARITHMETIC: True,
    MergeMethod.TIES: True,
    MergeMethod.DARE_TIES: True,
    MergeMethod.DARE_LINEAR: True,
    MergeMethod.BREADCRUMBS: True,
    MergeMethod.BREADCRUMBS_TIES: True,
    MergeMethod.DELLA: True,
    MergeMethod.DELLA_LINEAR: True,
}

# Minimum models required for each method
METHOD_MIN_MODELS = {
    MergeMethod.LINEAR: 2,
    MergeMethod.SLERP: 2,  # exactly 2 models
    MergeMethod.NUSLERP: 2,  # exactly 2 models
    MergeMethod.KARCHER: 2,
    MergeMethod.PASSTHROUGH: 1,
    MergeMethod.MODEL_STOCK: 3,  # requires 3+ models
    MergeMethod.ARCEE_FUSION: 2,  # exactly 2 models
    MergeMethod.TASK_ARITHMETIC: 2,
    MergeMethod.TIES: 2,
    MergeMethod.DARE_TIES: 2,
    MergeMethod.DARE_LINEAR: 2,
    MergeMethod.BREADCRUMBS: 2,
    MergeMethod.BREADCRUMBS_TIES: 2,
    MergeMethod.DELLA: 2,
    MergeMethod.DELLA_LINEAR: 2,
}

# Maximum models for each method (None = unlimited)
METHOD_MAX_MODELS = {
    MergeMethod.LINEAR: None,
    MergeMethod.SLERP: 2,  # exactly 2 models
    MergeMethod.NUSLERP: 2,  # exactly 2 models
    MergeMethod.KARCHER: None,
    MergeMethod.PASSTHROUGH: 1,  # exactly 1 model
    MergeMethod.MODEL_STOCK: None,
    MergeMethod.ARCEE_FUSION: 2,  # exactly 2 models
    MergeMethod.TASK_ARITHMETIC: None,
    MergeMethod.TIES: None,
    MergeMethod.DARE_TIES: None,
    MergeMethod.DARE_LINEAR: None,
    MergeMethod.BREADCRUMBS: None,
    MergeMethod.BREADCRUMBS_TIES: None,
    MergeMethod.DELLA: None,
    MergeMethod.DELLA_LINEAR: None,
}


@dataclass
class LayerGroupGenome:
    """Represents the genetic encoding for a single layer group."""

    method: MergeMethod
    model_selection: np.ndarray  # Which models to use (binary mask or weights)
    parameters: np.ndarray  # Method-specific parameters

    def __post_init__(self):
        assert len(self.parameters) == METHOD_PARAM_COUNTS[self.method]


class MultiMethodGenomeDefinition(BaseModel, frozen=True):
    """Definition for a genome that can evolve both methods and parameters."""

    models: List[ModelReference]
    base_model: Optional[ModelReference] = None
    tokenizer_source: Optional[str] = None
    layer_granularity: int = 0
    normalize: Optional[bool] = None
    allow_negative_weights: bool = False
    filters: Optional[List[str]] = None

    # Multi-method specific options
    allowed_methods: List[str] = [
        "linear",
        "task_arithmetic",
        "ties",
        "dare_ties",
        "slerp",
    ]
    max_models_per_layer: int = 4  # Limit model combinations for complexity
    enable_method_evolution: bool = True
    enable_model_selection: bool = True

    @model_validator(mode="after")
    def validate(self):
        valid_methods = set(METHOD_NAMES.values())
        for method in self.allowed_methods:
            assert method in valid_methods, f"Invalid method: {method}"

        # Check base model requirements
        base_required_methods = {
            name
            for method, name in METHOD_NAMES.items()
            if METHOD_REQUIRES_BASE.get(method, False)
        }
        if any(m in base_required_methods for m in self.allowed_methods):
            assert (
                self.base_model is not None
            ), f"base_model required for methods: {[m for m in self.allowed_methods if m in base_required_methods]}"

        # Check model count requirements
        for method_name in self.allowed_methods:
            method = next(m for m, n in METHOD_NAMES.items() if n == method_name)
            min_models = METHOD_MIN_MODELS.get(method, 2)

            available_models = len(self.models)
            max_per_layer = self.max_models_per_layer or available_models

            if available_models < min_models:
                raise ValueError(
                    f"Method {method_name} requires at least {min_models} models, got {available_models}"
                )

            if max_per_layer < min_models:
                raise ValueError(
                    f"Method {method_name} requires at least {min_models} models per layer, but max_models_per_layer is {max_per_layer}"
                )

        return self


class MultiMethodGenome:
    """A genome that can evolve merge methods, model selection, AND parameters."""

    def __init__(
        self, definition: MultiMethodGenomeDefinition, trust_remote_code: bool = False
    ):
        self.definition = definition

        # Get model info
        self._input_config_example = self.definition.models[0].config(
            trust_remote_code=trust_remote_code
        )
        self.num_layers = self._input_config_example.num_hidden_layers

        # Calculate layer groups
        if self.definition.layer_granularity > 0:
            assert self.num_layers % self.definition.layer_granularity == 0
            self.num_layer_groups = self.num_layers // self.definition.layer_granularity
        else:
            self.num_layer_groups = 1

        self.num_models = len(self.definition.models)
        self.max_models = min(self.definition.max_models_per_layer, self.num_models)

        # Calculate genome dimensions
        self._calculate_genome_dimensions()

    def _calculate_genome_dimensions(self):
        """Calculate the total genome size for flat representation."""
        # Per layer group:
        # - 1 value for method selection (if enabled)
        # - max_models values for model selection weights
        # - max(param_counts) values for parameters (padded)

        max_params = max(METHOD_PARAM_COUNTS.values())

        self.method_dim = 1 if self.definition.enable_method_evolution else 0
        self.model_selection_dim = (
            self.max_models
            if self.definition.enable_model_selection
            else self.num_models
        )
        self.param_dim = max_params

        self.layer_group_dim = (
            self.method_dim + self.model_selection_dim + self.param_dim
        )
        self.total_dim = self.num_layer_groups * self.layer_group_dim

    def method_name_from_gene_value(self, method_value: Union[float, int]) -> str:
        """Decode a method gene into the configured method name."""
        if not self.definition.allowed_methods:
            raise ValueError("No allowed_methods configured for multi-method genome")

        idx = int(
            np.clip(float(method_value), 0, len(self.definition.allowed_methods) - 1)
        )
        return self.definition.allowed_methods[idx]

    def method_gene_value(self, method_name: str) -> float:
        """Encode a configured method name into the stored gene value."""
        if method_name not in self.definition.allowed_methods:
            raise ValueError(f"Method {method_name} is not enabled in allowed_methods")
        return float(self.definition.allowed_methods.index(method_name))

    def method_enum_from_name(self, method_name: str) -> MergeMethod:
        return next(
            method for method, name in METHOD_NAMES.items() if name == method_name
        )

    def project_model_selection_for_method(
        self, model_weights: np.ndarray, method_name: str
    ) -> np.ndarray:
        """Project model-selection weights to satisfy the sampled method."""
        weights = np.abs(np.asarray(model_weights, dtype=np.float32))
        if weights.size == 0:
            return weights

        total = float(weights.sum())
        if total <= 1e-8 or not np.isfinite(total):
            weights[:] = 1.0 / float(len(weights))
        else:
            weights /= total

        method_enum = self.method_enum_from_name(method_name)
        max_models = METHOD_MAX_MODELS.get(method_enum, None)
        if max_models is None:
            return weights

        top_k = max(1, int(max_models))
        indices = np.argsort(-weights)[:top_k]
        projected = np.zeros_like(weights)
        projected[indices] = weights[indices]

        if top_k == 1:
            projected[indices[0]] = 1.0
            return projected

        projected_total = float(projected.sum())
        if projected_total <= 1e-8 or not np.isfinite(projected_total):
            projected[indices] = 1.0 / float(top_k)
        else:
            projected /= projected_total
        return projected

    def sample_parameters_for_method(
        self,
        method_name: str,
        rs: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Sample bounded method parameters for adaptive operator selection."""
        random_state = rs if rs is not None else np.random.RandomState()
        method = self.method_enum_from_name(method_name)
        params = np.zeros(self.param_dim, dtype=np.float32)
        param_count = METHOD_PARAM_COUNTS.get(method, 0)
        if param_count == 0:
            return params

        if method_name in {"slerp", "nuslerp"}:
            params[0] = float(random_state.uniform(0.2, 0.8))
        elif method_name in {"linear", "task_arithmetic", "karcher", "model_stock"}:
            params[0] = float(random_state.uniform(0.8, 1.2))
        elif method_name in {"ties", "dare_ties", "dare_linear"}:
            params[0] = float(random_state.uniform(0.8, 1.2))
            if param_count > 1:
                params[1] = float(random_state.uniform(0.2, 0.6))
        else:
            params[:param_count] = random_state.uniform(
                0.0, 1.0, size=param_count
            ).astype(np.float32)

        return self._constrain_parameters(method, params[:param_count]).astype(
            np.float32
        )

    def initial_genotype(self, random: bool = False) -> torch.Tensor:
        """Generate an initial genotype."""
        if random:
            return self._random_genotype()
        else:
            return self._default_genotype()

    def _default_genotype(self) -> torch.Tensor:
        """Create a sensible default genotype."""
        genotype = torch.zeros(self.total_dim)
        default_method_name = (
            "linear"
            if "linear" in self.definition.allowed_methods
            else self.definition.allowed_methods[0]
        )
        default_method = self.method_enum_from_name(default_method_name)
        default_param_count = METHOD_PARAM_COUNTS[default_method]

        for layer_idx in range(self.num_layer_groups):
            offset = layer_idx * self.layer_group_dim

            # Default to a smooth merge method when available, independent of
            # the configured allowed-method ordering.
            if self.method_dim > 0:
                genotype[offset] = self.method_gene_value(default_method_name)

            # Equal model weights
            model_start = offset + self.method_dim
            model_end = model_start + self.model_selection_dim
            projected_weights = self.project_model_selection_for_method(
                np.ones(self.model_selection_dim, dtype=np.float32), default_method_name
            )
            genotype[model_start:model_end] = torch.from_numpy(projected_weights)

            # Default parameters stay within the configured method's safe shape.
            param_start = model_end
            if default_method in {MergeMethod.LINEAR, MergeMethod.SLERP}:
                genotype[param_start] = 0.5
            elif default_method in {
                MergeMethod.TIES,
                MergeMethod.DARE_TIES,
                MergeMethod.TASK_ARITHMETIC,
                MergeMethod.DARE_LINEAR,
            }:
                genotype[param_start] = 0.5
                if default_param_count > 1:
                    genotype[param_start + 1] = 1.0
            elif default_method == MergeMethod.NUSLERP:
                genotype[param_start] = 1.0
            else:
                genotype[param_start] = 1.0 / max(1, self.model_selection_dim)

        return genotype

    def _random_genotype(self) -> torch.Tensor:
        """Create a random genotype."""
        genotype = torch.rand(self.total_dim)

        # Normalize method selection to valid range
        if self.method_dim > 0:
            num_methods = len(self.definition.allowed_methods)
            for layer_idx in range(self.num_layer_groups):
                offset = layer_idx * self.layer_group_dim
                genotype[offset] = genotype[offset] * num_methods

        return genotype

    def decode_genotype(
        self, genotype: Union[torch.Tensor, np.ndarray]
    ) -> List[LayerGroupGenome]:
        """Decode a flat genotype into structured layer group genomes."""
        if isinstance(genotype, np.ndarray):
            # Copy ensures we do not keep a read-only NumPy view that would warn when
            # converted into a tensor during Ray deserialization.
            genotype = torch.tensor(genotype, dtype=torch.float32)
        else:
            genotype = genotype.float()

        layer_groups = []

        for layer_idx in range(self.num_layer_groups):
            offset = layer_idx * self.layer_group_dim

            # Decode method
            if self.method_dim > 0:
                method_val = genotype[offset]
                method_name = self.method_name_from_gene_value(float(method_val))
                method = MergeMethod[method_name.upper()]
            else:
                method = MergeMethod.LINEAR  # Default

            # Decode model selection
            model_start = offset + self.method_dim
            model_end = model_start + self.model_selection_dim
            model_weights = genotype[model_start:model_end].numpy()

            # Normalize and threshold model weights
            if self.definition.enable_model_selection:
                model_weights = np.abs(model_weights)
                total = float(model_weights.sum())
                if not np.isfinite(total) or total <= 1e-8:
                    model_weights = np.ones_like(model_weights)
                    total = float(model_weights.sum())
                model_weights = model_weights / total

                # Keep only the number of models compatible with the chosen method.
                min_models = METHOD_MIN_MODELS.get(method, 2)
                max_models = METHOD_MAX_MODELS.get(method, None)
                top_k = int(np.sum(model_weights > 0.1))
                if top_k < min_models:
                    top_k = min_models
                if max_models is not None:
                    top_k = min(top_k, max_models)
                top_k = min(self.max_models, top_k)
                indices = np.argsort(-model_weights)[:top_k]
                mask = np.zeros_like(model_weights)
                mask[indices] = model_weights[indices]

                masked_total = float(mask.sum())
                if masked_total <= 1e-8 or not np.isfinite(masked_total):
                    mask[:] = 0.0
                    mask[indices] = 1.0
                    masked_total = float(mask.sum())
                model_weights = mask / masked_total
            else:
                # Ensure deterministic, normalized weights when selection is disabled
                total = float(np.sum(model_weights))
                if total <= 1e-8 or not np.isfinite(total):
                    model_weights = np.ones_like(model_weights) / max(
                        1, len(model_weights)
                    )
                else:
                    model_weights = model_weights / total

            # Decode parameters
            param_start = model_end
            param_count = METHOD_PARAM_COUNTS[method]
            parameters = genotype[param_start : param_start + param_count].numpy()
            parameters = np.nan_to_num(parameters, nan=0.0, posinf=1.0, neginf=0.0)

            # Apply parameter constraints
            parameters = self._constrain_parameters(method, parameters)

            layer_groups.append(
                LayerGroupGenome(
                    method=method, model_selection=model_weights, parameters=parameters
                )
            )

        return layer_groups

    def _is_method_compatible(self, method_name: str) -> bool:
        """Check if a method is compatible with the current model configuration."""
        if method_name not in METHOD_NAMES.values():
            return False

        method_enum = next(m for m, n in METHOD_NAMES.items() if n == method_name)
        min_models = METHOD_MIN_MODELS.get(method_enum, 2)
        selectable_models = self.model_selection_dim
        if selectable_models < min_models:
            return False

        # Check base model requirement
        requires_base = METHOD_REQUIRES_BASE.get(method_enum, False)
        if requires_base and self.definition.base_model is None:
            return False

        return True

    def _constrain_parameters(
        self, method: MergeMethod, params: np.ndarray
    ) -> np.ndarray:
        """Apply method-specific parameter constraints."""
        constrained = params.copy()

        if method in [
            MergeMethod.LINEAR,
            MergeMethod.TASK_ARITHMETIC,
            MergeMethod.DARE_LINEAR,
            MergeMethod.DELLA_LINEAR,
        ]:
            # Weight parameter
            if not self.definition.allow_negative_weights:
                constrained[0] = abs(constrained[0])

        elif method in [MergeMethod.TIES, MergeMethod.DARE_TIES, MergeMethod.DELLA]:
            # Weight and density parameters
            if not self.definition.allow_negative_weights:
                constrained[0] = abs(constrained[0])
            constrained[1] = np.clip(constrained[1], 0, 1)  # Density in [0,1]

        elif method in [MergeMethod.BREADCRUMBS, MergeMethod.BREADCRUMBS_TIES]:
            # Weight, density, gamma parameters
            if not self.definition.allow_negative_weights:
                constrained[0] = abs(constrained[0])
            constrained[1] = np.clip(constrained[1], 0, 1)  # Density in [0,1]
            constrained[2] = np.clip(
                constrained[2], 0, 1
            )  # Gamma (outlier threshold) in [0,1]

        elif method in [MergeMethod.SLERP, MergeMethod.NUSLERP]:
            # t parameter in [0,1]
            constrained[0] = np.clip(constrained[0], 0, 1)

        elif method == MergeMethod.KARCHER:
            # Weight parameter for Karcher mean
            if not self.definition.allow_negative_weights:
                constrained[0] = abs(constrained[0])

        elif method == MergeMethod.MODEL_STOCK:
            # Weight parameter for model stock
            if not self.definition.allow_negative_weights:
                constrained[0] = abs(constrained[0])

        elif method == MergeMethod.PASSTHROUGH:
            # No parameters to constrain
            pass

        elif method == MergeMethod.ARCEE_FUSION:
            # No per-model parameters (uses dynamic thresholding)
            pass

        return constrained

    def _layer_range_for_index(self, layer_idx: int) -> Tuple[int, int]:
        if self.definition.layer_granularity > 0:
            start = layer_idx * self.definition.layer_granularity
            end = min(start + self.definition.layer_granularity, self.num_layers)
            return start, end
        return 0, self.num_layers

    def _select_models_for_layer_group(
        self, layer_group: LayerGroupGenome
    ) -> List[Tuple[ModelReference, float]]:
        min_models = METHOD_MIN_MODELS.get(layer_group.method, 1)
        selected_models: List[Tuple[ModelReference, float]] = []
        for i, weight in enumerate(layer_group.model_selection):
            if weight > 1e-6 and i < self.num_models:
                selected_models.append((self.definition.models[i], float(weight)))

        if len(selected_models) < min_models:
            fallback_k = max(1, min_models)
            indices = np.argsort(-layer_group.model_selection)[:fallback_k]
            selected_models = [
                (self.definition.models[i], float(layer_group.model_selection[i]))
                for i in indices
                if i < self.num_models
            ]
        return selected_models

    def _neutral_layer_group_for_method(
        self, method: MergeMethod, reference_group: LayerGroupGenome
    ) -> LayerGroupGenome:
        model_selection = np.zeros_like(
            reference_group.model_selection, dtype=np.float32
        )
        preferred_idx = 0
        if self.definition.base_model is not None:
            for idx, model_ref in enumerate(self.definition.models):
                if model_ref == self.definition.base_model:
                    preferred_idx = idx
                    break
        elif len(reference_group.model_selection) > 0:
            preferred_idx = int(np.argmax(reference_group.model_selection))

        max_models = METHOD_MAX_MODELS.get(method, None)
        min_models = METHOD_MIN_MODELS.get(method, 1)
        if max_models == 1 or method == MergeMethod.PASSTHROUGH:
            model_selection[preferred_idx] = 1.0
        elif max_models == 2:
            available = [
                idx for idx in range(min(self.num_models, len(model_selection)))
            ]
            if preferred_idx in available:
                available.remove(preferred_idx)
            secondary_idx = available[0] if available else preferred_idx
            model_selection[preferred_idx] = 1.0
            if secondary_idx != preferred_idx:
                model_selection[secondary_idx] = 1.0
        else:
            top_k = min(max(min_models, 1), len(model_selection))
            if top_k <= 1:
                model_selection[preferred_idx] = 1.0
            else:
                indices = np.argsort(-reference_group.model_selection)[:top_k]
                if len(indices) == 0:
                    model_selection[preferred_idx] = 1.0
                else:
                    model_selection[indices] = 1.0

        total = float(model_selection.sum())
        if total <= 1e-8:
            model_selection[preferred_idx] = 1.0
            total = float(model_selection.sum())
        model_selection /= total

        params = np.zeros(METHOD_PARAM_COUNTS.get(method, 0), dtype=np.float32)
        if method in [
            MergeMethod.LINEAR,
            MergeMethod.TASK_ARITHMETIC,
            MergeMethod.KARCHER,
            MergeMethod.MODEL_STOCK,
        ]:
            params[0] = 1.0
        elif method in [
            MergeMethod.TIES,
            MergeMethod.DARE_TIES,
            MergeMethod.DARE_LINEAR,
            MergeMethod.DELLA_LINEAR,
        ]:
            if len(params) > 0:
                params[0] = 1.0
            if len(params) > 1:
                params[1] = 1.0
        elif method in [
            MergeMethod.BREADCRUMBS,
            MergeMethod.BREADCRUMBS_TIES,
            MergeMethod.DELLA,
        ]:
            if len(params) > 0:
                params[0] = 1.0
            if len(params) > 1:
                params[1] = 1.0
            if len(params) > 2:
                params[2] = 0.5
        elif method in [MergeMethod.SLERP, MergeMethod.NUSLERP]:
            params[0] = 0.0

        params = self._constrain_parameters(method, params)
        return LayerGroupGenome(
            method=method,
            model_selection=model_selection,
            parameters=params,
        )

    def _slice_entry_for_layer_group(
        self,
        layer_group: LayerGroupGenome,
        layer_idx: int,
        forced_method: Optional[MergeMethod] = None,
    ) -> Dict[str, Any]:
        method = forced_method or layer_group.method
        group_for_slice = (
            layer_group
            if forced_method is None or layer_group.method == forced_method
            else self._neutral_layer_group_for_method(forced_method, layer_group)
        )
        method_name = METHOD_NAMES[method]
        selected_models = self._select_models_for_layer_group(group_for_slice)
        start, end = self._layer_range_for_index(layer_idx)
        slice_entry: Dict[str, Any] = {"sources": [], "merge_method": method_name}

        if method_name in ["linear", "task_arithmetic", "karcher", "model_stock"]:
            for model_ref, weight in selected_models:
                weight_value = float(weight * group_for_slice.parameters[0])
                slice_entry["sources"].append(
                    {
                        "model": model_ref,
                        "layer_range": [start, end],
                        "parameters": {
                            "weight": float(np.nan_to_num(weight_value, nan=0.0))
                        },
                    }
                )
            slice_entry["parameters"] = {"normalize": True, "int8_mask": True}
        elif method_name in ["ties", "dare_ties", "dare_linear", "della_linear"]:
            density_value = (
                float(np.clip(group_for_slice.parameters[1], 0.0, 1.0))
                if len(group_for_slice.parameters) > 1
                else 1.0
            )
            for model_ref, weight in selected_models:
                weight_value = float(weight * group_for_slice.parameters[0])
                params = {
                    "weight": float(np.nan_to_num(weight_value, nan=0.0)),
                    "density": density_value,
                }
                if (
                    method_name == "della_linear"
                    and len(group_for_slice.parameters) > 2
                ):
                    params["epsilon"] = float(
                        np.clip(group_for_slice.parameters[2], 0.0, 1.0)
                    )
                slice_entry["sources"].append(
                    {
                        "model": model_ref,
                        "layer_range": [start, end],
                        "parameters": params,
                    }
                )
            slice_entry.setdefault("parameters", {})
            slice_entry["parameters"].update({"normalize": True, "int8_mask": True})
        elif method_name in ["breadcrumbs", "breadcrumbs_ties", "della"]:
            density_value = (
                float(np.clip(group_for_slice.parameters[1], 0.0, 1.0))
                if len(group_for_slice.parameters) > 1
                else 1.0
            )
            third_value = (
                float(np.clip(group_for_slice.parameters[2], 0.0, 1.0))
                if len(group_for_slice.parameters) > 2
                else 0.5
            )
            for model_ref, weight in selected_models:
                weight_value = float(weight * group_for_slice.parameters[0])
                params = {
                    "weight": float(np.nan_to_num(weight_value, nan=0.0)),
                    "density": density_value,
                }
                if method_name == "della":
                    params["epsilon"] = third_value
                else:
                    params["gamma"] = third_value
                slice_entry["sources"].append(
                    {
                        "model": model_ref,
                        "layer_range": [start, end],
                        "parameters": params,
                    }
                )
        elif method_name == "slerp":
            ordered = sorted(selected_models, key=lambda item: item[1], reverse=True)[
                :2
            ]
            if len(ordered) < 2:
                raise InvalidGenotypeError("SLERP requires at least 2 selected models")
            (model1, weight1), (model2, weight2) = ordered
            slice_entry = {
                "sources": [
                    {
                        "model": model1,
                        "layer_range": [start, end],
                        "parameters": {"weight": float(weight1)},
                    },
                    {
                        "model": model2,
                        "layer_range": [start, end],
                        "parameters": {"weight": float(weight2)},
                    },
                ],
                "merge_method": method_name,
                "parameters": {
                    "t": float(np.clip(group_for_slice.parameters[0], 0, 1))
                },
            }
        elif method_name == "nuslerp":
            filtered = [
                (model, weight)
                for model, weight in selected_models
                if model != self.definition.base_model
            ]
            ordered = sorted(filtered, key=lambda item: abs(item[1]), reverse=True)[:2]
            if len(ordered) < 2:
                raise InvalidGenotypeError(
                    "NuSLERP requires at least 2 non-base selected models"
                )
            scale = (
                float(group_for_slice.parameters[0])
                if len(group_for_slice.parameters)
                else 1.0
            )
            slice_entry["sources"] = [
                {
                    "model": model_ref,
                    "layer_range": [start, end],
                    "parameters": {"weight": float(weight * scale)},
                }
                for model_ref, weight in ordered
            ]
            slice_entry["parameters"] = {
                "nuslerp_row_wise": False,
                "nuslerp_flatten": False,
            }
        elif method_name == "passthrough":
            model_ref = selected_models[0][0]
            slice_entry["sources"] = [{"model": model_ref, "layer_range": [start, end]}]
        elif method_name == "arcee_fusion":
            for model_ref, _weight in selected_models[:2]:
                slice_entry["sources"].append(
                    {
                        "model": model_ref,
                        "layer_range": [start, end],
                    }
                )
        else:
            raise InvalidGenotypeError(
                f"Unsupported method for slice config: {method_name}"
            )

        return slice_entry

    def _config_common_fields(self) -> Dict[str, Any]:
        config_dict: Dict[str, Any] = {"dtype": "bfloat16"}
        if self.definition.base_model is not None:
            config_dict["base_model"] = self.definition.base_model
        if self.definition.tokenizer_source:
            config_dict["tokenizer_source"] = self.definition.tokenizer_source
        elif self.definition.base_model is not None:
            config_dict["tokenizer_source"] = self.definition.base_model
        return config_dict

    def _single_method_layered_config(
        self,
        method: MergeMethod,
        layer_groups: List[LayerGroupGenome],
    ) -> MergeConfiguration:
        method_name = METHOD_NAMES[method]
        slices = [
            self._slice_entry_for_layer_group(layer_group, layer_idx)
            for layer_idx, layer_group in enumerate(layer_groups)
        ]
        config_dict: Dict[str, Any] = {
            "merge_method": method_name,
            "slices": slices,
            **self._config_common_fields(),
        }
        if method_name in {
            "linear",
            "ties",
            "dare_ties",
            "task_arithmetic",
            "dare_linear",
        }:
            config_dict["parameters"] = {"normalize": True, "int8_mask": True}
        return MergeConfiguration.model_validate(config_dict)

    def _mixed_method_plan(
        self, layer_groups: List[LayerGroupGenome]
    ) -> Dict[str, Any]:
        methods_in_order: List[MergeMethod] = []
        for layer_group in layer_groups:
            if layer_group.method not in methods_in_order:
                methods_in_order.append(layer_group.method)

        components: List[Dict[str, Any]] = []
        for method in methods_in_order:
            component_name = METHOD_NAMES[method]
            component_slices = [
                self._slice_entry_for_layer_group(
                    layer_group,
                    layer_idx,
                    forced_method=method,
                )
                for layer_idx, layer_group in enumerate(layer_groups)
            ]
            component_config = MergeConfiguration.model_validate(
                {
                    "merge_method": component_name,
                    "slices": component_slices,
                    **self._config_common_fields(),
                    "parameters": (
                        {"normalize": True, "int8_mask": True}
                        if component_name
                        in {
                            "linear",
                            "ties",
                            "dare_ties",
                            "task_arithmetic",
                            "dare_linear",
                        }
                        else None
                    ),
                }
            )
            components.append(
                {
                    "name": component_name,
                    "config": component_config,
                }
            )

        final_slices: List[Dict[str, Any]] = []
        current_component = METHOD_NAMES[layer_groups[0].method]
        current_start, current_end = self._layer_range_for_index(0)
        for layer_idx in range(1, len(layer_groups)):
            component_name = METHOD_NAMES[layer_groups[layer_idx].method]
            start, end = self._layer_range_for_index(layer_idx)
            if component_name == current_component and start == current_end:
                current_end = end
                continue
            final_slices.append(
                {
                    "component": current_component,
                    "layer_range": [current_start, current_end],
                }
            )
            current_component = component_name
            current_start, current_end = start, end
        final_slices.append(
            {
                "component": current_component,
                "layer_range": [current_start, current_end],
            }
        )

        return {
            "kind": "layered",
            "components": components,
            "final_slices": final_slices,
            "methods": [
                METHOD_NAMES[layer_group.method] for layer_group in layer_groups
            ],
            "tokenizer_source": self.definition.tokenizer_source
            or self.definition.base_model,
        }

    def genotype_to_merge_plan(
        self, genotype: Union[torch.Tensor, np.ndarray]
    ) -> Dict[str, Any]:
        layer_groups = self.decode_genotype(genotype)
        unique_methods = {layer_group.method for layer_group in layer_groups}
        if len(unique_methods) == 1:
            return {
                "kind": "config",
                "config": self._single_method_layered_config(
                    layer_groups[0].method, layer_groups
                ),
                "methods": [METHOD_NAMES[layer_groups[0].method]],
            }
        config = self._complex_config(layer_groups)
        return {
            "kind": "config",
            "config": config,
            "methods": [
                METHOD_NAMES[layer_group.method] for layer_group in layer_groups
            ],
        }

    def method_label_for_genotype(
        self, genotype: Union[torch.Tensor, np.ndarray]
    ) -> str:
        layer_groups = self.decode_genotype(genotype)
        unique_methods = []
        for layer_group in layer_groups:
            method_name = METHOD_NAMES[layer_group.method]
            if method_name not in unique_methods:
                unique_methods.append(method_name)
        if len(unique_methods) == 1:
            return unique_methods[0]
        return "layered_mixed"

    def execution_plan_dict(
        self, genotype: Union[torch.Tensor, np.ndarray]
    ) -> Dict[str, Any]:
        plan = self.genotype_to_merge_plan(genotype)
        if plan["kind"] == "config":
            config = plan["config"]
            return {
                "kind": "config",
                "merge_method": config.merge_method,
                "config": config.model_dump(exclude_defaults=True, mode="json"),
            }
        return {
            "kind": "layered",
            "components": [
                {
                    "name": component["name"],
                    "config": component["config"].model_dump(
                        exclude_defaults=True, mode="json"
                    ),
                }
                for component in plan["components"]
            ],
            "final_slices": plan["final_slices"],
            "methods": plan["methods"],
            "tokenizer_source": (
                str(plan["tokenizer_source"])
                if plan.get("tokenizer_source") is not None
                else None
            ),
        }

    def genotype_to_merge_config(
        self, genotype: Union[torch.Tensor, np.ndarray]
    ) -> MergeConfiguration:
        """Convert genotype to a MergeConfiguration."""
        plan = self.genotype_to_merge_plan(genotype)
        if plan["kind"] != "config":
            raise InvalidGenotypeError(
                "Mixed-method layered genotypes require execution via genotype_to_merge_plan"
            )
        return plan["config"]

    def _simple_config(self, layer_group: LayerGroupGenome) -> MergeConfiguration:
        """Create a simple config when all layers use the same method."""
        method_name = METHOD_NAMES[layer_group.method]
        min_models = METHOD_MIN_MODELS.get(layer_group.method, 1)

        # Select models based on weights
        selected_models = []
        for i, weight in enumerate(layer_group.model_selection):
            if weight > 1e-6 and i < self.num_models:
                selected_models.append((self.definition.models[i], weight))

        if len(selected_models) < min_models:
            fallback_k = max(1, min_models)
            indices = np.argsort(-layer_group.model_selection)[:fallback_k]
            selected_models = [
                (self.definition.models[i], layer_group.model_selection[i])
                for i in indices
                if i < self.num_models
            ]

        # Build model configs
        models = []
        weight_values: List[float] = []
        for model_ref, weight in selected_models:
            model_config: Dict[str, Any] = {"model": model_ref}

            # Add parameters
            if method_name in ["linear", "task_arithmetic", "karcher", "model_stock"]:
                weight_value = float(weight * layer_group.parameters[0])
                if not np.isfinite(weight_value):
                    weight_value = 0.0
                model_config["parameters"] = {"weight": weight_value}
                weight_values.append(weight_value)
            elif method_name in ["ties", "dare_ties"]:
                model_config["parameters"] = {
                    "weight": float(
                        np.nan_to_num(weight * layer_group.parameters[0], nan=0.0)
                    ),
                    "density": float(np.clip(layer_group.parameters[1], 0.0, 1.0)),
                }
                weight_values.append(float(model_config["parameters"]["weight"]))
            elif method_name in ["dare_linear", "della_linear"]:
                weight_val = float(
                    np.nan_to_num(weight * layer_group.parameters[0], nan=0.0)
                )
                model_config["parameters"] = {
                    "weight": weight_val,
                    "density": float(np.clip(layer_group.parameters[1], 0.0, 1.0)),
                }
                # Additional epsilon parameter for DELLA linear variants
                if method_name == "della_linear" and len(layer_group.parameters) > 2:
                    model_config["parameters"]["epsilon"] = float(
                        np.clip(layer_group.parameters[2], 0.0, 1.0)
                    )
                weight_values.append(weight_val)
            elif method_name in ["breadcrumbs", "breadcrumbs_ties"]:
                weight_val = float(
                    np.nan_to_num(weight * layer_group.parameters[0], nan=0.0)
                )
                model_config["parameters"] = {
                    "weight": weight_val,
                    "density": float(np.clip(layer_group.parameters[1], 0.0, 1.0)),
                    "gamma": float(np.clip(layer_group.parameters[2], 0.0, 1.0)),
                }
                weight_values.append(weight_val)
            elif method_name == "della":
                weight_val = float(
                    np.nan_to_num(weight * layer_group.parameters[0], nan=0.0)
                )
                model_config["parameters"] = {
                    "weight": weight_val,
                    "density": float(np.clip(layer_group.parameters[1], 0.0, 1.0)),
                    "epsilon": float(np.clip(layer_group.parameters[2], 0.0, 1.0)),
                }
                weight_values.append(weight_val)
            elif method_name == "slerp":
                # SLERP is handled differently - return a slice-based config
                return self._slerp_config(selected_models, layer_group.parameters[0])
            elif method_name == "nuslerp":
                return self._nuslerp_config(layer_group, selected_models)
            elif method_name == "passthrough":
                # No additional parameters required for passthrough, but keep model reference
                pass
            elif method_name == "arcee_fusion":
                pass

            models.append(model_config)

        if weight_values:
            total_weight = float(sum(weight_values))
            if total_weight <= 1e-8 or not np.isfinite(total_weight):
                equal_weight = 1.0 / len(weight_values)
                for model_config in models:
                    if (
                        "parameters" in model_config
                        and "weight" in model_config["parameters"]
                    ):
                        model_config["parameters"]["weight"] = float(equal_weight)

        config_dict: Dict[str, Any] = {
            "merge_method": method_name,
            "models": models,
            "dtype": "bfloat16",
        }

        if self.definition.base_model:
            config_dict["base_model"] = self.definition.base_model
        if self.definition.tokenizer_source:
            config_dict["tokenizer_source"] = self.definition.tokenizer_source
        elif self.definition.base_model:
            config_dict["tokenizer_source"] = self.definition.base_model

        return MergeConfiguration.model_validate(config_dict)

    def _complex_config(
        self, layer_groups: List[LayerGroupGenome]
    ) -> MergeConfiguration:
        """Create a complex slice-based config for different methods per layer."""
        slices = [
            self._slice_entry_for_layer_group(layer_group, layer_idx)
            for layer_idx, layer_group in enumerate(layer_groups)
        ]
        first_method_name = METHOD_NAMES[layer_groups[0].method]
        config_dict: Dict[str, Any] = {
            "merge_method": first_method_name,
            "slices": slices,
            **self._config_common_fields(),
        }
        return MergeConfiguration.model_validate(config_dict)

    def _slerp_config(
        self,
        selected_models: List[Tuple[ModelReference, float]],
        t: float,
        layer_idx: int = 0,
    ) -> MergeConfiguration:
        """Create a SLERP configuration."""
        if len(selected_models) < 2:
            raise ValueError("SLERP requires at least 2 models")

        # Use top 2 models for SLERP ordered by weight
        ordered = sorted(selected_models, key=lambda item: item[1], reverse=True)[:2]
        (model1, weight1), (model2, weight2) = ordered

        # Determine layer range. When we have multiple layer groups, fall back to
        # the full model range so the resulting configuration keeps
        # num_hidden_layers consistent with the base architecture. Downstream
        # support for per-block SLERP will reintroduce narrower ranges once the
        # complex config path is implemented.
        if self.definition.layer_granularity > 0 and self.num_layer_groups > 1:
            start, end = 0, self.num_layers
        elif self.definition.layer_granularity > 0:
            start = layer_idx * self.definition.layer_granularity
            end = min(start + self.definition.layer_granularity, self.num_layers)
        else:
            start, end = 0, self.num_layers

        slice_entry: Dict[str, Any] = {
            "sources": [
                {
                    "model": model1,
                    "layer_range": [start, end],
                    "parameters": {"weight": float(weight1)},
                },
                {
                    "model": model2,
                    "layer_range": [start, end],
                    "parameters": {"weight": float(weight2)},
                },
            ],
            "parameters": {"t": float(np.clip(t, 0, 1))},
        }

        config_dict: Dict[str, Any] = {
            "merge_method": "slerp",
            "slices": [slice_entry],
            "dtype": "bfloat16",
        }

        # Ensure base model is recorded for downstream tooling
        if self.definition.base_model is not None:
            config_dict["base_model"] = self.definition.base_model
        else:
            config_dict["base_model"] = model1

        if self.definition.tokenizer_source:
            config_dict["tokenizer_source"] = self.definition.tokenizer_source
        elif self.definition.base_model is not None:
            config_dict["tokenizer_source"] = self.definition.base_model

        return MergeConfiguration.model_validate(config_dict)

    def _nuslerp_config(
        self,
        layer_group: LayerGroupGenome,
        selected_models: List[Tuple[ModelReference, float]],
    ) -> MergeConfiguration:
        """Create a NuSLERP configuration ensuring the base model is excluded from sources."""

        filtered = [
            (model, weight)
            for model, weight in selected_models
            if model != self.definition.base_model
        ]

        if len(filtered) < 2:
            weights = layer_group.model_selection
            ordered_idx = np.argsort(-np.abs(weights))
            filtered = []
            for idx in ordered_idx:
                if idx >= self.num_models:
                    continue
                model_ref = self.definition.models[idx]
                if model_ref == self.definition.base_model:
                    continue
                filtered.append((model_ref, weights[idx]))
                if len(filtered) == 2:
                    break

        if len(filtered) < 2:
            for model_ref in self.definition.models:
                if model_ref == self.definition.base_model:
                    continue
                if any(existing[0] == model_ref for existing in filtered):
                    continue
                filtered.append((model_ref, 1.0))
                if len(filtered) == 2:
                    break

        if len(filtered) < 2:
            raise ValueError(
                "NuSLERP requires at least two non-base models to interpolate"
            )

        ordered = sorted(filtered, key=lambda item: abs(item[1]), reverse=True)[:2]
        scale = float(layer_group.parameters[0])

        models: List[Dict[str, Any]] = []
        for model_ref, weight in ordered:
            models.append(
                {
                    "model": model_ref,
                    "parameters": {"weight": float(weight * scale)},
                }
            )

        config_dict: Dict[str, Any] = {
            "merge_method": "nuslerp",
            "models": models,
            "dtype": "bfloat16",
        }

        if self.definition.base_model is not None:
            config_dict["base_model"] = self.definition.base_model

        if self.definition.tokenizer_source:
            config_dict["tokenizer_source"] = self.definition.tokenizer_source
        elif self.definition.base_model is not None:
            config_dict["tokenizer_source"] = self.definition.base_model

        return MergeConfiguration.model_validate(config_dict)

    def crossover_semantic(
        self, parent1: torch.Tensor, parent2: torch.Tensor
    ) -> torch.Tensor:
        """Semantic crossover that understands merge structure."""
        child = torch.zeros_like(parent1)

        # Decode both parents
        lg1 = self.decode_genotype(parent1)
        lg2 = self.decode_genotype(parent2)

        for layer_idx in range(self.num_layer_groups):
            offset = layer_idx * self.layer_group_dim

            # Method crossover: choose compatible method
            if self.method_dim > 0:
                if np.random.random() < 0.5:
                    chosen_method = lg1[layer_idx].method
                    source_lg = lg1[layer_idx]
                    child[offset] = parent1[offset]
                else:
                    chosen_method = lg2[layer_idx].method
                    source_lg = lg2[layer_idx]
                    child[offset] = parent2[offset]

                # Validate method compatibility with current models
                min_models = METHOD_MIN_MODELS.get(chosen_method, 2)
                if self.model_selection_dim < min_models:
                    # Fall back to a compatible method
                    compatible_methods = [
                        m
                        for m in self.definition.allowed_methods
                        if self._is_method_compatible(m)
                    ]
                    if compatible_methods:
                        method_name = np.random.choice(compatible_methods)
                        child[offset] = self.method_gene_value(str(method_name))
            else:
                source_lg = lg1[layer_idx]

            # Model selection crossover: blend or swap
            model_start = offset + self.method_dim
            model_end = model_start + self.model_selection_dim

            # Check if selected method requires specific model count
            current_method_idx = int(child[offset]) if self.method_dim > 0 else 0
            if current_method_idx < len(self.definition.allowed_methods):
                method_name = self.definition.allowed_methods[current_method_idx]
                method_enum = next(
                    m for m, n in METHOD_NAMES.items() if n == method_name
                )
                max_models_for_method = METHOD_MAX_MODELS.get(method_enum, None)

                if max_models_for_method and max_models_for_method <= 2:
                    # For methods that need exactly 1-2 models, use winner-take-all
                    if np.random.random() < 0.5:
                        child[model_start:model_end] = parent1[model_start:model_end]
                    else:
                        child[model_start:model_end] = parent2[model_start:model_end]
                else:
                    # For methods that can handle multiple models, blend
                    if np.random.random() < 0.3:  # Blend
                        alpha = np.random.random()
                        child[model_start:model_end] = (
                            alpha * parent1[model_start:model_end]
                            + (1 - alpha) * parent2[model_start:model_end]
                        )
                    else:  # Swap segments
                        if np.random.random() < 0.5:
                            child[model_start:model_end] = parent1[
                                model_start:model_end
                            ]
                        else:
                            child[model_start:model_end] = parent2[
                                model_start:model_end
                            ]
            else:
                # Default blending
                alpha = np.random.random()
                child[model_start:model_end] = (
                    alpha * parent1[model_start:model_end]
                    + (1 - alpha) * parent2[model_start:model_end]
                )

            # Parameter crossover: method-aware
            param_start = model_end
            param_count = METHOD_PARAM_COUNTS.get(source_lg.method, 1)

            # Use arithmetic crossover for parameters of the same method
            alpha = np.random.random()
            child[param_start : param_start + param_count] = (
                alpha * parent1[param_start : param_start + param_count]
                + (1 - alpha) * parent2[param_start : param_start + param_count]
            )

        return child

    def mutate_semantic(
        self,
        genotype: torch.Tensor,
        mutation_rate: float = 0.1,
        mutation_sigma: float = 0.05,
    ) -> torch.Tensor:
        """Semantic mutation that understands merge structure."""
        mutated = genotype.clone()

        for layer_idx in range(self.num_layer_groups):
            offset = layer_idx * self.layer_group_dim

            # Method mutation - switch to compatible method
            if self.method_dim > 0 and np.random.random() < mutation_rate:
                compatible_methods = [
                    m
                    for m in self.definition.allowed_methods
                    if self._is_method_compatible(m)
                ]
                if compatible_methods:
                    new_method_name = np.random.choice(compatible_methods)
                    mutated[offset] = self.method_gene_value(str(new_method_name))

            # Model selection mutation
            model_start = offset + self.method_dim
            model_end = model_start + self.model_selection_dim
            if np.random.random() < mutation_rate:
                # Get current method to check constraints
                current_method_idx = int(mutated[offset]) if self.method_dim > 0 else 0
                if current_method_idx < len(self.definition.allowed_methods):
                    method_name = self.definition.allowed_methods[current_method_idx]
                    method_enum = next(
                        m for m, n in METHOD_NAMES.items() if n == method_name
                    )
                    max_models = METHOD_MAX_MODELS.get(method_enum, None)
                    min_models = METHOD_MIN_MODELS.get(method_enum, 2)

                    if max_models == 1:
                        # For passthrough, select one model
                        model_weights = torch.zeros(self.model_selection_dim)
                        if self.model_selection_dim > 0:
                            selected_model = np.random.randint(
                                min(
                                    self.model_selection_dim,
                                    len(self.definition.models),
                                )
                            )
                            model_weights[selected_model] = 1.0
                        mutated[model_start:model_end] = model_weights
                    elif max_models == 2:
                        # For methods like SLERP, select exactly 2 models
                        model_weights = torch.zeros(self.model_selection_dim)
                        if self.model_selection_dim >= 2:
                            selected_models = np.random.choice(
                                min(
                                    self.model_selection_dim,
                                    len(self.definition.models),
                                ),
                                size=2,
                                replace=False,
                            )
                            for i, model_idx in enumerate(selected_models):
                                model_weights[model_idx] = 0.5 + np.random.normal(
                                    0, 0.1
                                )
                            model_weights = torch.abs(model_weights)
                            model_weights /= model_weights.sum() + 1e-8
                        mutated[model_start:model_end] = model_weights
                    else:
                        # For methods that can handle multiple models, add noise
                        noise = torch.randn(self.model_selection_dim) * mutation_sigma
                        mutated[model_start:model_end] += noise
                        # Ensure positive and normalized
                        mutated[model_start:model_end] = torch.abs(
                            mutated[model_start:model_end]
                        )
                        mutated[model_start:model_end] /= (
                            mutated[model_start:model_end].sum() + 1e-8
                        )

            # Parameter mutation
            param_start = model_end
            current_method_idx = int(mutated[offset]) if self.method_dim > 0 else 0
            if current_method_idx < len(self.definition.allowed_methods):
                method_name = self.definition.allowed_methods[current_method_idx]
                method_enum = next(
                    m for m, n in METHOD_NAMES.items() if n == method_name
                )
                param_count = METHOD_PARAM_COUNTS.get(method_enum, 1)

                for p in range(param_count):
                    if np.random.random() < mutation_rate:
                        mutated[param_start + p] += np.random.normal(0, mutation_sigma)

                # Apply constraints
                if param_count > 0:
                    params = mutated[param_start : param_start + param_count].numpy()
                    constrained = self._constrain_parameters(method_enum, params)
                    mutated[param_start : param_start + param_count] = torch.from_numpy(
                        constrained
                    )

        return mutated
