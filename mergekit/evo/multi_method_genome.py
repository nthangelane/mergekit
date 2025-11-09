# Copyright (C) 2025 Nkululeko Thangelane
# Multi-method genome for evolving both merge strategies AND parameters

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import IntEnum
from dataclasses import dataclass

import numpy as np
import torch
import transformers
from pydantic import BaseModel, model_validator

from mergekit.common import ModelReference
from mergekit.config import MergeConfiguration

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
    MergeMethod.DELLA_LINEAR: "della_linear"
}

# Parameter counts for each method (maximum parameters needed)
METHOD_PARAM_COUNTS = {
    MergeMethod.LINEAR: 1,           # weight
    MergeMethod.SLERP: 1,            # t
    MergeMethod.NUSLERP: 1,          # weight  
    MergeMethod.KARCHER: 1,          # weight
    MergeMethod.PASSTHROUGH: 0,      # no parameters
    MergeMethod.MODEL_STOCK: 1,      # weight
    MergeMethod.ARCEE_FUSION: 0,     # no per-model parameters (dynamic thresholding)
    MergeMethod.TASK_ARITHMETIC: 1,  # weight
    MergeMethod.TIES: 2,             # weight, density
    MergeMethod.DARE_TIES: 2,        # weight, density
    MergeMethod.DARE_LINEAR: 2,      # weight, density
    MergeMethod.BREADCRUMBS: 3,      # weight, density, gamma
    MergeMethod.BREADCRUMBS_TIES: 3, # weight, density, gamma
    MergeMethod.DELLA: 3,            # weight, density, epsilon
    MergeMethod.DELLA_LINEAR: 3,     # weight, density, epsilon
}

# Model requirements for each method
METHOD_REQUIRES_BASE = {
    MergeMethod.LINEAR: False,
    MergeMethod.SLERP: True,         # needs base model for 2-model interpolation
    MergeMethod.NUSLERP: False,      # optional base model
    MergeMethod.KARCHER: False,
    MergeMethod.PASSTHROUGH: False,
    MergeMethod.MODEL_STOCK: True,   # requires base model for task vector computation
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
    MergeMethod.SLERP: 2,            # exactly 2 models
    MergeMethod.NUSLERP: 2,          # exactly 2 models
    MergeMethod.KARCHER: 2,
    MergeMethod.PASSTHROUGH: 1,
    MergeMethod.MODEL_STOCK: 3,      # requires 3+ models
    MergeMethod.ARCEE_FUSION: 2,     # exactly 2 models  
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
    MergeMethod.SLERP: 2,            # exactly 2 models
    MergeMethod.NUSLERP: 2,          # exactly 2 models
    MergeMethod.KARCHER: None,
    MergeMethod.PASSTHROUGH: 1,      # exactly 1 model
    MergeMethod.MODEL_STOCK: None,
    MergeMethod.ARCEE_FUSION: 2,     # exactly 2 models
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
    parameters: np.ndarray       # Method-specific parameters
    
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
    allowed_methods: List[str] = ["linear", "task_arithmetic", "ties", "dare_ties", "slerp"]
    max_models_per_layer: int = 4  # Limit model combinations for complexity
    enable_method_evolution: bool = True
    enable_model_selection: bool = True

    @model_validator(mode="after")
    def validate(self):
        valid_methods = set(METHOD_NAMES.values())
        for method in self.allowed_methods:
            assert method in valid_methods, f"Invalid method: {method}"
        
        # Check base model requirements
        base_required_methods = {name for method, name in METHOD_NAMES.items() 
                               if METHOD_REQUIRES_BASE.get(method, False)}
        if any(m in base_required_methods for m in self.allowed_methods):
            assert self.base_model is not None, f"base_model required for methods: {[m for m in self.allowed_methods if m in base_required_methods]}"
        
        # Check model count requirements
        for method_name in self.allowed_methods:
            method = next(m for m, n in METHOD_NAMES.items() if n == method_name)
            min_models = METHOD_MIN_MODELS.get(method, 2)
            max_models = METHOD_MAX_MODELS.get(method, None)
            
            available_models = len(self.models)
            max_per_layer = self.max_models_per_layer or available_models
            effective_upper = min(available_models, max_per_layer)

            if available_models < min_models:
                raise ValueError(
                    f"Method {method_name} requires at least {min_models} models, got {available_models}"
                )

            if max_per_layer < min_models:
                raise ValueError(
                    f"Method {method_name} requires at least {min_models} models per layer, but max_models_per_layer is {max_per_layer}"
                )

            if max_models is not None and effective_upper > max_models:
                raise ValueError(
                    f"Method {method_name} supports at most {max_models} models per layer, "
                    f"but up to {effective_upper} would be selected. Reduce max_models_per_layer or remove the method."
                )
        
        return self


class MultiMethodGenome:
    """A genome that can evolve merge methods, model selection, AND parameters."""
    
    def __init__(self, definition: MultiMethodGenomeDefinition, trust_remote_code: bool = False):
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
        self.model_selection_dim = self.max_models if self.definition.enable_model_selection else self.num_models
        self.param_dim = max_params
        
        self.layer_group_dim = self.method_dim + self.model_selection_dim + self.param_dim
        self.total_dim = self.num_layer_groups * self.layer_group_dim
        
    def initial_genotype(self, random: bool = False) -> torch.Tensor:
        """Generate an initial genotype."""
        if random:
            return self._random_genotype()
        else:
            return self._default_genotype()
            
    def _default_genotype(self) -> torch.Tensor:
        """Create a sensible default genotype."""
        genotype = torch.zeros(self.total_dim)
        
        for layer_idx in range(self.num_layer_groups):
            offset = layer_idx * self.layer_group_dim
            
            # Default method (linear = 0)
            if self.method_dim > 0:
                genotype[offset] = 0  # MergeMethod.LINEAR
                
            # Equal model weights
            model_start = offset + self.method_dim
            model_end = model_start + self.model_selection_dim
            genotype[model_start:model_end] = 1.0 / self.model_selection_dim
            
            # Default parameters (equal weights, full density)
            param_start = model_end
            genotype[param_start] = 1.0 / self.model_selection_dim  # weight
            if self.param_dim > 1:
                genotype[param_start + 1] = 1.0  # density
                
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
        
    def decode_genotype(self, genotype: Union[torch.Tensor, np.ndarray]) -> List[LayerGroupGenome]:
        """Decode a flat genotype into structured layer group genomes."""
        if isinstance(genotype, np.ndarray):
            genotype = torch.from_numpy(genotype).float()
            
        layer_groups = []
        
        for layer_idx in range(self.num_layer_groups):
            offset = layer_idx * self.layer_group_dim
            
            # Decode method
            if self.method_dim > 0:
                method_val = genotype[offset]
                method_idx = int(torch.clamp(method_val, 0, len(self.definition.allowed_methods) - 1))
                method_name = self.definition.allowed_methods[method_idx]
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
                model_weights = model_weights / (model_weights.sum() + 1e-8)
                # Keep only top-k models
                top_k = min(self.max_models, np.sum(model_weights > 0.1))
                if top_k < 2:
                    top_k = 2  # Always use at least 2 models
                indices = np.argsort(-model_weights)[:top_k]
                mask = np.zeros_like(model_weights)
                mask[indices] = model_weights[indices]
                model_weights = mask / (mask.sum() + 1e-8)
            
            # Decode parameters
            param_start = model_end
            param_count = METHOD_PARAM_COUNTS[method]
            parameters = genotype[param_start:param_start + param_count].numpy()
            
            # Apply parameter constraints
            parameters = self._constrain_parameters(method, parameters)
            
            layer_groups.append(LayerGroupGenome(
                method=method,
                model_selection=model_weights,
                parameters=parameters
            ))
            
        return layer_groups
        
    def _is_method_compatible(self, method_name: str) -> bool:
        """Check if a method is compatible with the current model configuration."""
        if method_name not in METHOD_NAMES.values():
            return False
            
        method_enum = next(m for m, n in METHOD_NAMES.items() if n == method_name)
        min_models = METHOD_MIN_MODELS.get(method_enum, 2)
        max_models = METHOD_MAX_MODELS.get(method_enum, None)
        
        num_models = len(self.definition.models)
        if num_models < min_models:
            return False
        if max_models is not None and num_models > max_models:
            return False
            
        # Check base model requirement
        requires_base = METHOD_REQUIRES_BASE.get(method_enum, False)
        if requires_base and self.definition.base_model is None:
            return False
            
        return True
        
    def _constrain_parameters(self, method: MergeMethod, params: np.ndarray) -> np.ndarray:
        """Apply method-specific parameter constraints."""
        constrained = params.copy()
        
        if method in [MergeMethod.LINEAR, MergeMethod.TASK_ARITHMETIC, 
                     MergeMethod.DARE_LINEAR, MergeMethod.DELLA_LINEAR]:
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
            constrained[2] = np.clip(constrained[2], 0, 1)  # Gamma (outlier threshold) in [0,1]
            
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
        
    def genotype_to_merge_config(self, genotype: Union[torch.Tensor, np.ndarray]) -> MergeConfiguration:
        """Convert genotype to a MergeConfiguration."""
        layer_groups = self.decode_genotype(genotype)
        
        # Check if all layers use the same method (can use simple config)
        methods = [lg.method for lg in layer_groups]
        if len(set(methods)) == 1 and self.num_layer_groups == 1:
            return self._simple_config(layer_groups[0])
        else:
            return self._complex_config(layer_groups)
            
    def _simple_config(self, layer_group: LayerGroupGenome) -> MergeConfiguration:
        """Create a simple config when all layers use the same method."""
        method_name = METHOD_NAMES[layer_group.method]
        
        # Select models based on weights
        selected_models = []
        for i, weight in enumerate(layer_group.model_selection):
            if weight > 1e-6 and i < self.num_models:
                selected_models.append((self.definition.models[i], weight))
                
        if len(selected_models) < 2:
            # Fallback: use top 2 models
            indices = np.argsort(-layer_group.model_selection)[:2]
            selected_models = [(self.definition.models[i], layer_group.model_selection[i]) 
                             for i in indices if i < self.num_models]
        
        # Build model configs
        models = []
        for model_ref, weight in selected_models:
            model_config = {"model": model_ref}
            
            # Add parameters
            if method_name in ["linear", "task_arithmetic"]:
                model_config["parameters"] = {"weight": float(weight * layer_group.parameters[0])}
            elif method_name in ["ties", "dare_ties"]:
                model_config["parameters"] = {
                    "weight": float(weight * layer_group.parameters[0]),
                    "density": float(layer_group.parameters[1])
                }
            elif method_name == "slerp":
                # SLERP is handled differently - return a slice-based config
                return self._slerp_config(selected_models, layer_group.parameters[0])
                
            models.append(model_config)
            
        config_dict = {
            "merge_method": method_name,
            "models": models,
            "dtype": "bfloat16"
        }
        
        if self.definition.base_model:
            config_dict["base_model"] = self.definition.base_model
        if self.definition.tokenizer_source:
            config_dict["tokenizer_source"] = self.definition.tokenizer_source
            
        return MergeConfiguration.model_validate(config_dict)
        
    def _complex_config(self, layer_groups: List[LayerGroupGenome]) -> MergeConfiguration:
        """Create a complex slice-based config for different methods per layer."""
        # This would implement slice-based configuration
        # For now, fall back to the first layer group's method
        return self._simple_config(layer_groups[0])
        
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

        return MergeConfiguration.model_validate(config_dict)


    def crossover_semantic(self, parent1: torch.Tensor, parent2: torch.Tensor) -> torch.Tensor:
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
                max_models = METHOD_MAX_MODELS.get(chosen_method, None)
                
                if len(self.definition.models) < min_models or (max_models and len(self.definition.models) > max_models):
                    # Fall back to a compatible method
                    compatible_methods = [m for m in self.definition.allowed_methods 
                                        if self._is_method_compatible(m)]
                    if compatible_methods:
                        method_name = np.random.choice(compatible_methods)
                        method_enum = next(m for m, n in METHOD_NAMES.items() if n == method_name)
                        child[offset] = float(list(METHOD_NAMES.keys()).index(method_enum))
            else:
                source_lg = lg1[layer_idx]
                
            # Model selection crossover: blend or swap
            model_start = offset + self.method_dim
            model_end = model_start + self.model_selection_dim
            
            # Check if selected method requires specific model count
            current_method_idx = int(child[offset]) if self.method_dim > 0 else 0
            if current_method_idx < len(self.definition.allowed_methods):
                method_name = self.definition.allowed_methods[current_method_idx]
                method_enum = next(m for m, n in METHOD_NAMES.items() if n == method_name)
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
                        child[model_start:model_end] = (alpha * parent1[model_start:model_end] + 
                                                      (1-alpha) * parent2[model_start:model_end])
                    else:  # Swap segments
                        if np.random.random() < 0.5:
                            child[model_start:model_end] = parent1[model_start:model_end]
                        else:
                            child[model_start:model_end] = parent2[model_start:model_end]
            else:
                # Default blending
                alpha = np.random.random()
                child[model_start:model_end] = (alpha * parent1[model_start:model_end] + 
                                              (1-alpha) * parent2[model_start:model_end])
                    
            # Parameter crossover: method-aware
            param_start = model_end
            param_count = METHOD_PARAM_COUNTS.get(source_lg.method, 1)
            
            # Use arithmetic crossover for parameters of the same method
            alpha = np.random.random()
            child[param_start:param_start + param_count] = (
                alpha * parent1[param_start:param_start + param_count] +
                (1-alpha) * parent2[param_start:param_start + param_count]
            )
                
        return child
        
    def mutate_semantic(self, genotype: torch.Tensor, mutation_rate: float = 0.1, 
                       mutation_sigma: float = 0.05) -> torch.Tensor:
        """Semantic mutation that understands merge structure."""
        mutated = genotype.clone()
        
        for layer_idx in range(self.num_layer_groups):
            offset = layer_idx * self.layer_group_dim
            
            # Method mutation - switch to compatible method
            if self.method_dim > 0 and np.random.random() < mutation_rate:
                compatible_methods = [m for m in self.definition.allowed_methods 
                                    if self._is_method_compatible(m)]
                if compatible_methods:
                    new_method_name = np.random.choice(compatible_methods)
                    new_method_enum = next(m for m, n in METHOD_NAMES.items() if n == new_method_name)
                    new_method_idx = list(METHOD_NAMES.keys()).index(new_method_enum)
                    mutated[offset] = float(new_method_idx)
                
            # Model selection mutation  
            model_start = offset + self.method_dim
            model_end = model_start + self.model_selection_dim
            if np.random.random() < mutation_rate:
                # Get current method to check constraints
                current_method_idx = int(mutated[offset]) if self.method_dim > 0 else 0
                if current_method_idx < len(self.definition.allowed_methods):
                    method_name = self.definition.allowed_methods[current_method_idx]
                    method_enum = next(m for m, n in METHOD_NAMES.items() if n == method_name)
                    max_models = METHOD_MAX_MODELS.get(method_enum, None)
                    min_models = METHOD_MIN_MODELS.get(method_enum, 2)
                    
                    if max_models == 1:
                        # For passthrough, select one model
                        model_weights = torch.zeros(self.model_selection_dim)
                        if self.model_selection_dim > 0:
                            selected_model = np.random.randint(min(self.model_selection_dim, len(self.definition.models)))
                            model_weights[selected_model] = 1.0
                        mutated[model_start:model_end] = model_weights
                    elif max_models == 2:
                        # For methods like SLERP, select exactly 2 models
                        model_weights = torch.zeros(self.model_selection_dim)
                        if self.model_selection_dim >= 2:
                            selected_models = np.random.choice(min(self.model_selection_dim, len(self.definition.models)), 
                                                             size=2, replace=False)
                            for i, model_idx in enumerate(selected_models):
                                model_weights[model_idx] = 0.5 + np.random.normal(0, 0.1)
                            model_weights = torch.abs(model_weights)
                            model_weights /= (model_weights.sum() + 1e-8)
                        mutated[model_start:model_end] = model_weights
                    else:
                        # For methods that can handle multiple models, add noise
                        noise = torch.randn(self.model_selection_dim) * mutation_sigma
                        mutated[model_start:model_end] += noise
                        # Ensure positive and normalized
                        mutated[model_start:model_end] = torch.abs(mutated[model_start:model_end])
                        mutated[model_start:model_end] /= (mutated[model_start:model_end].sum() + 1e-8)
                
            # Parameter mutation
            param_start = model_end  
            current_method_idx = int(mutated[offset]) if self.method_dim > 0 else 0
            if current_method_idx < len(self.definition.allowed_methods):
                method_name = self.definition.allowed_methods[current_method_idx]
                method_enum = next(m for m, n in METHOD_NAMES.items() if n == method_name)
                param_count = METHOD_PARAM_COUNTS.get(method_enum, 1)
                
                for p in range(param_count):
                    if np.random.random() < mutation_rate:
                        mutated[param_start + p] += np.random.normal(0, mutation_sigma)
                        
                # Apply constraints
                if param_count > 0:
                    params = mutated[param_start:param_start + param_count].numpy()
                    constrained = self._constrain_parameters(method_enum, params)
                    mutated[param_start:param_start + param_count] = torch.from_numpy(constrained)
                
        return mutated