# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import gc
import logging
import os
import tempfile
from typing import Any, Dict, Optional

import lm_eval
import lm_eval.api.model
import lm_eval.models.huggingface
import lm_eval.tasks
import ray
import ray.util.queue
import ray.util.scheduling_strategies
import torch
import transformers
from transformers.utils import is_flash_attn_2_available

from mergekit.architecture.base import ConfiguredModelArchitecture

try:
    import vllm
except ImportError:
    vllm = None


from mergekit.architecture import arch_info_for_config
from mergekit.common import (
    call_with_dtype,
    get_torch_accelerator_module,
    get_torch_accelerator_type,
    set_config_dtype_field,
)
from mergekit.config import MergeConfiguration
from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.genome import InvalidGenotypeError, ModelGenome
from mergekit.evo.helpers import (
    _eval_model,
    evaluate_model,
    evaluate_model_cpu,
    merge_model_with_details,
)
from mergekit.evo.monkeypatch import (
    NoInit,
    monkeypatch_lmeval_shuffle,
    monkeypatch_lmeval_vllm,
)
from mergekit.evo.ray_observability import RayRunObserver
from mergekit.graph import Executor
from mergekit.io.tasks import LoaderCache, ReturnTensor
from mergekit.merge import _model_out_config
from mergekit.options import MergeOptions
from mergekit.plan import MergePlanner

LOG = logging.getLogger(__name__)


def _get_lm_eval_vllm_class():
    vllm_models = getattr(getattr(lm_eval, "models", None), "vllm_causallms", None)
    return getattr(vllm_models, "VLLM", None)


def _accelerated_eval_model_kwargs(
    *,
    vllm: bool,
    quantization_config: Optional[transformers.BitsAndBytesConfig] = None,
) -> Optional[dict]:
    model_kwargs = {}
    if not vllm:
        model_kwargs.update(
            {
                "device": "cuda",
                "dtype": "bfloat16",
                "device_map": None,
                "low_cpu_mem_usage": True,
            }
        )
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
    return model_kwargs or None


def _evaluate_merged_path_accelerated(
    merged_path: str,
    config: EvolMergeConfiguration,
    *,
    vllm: bool,
    tensor_parallel_size: int,
    batch_size: Optional[int],
    task_manager: Optional[lm_eval.tasks.TaskManager],
    quantization_config: Optional[transformers.BitsAndBytesConfig],
) -> dict:
    return evaluate_model(
        merged_path,
        config.tasks,
        num_fewshot=config.num_fewshot,
        limit=config.limit,
        vllm=vllm,
        tensor_parallel_size=tensor_parallel_size,
        batch_size=batch_size,
        task_manager=task_manager,
        apply_chat_template=config.apply_chat_template,
        fewshot_as_multiturn=config.fewshot_as_multiturn,
        fitness_mode=getattr(config, "fitness_mode", "weighted_sum"),
        fitness_version=getattr(getattr(config, "fitness", None), "version", "v1"),
        lower_is_better_transform=getattr(
            getattr(config, "fitness", None),
            "lower_is_better_transform",
            "legacy_reciprocal",
        ),
        task_mix_profile=getattr(config, "task_mix_profile", None),
        behavior_prompts=getattr(config, "behavior_prompts", None),
        behavior_probe_max_new_tokens=getattr(
            config, "behavior_probe_max_new_tokens", 24
        ),
        behavior_repetition_ngram_size=getattr(
            config, "behavior_repetition_ngram_size", 4
        ),
        behavior_min_distinct_ratio=getattr(config, "behavior_min_distinct_ratio", 0.2),
        behavior_reject_on_degenerate=getattr(
            config, "behavior_reject_on_degenerate", False
        ),
        model_kwargs=_accelerated_eval_model_kwargs(
            vllm=vllm,
            quantization_config=quantization_config,
        ),
    )


class MergeActorBase:
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        model_storage_path: Optional[str] = None,
        vllm: bool = False,
        tensor_parallel_size: int = 1,
        batch_size: Optional[int] = None,
        task_manager: Optional[lm_eval.tasks.TaskManager] = None,
        quantization_config: Optional[transformers.BitsAndBytesConfig] = None,
        worker_name: Optional[str] = None,
        observer_config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.cache = LoaderCache()
        self.cache.setup(merge_options)
        self.model_storage_path = model_storage_path
        self.vllm = vllm
        self.tensor_parallel_size = tensor_parallel_size
        self.batch_size = batch_size
        self.task_manager = task_manager
        self.quantization_config = quantization_config
        self.worker_name = worker_name or f"mergekit-worker-{os.getpid()}"
        self._observer = RayRunObserver.from_config(
            observer_config,
            actor_name=self.worker_name,
            role="worker",
        )
        self._status: Dict[str, Any] = {
            "worker_name": self.worker_name,
            "active": False,
            "stage": "idle",
            "generation": None,
            "candidate_index": None,
            "merge_method": None,
            "evaluation_stage": None,
            "last_score": None,
            "last_error_type": None,
        }

        if config.shuffle:
            monkeypatch_lmeval_shuffle()

        # monkeypatch_tqdm()
        monkeypatch_lmeval_vllm()
        self._set_status(stage="idle", active=False)

    def _set_status(
        self,
        *,
        stage: str,
        active: bool,
        context: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None,
        error_type: Optional[str] = None,
    ) -> None:
        ctx = dict(context or {})
        self._status.update(
            {
                "active": bool(active),
                "stage": stage,
                "generation": ctx.get("generation"),
                "candidate_index": ctx.get("candidate_index"),
                "merge_method": ctx.get("merge_method"),
                "evaluation_stage": ctx.get("evaluation_stage"),
                "last_score": score,
                "last_error_type": error_type,
            }
        )
        if self._observer is not None:
            self._observer.record_actor_status(
                stage=str(ctx.get("evaluation_stage") or stage),
                active=active,
                generation=ctx.get("generation"),
                candidate_index=ctx.get("candidate_index"),
                score=score,
                error_type=error_type,
            )

    def get_status(self) -> Dict[str, Any]:
        return dict(self._status)

    def register_reentrant_parent(self, checkpoint_path: str) -> int:
        from mergekit.evo.reentry import register_reentrant_parent

        return register_reentrant_parent(self.genome, checkpoint_path)


@ray.remote(num_cpus=1, num_gpus=1.0)
class OnDiskMergeEvaluator(MergeActorBase):
    """
    Merges models to disk then evaluates them in a separate process.

    Maximum compatibility and potential for parallelism, but higher overhead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_genotype(
        self,
        genotype: torch.Tensor,
        eval_config: Optional[EvolMergeConfiguration] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> dict:
        self._set_status(stage="merge", active=True, context=context)
        gc.collect()
        try:
            torch_accelerator_module = get_torch_accelerator_module(
                self.merge_options.device
            )
            # empty_cache is not available for CPU; guard call
            empty_cache = getattr(torch_accelerator_module, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
        except Exception:
            pass
        LOG.info("[%s] Merging model", self.worker_name)
        merge_info = merge_model_with_details(
            genotype, self.genome, self.model_storage_path, self.merge_options
        )
        merged_path = merge_info.get("merged_path")
        if not merged_path:
            LOG.error(
                "[%s] Model merge failed: %s",
                self.worker_name,
                merge_info.get("error_message", "Unknown merge failure"),
            )
            self._set_status(
                stage="merge",
                active=False,
                context=context,
                error_type=merge_info.get("error_type", "merge_failed"),
            )
            return {
                "score": None,
                "results": None,
                "error_stage": merge_info.get("error_stage", "merge"),
                "error_type": merge_info.get("error_type", "merge_failed"),
                "error_message": merge_info.get(
                    "error_message",
                    "Model merge failed",
                ),
            }

        LOG.info("[%s] Model merged to %s", self.worker_name, merged_path)
        self._set_status(stage="evaluate", active=True, context=context)
        result = _evaluate_merged_path_accelerated(
            merged_path,
            eval_config or self.config,
            vllm=self.vllm,
            tensor_parallel_size=self.tensor_parallel_size,
            batch_size=self.batch_size,
            task_manager=self.task_manager,
            quantization_config=self.quantization_config,
        )
        self._set_status(
            stage="idle",
            active=False,
            context=context,
            score=result.get("score"),
            error_type=result.get("error_type"),
        )
        return result


@ray.remote(num_cpus=1)
class OnDiskMergeEvaluatorCPU(MergeActorBase):
    """
    CPU-only variant: merges to disk and evaluates with HF backend on CPU.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_genotype(
        self,
        genotype: torch.Tensor,
        eval_config: Optional[EvolMergeConfiguration] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> dict:
        self._set_status(stage="merge", active=True, context=context)
        gc.collect()
        os.environ.setdefault("TRANSFORMERS_NO_CUDA", "1")
        LOG.info("[%s] Merging model (CPU)", self.worker_name)
        merge_info = merge_model_with_details(
            genotype, self.genome, self.model_storage_path, self.merge_options
        )
        merged_path = merge_info.get("merged_path")
        if not merged_path:
            LOG.error(
                "[%s] Model merge failed: %s",
                self.worker_name,
                merge_info.get("error_message", "Unknown merge failure"),
            )
            self._set_status(
                stage="merge",
                active=False,
                context=context,
                error_type=merge_info.get("error_type", "merge_failed"),
            )
            return {
                "score": None,
                "results": None,
                "error_stage": merge_info.get("error_stage", "merge"),
                "error_type": merge_info.get("error_type", "merge_failed"),
                "error_message": merge_info.get(
                    "error_message",
                    "Model merge failed",
                ),
            }

        model_kwargs = {
            "device": "cpu",
            "dtype": "float32",
            "device_map": None,
            "low_cpu_mem_usage": False,
        }
        if self.quantization_config is not None:
            model_kwargs["quantization_config"] = self.quantization_config
        LOG.info("[%s] Model merged to %s", self.worker_name, merged_path)
        config = eval_config or self.config
        self._set_status(stage="evaluate", active=True, context=context)
        result = evaluate_model_cpu(
            merged_path,
            config.tasks,
            num_fewshot=config.num_fewshot,
            limit=config.limit,
            batch_size=self.batch_size,
            task_manager=self.task_manager,
            apply_chat_template=config.apply_chat_template,
            fewshot_as_multiturn=config.fewshot_as_multiturn,
            fitness_mode=getattr(config, "fitness_mode", "weighted_sum"),
            fitness_version=getattr(getattr(config, "fitness", None), "version", "v1"),
            lower_is_better_transform=getattr(
                getattr(config, "fitness", None),
                "lower_is_better_transform",
                "legacy_reciprocal",
            ),
            task_mix_profile=getattr(config, "task_mix_profile", None),
            behavior_prompts=getattr(config, "behavior_prompts", None),
            behavior_probe_max_new_tokens=getattr(
                config, "behavior_probe_max_new_tokens", 24
            ),
            behavior_repetition_ngram_size=getattr(
                config, "behavior_repetition_ngram_size", 4
            ),
            behavior_min_distinct_ratio=getattr(
                config, "behavior_min_distinct_ratio", 0.2
            ),
            behavior_reject_on_degenerate=getattr(
                config, "behavior_reject_on_degenerate", False
            ),
            model_kwargs=model_kwargs,
        )
        self._set_status(
            stage="idle",
            active=False,
            context=context,
            score=result.get("score"),
            error_type=result.get("error_type"),
        )
        return result


@ray.remote(num_cpus=1, num_gpus=1)
class InMemoryMergeEvaluator(MergeActorBase):
    """
    Performs merges in memory, using a single model instance.

    This reduces overhead from disk I/O and model loading, but prevents
    parallelism and may be slower for large models.

    Implementation is dark sorcery tampering with the internals of lm-eval,
    transformers, and vLLM and may break at any time.
    """

    model: Optional[Any] = None
    arch_info: Optional[ConfiguredModelArchitecture] = None

    def __init__(
        self,
        *args,
        vllm: bool = False,
        **kwargs,
    ):
        # assert not vllm, "VLLM is not supported for in-memory merging"
        super().__init__(*args, vllm=vllm, **kwargs)

    def _maybe_init_model(self, config: MergeConfiguration):
        ai = arch_info_for_config(self.genome._input_config_example)
        cfg_out = _model_out_config(
            config,
            ai,
            trust_remote_code=self.merge_options.trust_remote_code,
        )
        cfg_out.use_cache = True
        set_config_dtype_field(cfg_out, torch.bfloat16)

        if self.arch_info is not None:
            different = False
            for key in cfg_out.to_diff_dict():
                if key in ["architectures", "model_type"]:
                    # to get to here we must have --allow-crimes set, so let it ride
                    continue
                elif key in ["use_cache", "dtype", "torch_dtype"]:
                    continue
                elif key.endswith("_token_id"):
                    # update our config but don't fail if it's different
                    setattr(self.arch_info.config, key, getattr(cfg_out, key, None))
                    continue

                if getattr(cfg_out, key) != getattr(self.arch_info.config, key, None):
                    LOG.warning(f"Config key {key} changed, reinitializing model")
                    different = True
                    break

            if not different:
                return

        self.inner_model = None

        model_kwargs = {
            "trust_remote_code": self.merge_options.trust_remote_code,
        }
        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        with NoInit():
            inner_model = call_with_dtype(
                transformers.AutoModelForCausalLM.from_config,
                cfg_out,
                dtype=torch.bfloat16,
                **model_kwargs,
            )
            inner_model = (
                inner_model.bfloat16()
                .to(self.merge_options.device)
                .eval()
                .requires_grad_(False)
            )

        if self.vllm:
            if self.tensor_parallel_size != 1:
                raise RuntimeError(
                    "In-memory vLLM evaluation only supports tensor_parallel_size=1"
                )
            vllm_model_cls = _get_lm_eval_vllm_class()
            if vllm_model_cls is None:
                raise RuntimeError(
                    "lm_eval vLLM backend is unavailable in this environment"
                )
            # oh i hate this
            with tempfile.TemporaryDirectory(
                dir=self.model_storage_path, prefix="vllm"
            ) as tempdir:
                inner_model.save_pretrained(
                    tempdir, safe_serialization=True, out_shard_size=1_000_000_000_000
                )
                del inner_model
                tokenizer_donor = self.genome.definition.base_model
                if tokenizer_donor is None:
                    LOG.warning(
                        "Base model not set, using tokenizer from first model in genome"
                    )
                    tokenizer_donor = self.genome.definition.models[0]
                tok = transformers.AutoTokenizer.from_pretrained(
                    tokenizer_donor.model.path, use_fast=True
                )
                tok.save_pretrained(tempdir)

                max_model_len = None
                if (
                    seq_len := getattr(cfg_out, "max_position_embeddings", None)
                ) is not None:
                    max_model_len = seq_len
                if (window_sz := getattr(cfg_out, "sliding_window", None)) is not None:
                    max_model_len = min(max_model_len or 1024, window_sz)
                if max_model_len and max_model_len > 8192:
                    max_model_len = 8192
                    LOG.warning(f"Clipping sequence length to {max_model_len}")

                accelerator_type = get_torch_accelerator_type(self.merge_options.device)
                mem_util = (
                    0.7 if accelerator_type in ["cuda", "xpu"] else 0.9
                )  # reduce memory usage if we're also using accelerator for the merge
                self.model = vllm_model_cls(
                    pretrained=tempdir,
                    batch_size=self.batch_size or "auto",
                    max_model_len=max_model_len,
                    gpu_memory_utilization=mem_util,
                    tensor_parallel_size=self.tensor_parallel_size,
                    dtype="bfloat16",
                    device=self.merge_options.device,
                    trust_remote_code=self.merge_options.trust_remote_code,
                )
        else:
            self.model = lm_eval.models.huggingface.HFLM(pretrained=inner_model)
        self.arch_info = (
            ConfiguredModelArchitecture(
                info=ai,
                config=cfg_out,
            )
            if ai
            else None
        )
        LOG.info("Model initialized")

    def evaluate(
        self,
        genotype: torch.Tensor,
        eval_config: Optional[EvolMergeConfiguration] = None,
    ) -> dict:
        try:
            if hasattr(self.genome, "genotype_to_merge_plan"):
                plan = self.genome.genotype_to_merge_plan(genotype)
                if plan["kind"] != "config":
                    raise InvalidGenotypeError(
                        "In-memory evaluation does not support layered mixed-method plans"
                    )
                config = plan["config"]
            elif hasattr(self.genome, "genotype_to_merge_config"):
                config = self.genome.genotype_to_merge_config(genotype)
            else:
                config = self.genome.genotype_merge_config(genotype)
        except InvalidGenotypeError as e:
            LOG.error("Invalid genotype", exc_info=e)
            return {
                "score": None,
                "results": None,
                "error_stage": "merge",
                "error_type": "invalid_genotype",
                "error_message": str(e),
            }

        self._maybe_init_model(config)

        planner = MergePlanner(
            config,
            self.arch_info.info,
            self.merge_options,
            self.arch_info.config,
        )

        tasks = planner.plan_in_memory()

        model = self.model.model
        if vllm is not None and isinstance(model, vllm.LLM):
            assert (
                model.llm_engine.parallel_config.world_size == 1
            ), "Must be single GPU"
            engine = model.llm_engine
            if hasattr(engine, "model_executor"):
                worker = engine.model_executor.worker
            elif hasattr(engine, "driver_worker"):
                worker = engine.driver_worker
            else:
                raise ValueError("Unknown LLM engine type")
            model = worker.model_runner.model
        param_dict = dict(model.named_parameters())

        stacked_mapping = {
            # mappings for Llama/Mistral attention weights to vLLM packed tensors
            ".q_proj.": (".qkv_proj.", "q"),
            ".k_proj.": (".qkv_proj.", "k"),
            ".v_proj.": (".qkv_proj.", "v"),
            ".gate_proj.": (".gate_up_proj.", 0),
            ".up_proj.": (".gate_up_proj.", 1),
        }

        accelerator_type = get_torch_accelerator_type(self.merge_options.device)
        executor = Executor(
            tasks,
            math_device=(
                self.merge_options.device
                if accelerator_type in ["cuda", "xpu"]
                else "cpu"
            ),
            storage_device=(
                self.merge_options.device
                if accelerator_type in ["cuda", "xpu"]
                else "cpu"
            ),
        )
        for tensor_task, value in executor.run(quiet=True):
            assert isinstance(tensor_task, ReturnTensor)
            name = tensor_task.weight_info.name

            if name in param_dict:
                param_dict[name].data.copy_(value, non_blocking=True)
            elif self.vllm:
                stacked = False
                for needle, (replacement, shard_id) in stacked_mapping.items():
                    if needle in name:
                        target = name.replace(needle, replacement)
                        param = param_dict[target]
                        weight_loader = param.weight_loader
                        weight_loader(param, value, shard_id)
                        stacked = True
                        break

                if not stacked:
                    raise ValueError(f"Unknown parameter {name}")
            else:
                raise ValueError(f"Unknown parameter {name}")

            del value

        config = eval_config or self.config
        return _eval_model(
            self.model,
            config.tasks,
            num_fewshot=config.num_fewshot,
            limit=config.limit,
            task_manager=self.task_manager,
            batch_size=self.batch_size,
            apply_chat_template=config.apply_chat_template,
            fewshot_as_multiturn=config.fewshot_as_multiturn,
        )

    def evaluate_genotype(
        self,
        genotype: torch.Tensor,
        eval_config: Optional[EvolMergeConfiguration] = None,
    ) -> dict:
        return self.evaluate(genotype, eval_config=eval_config)
