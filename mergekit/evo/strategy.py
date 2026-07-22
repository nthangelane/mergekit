# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import transformers

from mergekit.common import get_torch_accelerator_count
from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.genome import ModelGenome
from mergekit.evo.helpers import (
    evaluate_model,
    evaluate_model_cpu,
    merge_model,
    merge_model_with_details,
)
from mergekit.evo.ranking import weighted_rank_scores
from mergekit.evo.ray_observability import (
    RayRunObserver,
    sanitize_observability_label,
    worker_actor_name,
)
from mergekit.evo.task_utils import create_task_manager
from mergekit.options import MergeOptions


def _require_ray():
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Ray is required for pool, buffered, and GPU-serial evaluation. "
            "Install the distributed dependencies or use --strategy serial "
            "with --device cpu."
        ) from exc
    return ray


def _gpus_per_evaluation(
    *,
    total_gpus: Optional[int],
    vllm: bool,
    tensor_parallel_size: int,
    in_memory: bool = False,
) -> int:
    if total_gpus is None or total_gpus <= 0:
        return 0
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be at least 1")
    if in_memory and tensor_parallel_size > 1:
        raise ValueError(
            "In-memory evaluation does not support tensor_parallel_size > 1"
        )
    if tensor_parallel_size > 1 and not vllm:
        raise ValueError("tensor_parallel_size > 1 requires the vLLM backend")
    if tensor_parallel_size > total_gpus:
        raise ValueError(
            f"tensor_parallel_size={tensor_parallel_size} exceeds num_gpus={total_gpus}"
        )
    return tensor_parallel_size if vllm else 1


def _gpu_worker_capacity(
    *,
    total_gpus: Optional[int],
    vllm: bool,
    tensor_parallel_size: int,
    in_memory: bool = False,
) -> int:
    gpus_per_eval = _gpus_per_evaluation(
        total_gpus=total_gpus,
        vllm=vllm,
        tensor_parallel_size=tensor_parallel_size,
        in_memory=in_memory,
    )
    if gpus_per_eval <= 0:
        return 0
    return max(1, int(total_gpus or 0) // gpus_per_eval)


class EvaluationStrategyBase(ABC):
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        num_gpus: Optional[int] = None,
        num_workers: Optional[int] = None,
        tensor_parallel_size: int = 1,
        batch_size: Optional[int] = None,
        task_search_path: Union[str, List[str], None] = None,
        model_storage_path: Optional[str] = None,
        quantization_config: Optional[transformers.BitsAndBytesConfig] = None,
        run_label: Optional[str] = None,
        run_observer: Optional[RayRunObserver] = None,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.num_gpus = (
            num_gpus
            if num_gpus is not None
            else get_torch_accelerator_count(self.merge_options.device)
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tensor_parallel_size = tensor_parallel_size
        required_task_names: List[str] = []
        for task in self.config.tasks:
            if task.name not in required_task_names:
                required_task_names.append(task.name)
        for task in getattr(self.config, "stage1_tasks", None) or []:
            if task.name not in required_task_names:
                required_task_names.append(task.name)
        self.task_manager = create_task_manager(
            task_search_path,
            required_tasks=required_task_names,
        )
        self.model_storage_path = model_storage_path
        self.quantization_config = quantization_config
        self.run_label = sanitize_observability_label(run_label or "mergekit")
        self.run_observer = run_observer
        self.current_generation: Optional[int] = None
        self.current_phase: str = "init"
        if self.model_storage_path:
            os.makedirs(self.model_storage_path, exist_ok=True)

    def register_reentrant_parent(self, checkpoint_path: str) -> int:
        """Register a repaired parent locally and in persistent evaluators."""
        from mergekit.evo.reentry import (
            align_reentrant_parent_vocab,
            register_reentrant_parent,
        )

        align_reentrant_parent_vocab(
            self.genome,
            checkpoint_path,
            trust_remote_code=self.merge_options.trust_remote_code,
        )
        parent_index = register_reentrant_parent(self.genome, checkpoint_path)
        self._propagate_reentrant_parent(checkpoint_path)
        return parent_index

    def _propagate_reentrant_parent(self, checkpoint_path: str) -> None:
        """Update evaluator-owned genome copies, if this strategy has any."""
        del checkpoint_path

    def set_runtime_context(
        self,
        *,
        generation: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> None:
        if generation is not None:
            self.current_generation = int(generation)
        if phase is not None:
            self.current_phase = str(phase)

    def _stage_label_for_config(self, eval_config: EvolMergeConfiguration) -> str:
        if getattr(self.config, "two_stage", False):
            stage1_names = [
                task.name for task in getattr(self.config, "stage1_tasks", [])
            ]
            stage2_names = [task.name for task in self.config.tasks]
            eval_names = [task.name for task in eval_config.tasks]
            if eval_names == stage1_names and eval_config.limit == (
                self.config.stage1_limit
                if self.config.stage1_limit is not None
                else self.config.limit
            ):
                return "stage1"
            if eval_names == stage2_names and eval_config.limit == (
                self.config.stage2_limit
                if self.config.stage2_limit is not None
                else self.config.limit
            ):
                return "stage2"
        return "full"

    def _method_for_genotype(self, genotype: np.ndarray) -> str:
        try:
            if hasattr(self.genome, "method_label_for_genotype"):
                return str(self.genome.method_label_for_genotype(genotype))
            cfg = (
                self.genome.genotype_to_merge_config(genotype)
                if hasattr(self.genome, "genotype_to_merge_config")
                else self.genome.genotype_merge_config(genotype)
            )
            return str(getattr(cfg, "merge_method", None) or "unknown")
        except Exception:
            return "unknown"

    def _candidate_contexts(
        self,
        genotypes: List[np.ndarray],
        eval_config: EvolMergeConfiguration,
    ) -> List[Dict[str, Any]]:
        stage_label = self._stage_label_for_config(eval_config)
        current_generation = getattr(self, "current_generation", None)
        current_phase = str(getattr(self, "current_phase", "ga"))
        contexts: List[Dict[str, Any]] = []
        for idx, genotype in enumerate(genotypes):
            contexts.append(
                {
                    "generation": current_generation,
                    "candidate_index": idx,
                    "evaluation_stage": stage_label,
                    "phase": current_phase,
                    "merge_method": self._method_for_genotype(genotype),
                }
            )
        return contexts

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[dict]:
        if not genotypes:
            return []
        if not getattr(self.config, "two_stage", False) or len(genotypes) == 1:
            return self._evaluate_genotypes_once(genotypes, self.config)

        stage1_config = self._stage_config(stage=1)
        stage2_config = self._stage_config(stage=2)
        stage1_results = self._evaluate_genotypes_once(genotypes, stage1_config)

        if getattr(self.config, "metric_guard_mode", "reject") == "quarantine":
            stage1_results = self._resolve_quarantined_candidates(
                genotypes, stage1_results, stage1_config
            )

        combined_results: List[dict] = []
        successful_indices: List[int] = []
        for idx, stage1_result in enumerate(stage1_results):
            result = dict(stage1_result)
            result["stage1_score"] = stage1_result.get("score")
            result["stage1_results"] = stage1_result.get("results")
            result["stage1_limit"] = stage1_config.limit
            result["stage1_tasks"] = [task.name for task in stage1_config.tasks]
            result["evaluation_stage"] = "stage1"
            result.setdefault("quarantined", False)
            result.setdefault("quarantine_outcome", None)
            if stage1_result.get("score") is not None:
                successful_indices.append(idx)
            combined_results.append(result)

        if not successful_indices:
            return combined_results

        stage2_top_k = min(
            self._stage2_top_k(len(genotypes), len(successful_indices)),
            len(successful_indices),
        )
        ranked_indices = self._rank_stage_results(stage1_results, successful_indices)
        shortlisted_indices = ranked_indices[:stage2_top_k]
        shortlisted_genotypes = [genotypes[idx] for idx in shortlisted_indices]
        stage2_results = self._evaluate_genotypes_once(
            shortlisted_genotypes, stage2_config
        )

        for idx in successful_indices:
            if idx not in shortlisted_indices:
                combined_results[idx]["stage2_skipped"] = True
                combined_results[idx]["score_source"] = "stage1"

        for idx, stage2_result in zip(shortlisted_indices, stage2_results):
            result = combined_results[idx]
            result["stage2_score"] = stage2_result.get("score")
            result["stage2_results"] = stage2_result.get("results")
            result["stage2_limit"] = stage2_config.limit
            result["stage2_tasks"] = [task.name for task in stage2_config.tasks]
            result["stage2_skipped"] = False
            result["evaluation_stage"] = "stage2"
            result["score_source"] = "stage2"
            if stage2_result.get("score") is None:
                result["score"] = None
                result["results"] = stage2_result.get("results")
                result["error_stage"] = stage2_result.get("error_stage")
                result["error_type"] = stage2_result.get("error_type")
                result["error_message"] = stage2_result.get("error_message")
            else:
                result["score"] = stage2_result.get("score")
                result["results"] = stage2_result.get("results")
            repair_metadata = stage2_result.get("repair")
            if repair_metadata is not None:
                result["repair"] = repair_metadata
                result["repair_pre_score"] = stage2_result.get("repair_pre_score")
                result["repair_post_score"] = stage2_result.get("repair_post_score")
                result["repair_comparison_limit"] = stage2_result.get(
                    "repair_comparison_limit"
                )
                result["repair_comparison_audited"] = stage2_result.get(
                    "repair_comparison_audited", False
                )
                if stage2_result.get("reentry_checkpoint_path"):
                    result["reentry_checkpoint_path"] = stage2_result[
                        "reentry_checkpoint_path"
                    ]

        return combined_results

    def _resolve_quarantined_candidates(
        self,
        genotypes: List[np.ndarray],
        stage1_results: List[dict],
        stage1_config: EvolMergeConfiguration,
    ) -> List[dict]:
        resolved = [dict(result) for result in stage1_results]
        for idx, (genotype, result) in enumerate(zip(genotypes, stage1_results)):
            if result.get("error_type") != "metric_guard":
                resolved[idx].setdefault("quarantined", False)
                resolved[idx].setdefault("quarantine_outcome", None)
                continue

            guarded_task_name = result.get("guarded_task")
            guarded_tasks = [
                task
                for task in stage1_config.tasks
                if guarded_task_name is None or task.name == guarded_task_name
            ]
            if not guarded_tasks:
                guarded_tasks = list(stage1_config.tasks)
            quarantine_config = stage1_config.model_copy(
                update={
                    "tasks": guarded_tasks,
                    "limit": int(self.config.quarantine_audit_limit),
                    "two_stage": False,
                    "metric_guard_mode": "reject",
                }
            )
            self.set_runtime_context(phase="quarantine")
            audit_result = self._evaluate_genotypes_once([genotype], quarantine_config)[
                0
            ]
            self.set_runtime_context(phase="ga")
            if audit_result.get("score") is not None:
                admitted = dict(audit_result)
                admitted.update(
                    {
                        "quarantined": True,
                        "quarantine_outcome": "passed",
                        "quarantine_stage1_score": result.get("score"),
                        "quarantine_limit": int(self.config.quarantine_audit_limit),
                        "score_source": "quarantine_stage2",
                    }
                )
                resolved[idx] = admitted
            else:
                confirmed = dict(result)
                confirmed.update(
                    {
                        "score": None,
                        "quarantined": True,
                        "quarantine_outcome": "failed",
                        "quarantine_limit": int(self.config.quarantine_audit_limit),
                        "error_type": "metric_guard_confirmed",
                        "error_message": audit_result.get("error_message")
                        or result.get("error_message"),
                    }
                )
                resolved[idx] = confirmed
        return resolved

    def audit_genotypes(
        self, genotypes: List[np.ndarray], *, limit: Optional[int]
    ) -> List[dict]:
        """Evaluate genotypes once at audit fidelity without repair or staging."""
        repair_config = getattr(self.config, "repair", None)
        audit_config = self.config.model_copy(
            update={
                "tasks": self.config.tasks,
                "limit": limit,
                "two_stage": False,
                "repair": (
                    repair_config.model_copy(update={"enabled": False})
                    if repair_config is not None
                    else None
                ),
                "metric_guard_mode": "reject",
            }
        )
        self.set_runtime_context(phase="audit")
        try:
            return self._evaluate_genotypes_once(genotypes, audit_config)
        finally:
            self.set_runtime_context(phase="ga")

    def _rank_stage_results(
        self, stage_results: List[dict], candidate_indices: List[int]
    ) -> List[int]:
        if getattr(self.config, "fitness_mode", "weighted_sum") == "weighted_rank":
            objective_weights = None
            if getattr(self.config, "ga", None) is not None:
                objective_weights = getattr(
                    self.config.ga, "rank_objective_weights", None
                )
            rank_scores, _details = weighted_rank_scores(
                stage_results,
                objective_weights=objective_weights,
            )
            return sorted(
                candidate_indices,
                key=lambda idx: float(rank_scores[idx]),
                reverse=True,
            )
        return sorted(
            candidate_indices,
            key=lambda idx: float(stage_results[idx]["score"]),
            reverse=True,
        )

    def evaluate_genotype(self, genotype: np.ndarray) -> dict:
        return self.evaluate_genotypes([genotype])[0]

    def _stage_config(self, stage: int) -> EvolMergeConfiguration:
        if stage == 1:
            repair_config = getattr(self.config, "repair", None)
            return self.config.model_copy(
                update={
                    "tasks": getattr(self.config, "stage1_tasks", None)
                    or self.config.tasks,
                    "limit": (
                        getattr(self.config, "stage1_limit", None)
                        if getattr(self.config, "stage1_limit", None) is not None
                        else self.config.limit
                    ),
                    "two_stage": False,
                    "repair": (
                        repair_config.model_copy(update={"enabled": False})
                        if repair_config is not None
                        else None
                    ),
                }
            )
        if stage == 2:
            return self.config.model_copy(
                update={
                    "limit": (
                        getattr(self.config, "stage2_limit", None)
                        if getattr(self.config, "stage2_limit", None) is not None
                        else self.config.limit
                    ),
                    "two_stage": False,
                }
            )
        raise ValueError(f"Unknown stage {stage}")

    def _stage2_top_k(self, population_size: int, successful_count: int) -> int:
        if getattr(self.config, "stage2_top_k", None) is not None:
            return int(self.config.stage2_top_k)
        return max(1, int(math.ceil(successful_count / 2.0)))

    @abstractmethod
    def _evaluate_genotypes_once(
        self, genotypes: List[np.ndarray], eval_config: EvolMergeConfiguration
    ) -> List[dict]:
        pass


class ActorPoolEvaluationStrategy(EvaluationStrategyBase):
    """
    Uses a fixed-size pool of actors to evaluate genotypes in parallel.
    """

    def __init__(
        self,
        *args,
        in_memory: bool = False,
        vllm: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        ray = _require_ray()
        from mergekit.evo.actors import (
            InMemoryMergeEvaluator,
            OnDiskMergeEvaluator,
            OnDiskMergeEvaluatorCPU,
        )

        if in_memory:
            if self.num_gpus and self.num_gpus > 0:
                self.actor_cls = InMemoryMergeEvaluator
            else:
                raise ValueError("In-memory evaluation is not supported on CPU")
        else:
            self.actor_cls = (
                OnDiskMergeEvaluator
                if (self.num_gpus and self.num_gpus > 0)
                else OnDiskMergeEvaluatorCPU
            )

        worker_count = (
            _gpu_worker_capacity(
                total_gpus=self.num_gpus,
                vllm=vllm,
                tensor_parallel_size=self.tensor_parallel_size,
                in_memory=in_memory,
            )
            if (self.num_gpus and self.num_gpus > 0)
            else (self.num_workers or 1)
        )
        actor_gpu_request = _gpus_per_evaluation(
            total_gpus=self.num_gpus,
            vllm=vllm,
            tensor_parallel_size=self.tensor_parallel_size,
            in_memory=in_memory,
        )
        self._actors = [
            (
                self.actor_cls.options(
                    name=worker_actor_name(self.run_label, "pool", worker_idx),
                    num_gpus=actor_gpu_request,
                ).remote
                if actor_gpu_request > 0
                else self.actor_cls.options(
                    name=worker_actor_name(self.run_label, "pool", worker_idx)
                ).remote
            )(
                self.config,
                self.genome,
                self.merge_options,
                model_storage_path=self.model_storage_path,
                vllm=vllm,
                tensor_parallel_size=self.tensor_parallel_size,
                batch_size=self.batch_size,
                task_manager=self.task_manager,
                quantization_config=self.quantization_config,
                worker_name=worker_actor_name(self.run_label, "pool", worker_idx),
                observer_config=(
                    self.run_observer.export_config(role="worker")
                    if self.run_observer is not None
                    else None
                ),
            )
            for worker_idx in range(worker_count)
        ]
        self.actor_pool = ray.util.ActorPool(self._actors)

    def _propagate_reentrant_parent(self, checkpoint_path: str) -> None:
        ray = _require_ray()
        ray.get(
            [
                actor.register_reentrant_parent.remote(checkpoint_path)
                for actor in self._actors
            ]
        )

    def _evaluate_genotypes_once(
        self, genotypes: List[np.ndarray], eval_config: EvolMergeConfiguration
    ) -> List[dict]:
        candidate_contexts = self._candidate_contexts(genotypes, eval_config)
        return list(
            self.actor_pool.map(
                lambda a, item: a.evaluate_genotype.remote(
                    item[0], eval_config, item[1]
                ),
                list(zip(genotypes, candidate_contexts)),
            )
        )


class BufferedRayEvaluationStrategyActor:
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        vllm: bool = False,
        num_gpus: Optional[int] = None,
        num_workers: Optional[int] = None,
        tensor_parallel_size: int = 1,
        batch_size: Optional[int] = None,
        task_manager: Optional[Any] = None,
        model_storage_path: Optional[str] = None,
        quantization_config: Optional[transformers.BitsAndBytesConfig] = None,
        worker_name: Optional[str] = None,
        observer_config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.vllm = vllm
        self.num_gpus = (
            num_gpus
            if num_gpus is not None
            else get_torch_accelerator_count(self.merge_options.device)
        )
        self.num_workers = num_workers or 1
        self.tensor_parallel_size = tensor_parallel_size
        self.eval_gpus_per_task = _gpus_per_evaluation(
            total_gpus=self.num_gpus,
            vllm=vllm,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.input_queue = []
        self.batch_size = batch_size
        self.task_manager = task_manager
        self.model_storage_path = model_storage_path
        self.quantization_config = quantization_config
        self._shutdown = False
        self.worker_name = worker_name or "mergekit-buffered-worker"
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
            "queued": 0,
        }

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
                "queued": len(self.input_queue),
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
        status = dict(self._status)
        status["queued"] = len(self.input_queue)
        return status

    async def evaluate_genotype(
        self,
        genotype: np.ndarray,
        eval_config: Optional[EvolMergeConfiguration] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        future_result = asyncio.Future()
        self.input_queue.append((genotype, future_result, eval_config, context))
        self._set_status(stage="queued", active=False, context=context)
        return await future_result

    async def process_queue(self):
        ray = _require_ray()
        merge_model_ray = ray.remote(
            num_cpus=1,
            num_gpus=1,
            max_retries=3,
            retry_exceptions=[ConnectionError],
        )(merge_model)
        merge_model_ray_cpu = ray.remote(
            num_cpus=1,
            max_retries=3,
            retry_exceptions=[ConnectionError],
        )(merge_model)
        evaluate_model_ray = ray.remote(num_cpus=1, num_gpus=1.0)(evaluate_model)
        evaluate_model_ray_cpu = ray.remote(num_cpus=1)(evaluate_model_cpu)
        merging: Dict[ray.ObjectRef, asyncio.Future] = {}
        merged: List[Tuple[asyncio.Future, ray.ObjectRef]] = []
        evaluating: Dict[ray.ObjectRef, asyncio.Future] = {}

        logging.info("Starting processing loop")

        try:
            while not self._shutdown:
                merge_capacity = (
                    self.num_gpus if self.num_gpus > 0 else self.num_workers
                )
                eval_capacity = (
                    max(1, self.num_gpus // self.eval_gpus_per_task)
                    if self.num_gpus > 0 and self.eval_gpus_per_task > 0
                    else self.num_workers
                )
                while self.input_queue and (
                    len(merging) + len(merged) < merge_capacity
                ):
                    genotype, future_result, eval_config, context = (
                        self.input_queue.pop(0)
                    )
                    self._set_status(stage="merge", active=True, context=context)
                    if self.num_gpus > 0:
                        merging[
                            merge_model_ray.remote(
                                genotype,
                                self.genome,
                                self.model_storage_path,
                                self.merge_options,
                            )
                        ] = (future_result, eval_config, context)
                    else:
                        merging[
                            merge_model_ray_cpu.remote(
                                genotype,
                                self.genome,
                                self.model_storage_path,
                                self.merge_options,
                            )
                        ] = (future_result, eval_config, context)

                while merged and len(evaluating) < eval_capacity:
                    future_result, merged_path, eval_config, context = merged.pop()
                    self._set_status(stage="evaluate", active=True, context=context)
                    config = eval_config or self.config
                    kwargs = {}
                    if self.quantization_config is not None:
                        kwargs["quantization_config"] = self.quantization_config
                    if self.num_gpus > 0:
                        evaluating[
                            evaluate_model_ray.options(
                                num_gpus=self.eval_gpus_per_task
                            ).remote(
                                merged_path,
                                config.tasks,
                                num_fewshot=config.num_fewshot,
                                limit=config.limit,
                                vllm=self.vllm,
                                tensor_parallel_size=self.tensor_parallel_size,
                                batch_size=self.batch_size,
                                task_manager=self.task_manager,
                                apply_chat_template=config.apply_chat_template,
                                fewshot_as_multiturn=config.fewshot_as_multiturn,
                                **kwargs,
                            )
                        ] = (future_result, context)
                    else:
                        evaluating[
                            evaluate_model_ray_cpu.remote(
                                merged_path,
                                config.tasks,
                                num_fewshot=config.num_fewshot,
                                limit=config.limit,
                                batch_size=self.batch_size,
                                task_manager=self.task_manager,
                                apply_chat_template=config.apply_chat_template,
                                fewshot_as_multiturn=config.fewshot_as_multiturn,
                            )
                        ] = (future_result, context)

                ready, _ = ray.wait(
                    list(merging.keys()) + list(evaluating.keys()),
                    num_returns=1,
                    fetch_local=False,
                    timeout=1,
                )
                for r in ready:
                    if r in merging:
                        future_result, eval_config, context = merging.pop(r)
                        merged.append((future_result, r, eval_config, context))
                    elif r in evaluating:
                        future_result, context = evaluating.pop(r)
                        result = await r
                        self._set_status(
                            stage="idle",
                            active=False,
                            context=context,
                            score=result.get("score"),
                            error_type=result.get("error_type"),
                        )
                        future_result.set_result(result)

                if (
                    not self.input_queue
                    and not merging
                    and not merged
                    and not evaluating
                ):
                    self._set_status(stage="idle", active=False)
                    await asyncio.sleep(1)
        except Exception as e:
            logging.error("Error in processing loop", exc_info=e)
            raise

    async def shutdown(self):
        self._shutdown = True

    def register_reentrant_parent(self, checkpoint_path: str) -> int:
        from mergekit.evo.reentry import register_reentrant_parent

        return register_reentrant_parent(self.genome, checkpoint_path)


class BufferedRayEvaluationStrategy(EvaluationStrategyBase):
    def __init__(
        self,
        *args,
        vllm: bool = False,
        in_memory: bool = False,
        **kwargs,
    ):
        if in_memory:
            raise ValueError("In-memory evaluation is not supported for buffered mode")

        super().__init__(*args, **kwargs)
        ray = _require_ray()
        actor_name = worker_actor_name(self.run_label, "buffered", 0)
        actor_cls = ray.remote(BufferedRayEvaluationStrategyActor)
        self.actor = actor_cls.options(
            max_concurrency=1000,
            name=actor_name,
        ).remote(
            self.config,
            self.genome,
            self.merge_options,
            model_storage_path=self.model_storage_path,
            vllm=vllm,
            num_gpus=self.num_gpus,
            num_workers=self.num_workers,
            tensor_parallel_size=self.tensor_parallel_size,
            task_manager=self.task_manager,
            batch_size=self.batch_size,
            quantization_config=self.quantization_config,
            worker_name=actor_name,
            observer_config=(
                self.run_observer.export_config(role="worker")
                if self.run_observer is not None
                else None
            ),
        )
        self.actor.process_queue.remote()

    def _propagate_reentrant_parent(self, checkpoint_path: str) -> None:
        ray = _require_ray()
        ray.get(self.actor.register_reentrant_parent.remote(checkpoint_path))

    def _evaluate_genotypes_once(
        self, genotypes: List[np.ndarray], eval_config: EvolMergeConfiguration
    ) -> List[dict]:
        ray = _require_ray()
        candidate_contexts = self._candidate_contexts(genotypes, eval_config)
        return ray.get(
            [
                self.actor.evaluate_genotype.remote(x, eval_config, context)
                for x, context in zip(genotypes, candidate_contexts)
            ]
        )


def evaluate_genotype_serial(
    genotype: np.ndarray,
    config: EvolMergeConfiguration,
    genome: ModelGenome,
    merge_options: MergeOptions,
    model_storage_path: Optional[str] = None,
    vllm: bool = False,
    tensor_parallel_size: int = 1,
    batch_size: Optional[int] = None,
    task_manager: Optional[Any] = None,
    quantization_config: Optional[transformers.BitsAndBytesConfig] = None,
    context: Optional[Dict[str, Any]] = None,
):
    ray = _require_ray()
    gpus_per_eval = _gpus_per_evaluation(
        total_gpus=tensor_parallel_size,
        vllm=vllm,
        tensor_parallel_size=tensor_parallel_size,
    )
    pg = ray.util.placement_group(
        [{"CPU": 1, "GPU": gpus_per_eval}], strategy="STRICT_PACK"
    )
    strat = ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
        placement_group=pg
    )
    merge_remote = ray.remote(
        num_cpus=1,
        num_gpus=1,
        max_retries=3,
        retry_exceptions=[ConnectionError],
    )(merge_model_with_details)
    evaluate_remote = ray.remote(num_cpus=1, num_gpus=1.0)(evaluate_model)
    merge_info = merge_remote.options(scheduling_strategy=strat).remote(
        genotype, genome, model_storage_path, merge_options
    )
    merge_info = ray.get(merge_info)
    merged_path = merge_info.get("merged_path")
    if not merged_path:
        return {
            "score": None,
            "results": None,
            "error_stage": merge_info.get("error_stage", "merge"),
            "error_type": merge_info.get("error_type", "merge_failed"),
            "error_message": merge_info.get("error_message", "Model merge failed"),
        }
    kwargs = {}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    eval_config = config
    res = ray.get(
        evaluate_remote.options(
            scheduling_strategy=strat,
            num_gpus=gpus_per_eval,
        ).remote(
            merged_path,
            eval_config.tasks,
            num_fewshot=eval_config.num_fewshot,
            limit=eval_config.limit,
            vllm=vllm,
            tensor_parallel_size=tensor_parallel_size,
            batch_size=batch_size,
            task_manager=task_manager,
            apply_chat_template=eval_config.apply_chat_template,
            fewshot_as_multiturn=eval_config.fewshot_as_multiturn,
            **kwargs,
        )
    )
    ray.util.remove_placement_group(pg)
    return res


def _evaluate_genotype_serial_cpu_impl(
    genotype: np.ndarray,
    config: EvolMergeConfiguration,
    genome: ModelGenome,
    merge_options: MergeOptions,
    model_storage_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    task_manager: Optional[Any] = None,
    repair_callback=None,
    repair_eval_config: Optional[EvolMergeConfiguration] = None,
    retain_repaired_checkpoint: bool = False,
):
    import sys
    import time

    start_time = time.perf_counter()
    print(f"[EVAL] Starting genotype evaluation...", flush=True)
    sys.stdout.flush()

    print(f"[EVAL] Step 1/2: Merging models...", flush=True)
    sys.stdout.flush()
    merge_start = time.perf_counter()
    merge_info = merge_model_with_details(
        genotype,
        genome,
        model_storage_path,
        merge_options,
    )
    merged_path = merge_info.get("merged_path")
    if not merged_path:
        merge_seconds = time.perf_counter() - merge_start
        total_seconds = time.perf_counter() - start_time
        print(
            "[EVAL] Merge failed after "
            f"{merge_seconds:.1f}s; skipping evaluation. "
            f"reason={merge_info.get('error_type', 'merge_failed')}: "
            f"{merge_info.get('error_message', 'Model merge failed')}",
            flush=True,
        )
        print(f"[EVAL] Total time: {total_seconds:.1f}s | Score: N/A", flush=True)
        sys.stdout.flush()
        return {
            "score": None,
            "results": None,
            "error_stage": merge_info.get("error_stage", "merge"),
            "error_type": merge_info.get("error_type", "merge_failed"),
            "error_message": merge_info.get("error_message", "Model merge failed"),
            "error": "merge_failed",
        }
    print(
        f"[EVAL] Merge completed in {time.perf_counter() - merge_start:.1f}s",
        flush=True,
    )
    sys.stdout.flush()

    print(f"[EVAL] Step 2/2: Evaluating merged model on {config.tasks}...", flush=True)
    sys.stdout.flush()
    eval_start = time.perf_counter()

    def evaluate_checkpoint(eval_config, *, cleanup: bool):
        return evaluate_model_cpu(
            merged_path,
            eval_config.tasks,
            num_fewshot=eval_config.num_fewshot,
            limit=eval_config.limit,
            batch_size=batch_size,
            task_manager=task_manager,
            fitness_mode=getattr(eval_config, "fitness_mode", "weighted_sum"),
            fitness_version=getattr(
                getattr(eval_config, "fitness", None), "version", "v1"
            ),
            lower_is_better_transform=getattr(
                getattr(eval_config, "fitness", None),
                "lower_is_better_transform",
                "legacy_reciprocal",
            ),
            task_mix_profile=getattr(eval_config, "task_mix_profile", None),
            behavior_prompts=getattr(eval_config, "behavior_prompts", None),
            behavior_probe_max_new_tokens=getattr(
                eval_config, "behavior_probe_max_new_tokens", 24
            ),
            behavior_repetition_ngram_size=getattr(
                eval_config, "behavior_repetition_ngram_size", 4
            ),
            behavior_min_distinct_ratio=getattr(
                eval_config, "behavior_min_distinct_ratio", 0.2
            ),
            behavior_reject_on_degenerate=getattr(
                eval_config, "behavior_reject_on_degenerate", False
            ),
            cleanup_merged_path=cleanup,
        )

    repair_metadata = None
    if repair_callback is not None:
        comparison_config = repair_eval_config or config
        pre_repair = evaluate_checkpoint(comparison_config, cleanup=False)
        print("[EVAL] Running gated Stage-2 repair...", flush=True)
        repair_metadata = repair_callback(merged_path, genotype, comparison_config)
        retain = bool(retain_repaired_checkpoint and repair_metadata.get("repaired"))
        res = evaluate_checkpoint(comparison_config, cleanup=not retain)
        res = dict(res)
        res["repair_pre_score"] = pre_repair.get("score")
        res["repair_post_score"] = res.get("score")
        res["repair_comparison_limit"] = comparison_config.limit
        res["repair_comparison_audited"] = comparison_config is not config
        if retain:
            res["reentry_checkpoint_path"] = merged_path
        elif os.path.exists(merged_path):
            shutil.rmtree(merged_path, ignore_errors=True)
    else:
        res = evaluate_checkpoint(config, cleanup=True)
    eval_seconds = time.perf_counter() - eval_start
    if res.get("score") is None:
        print(
            f"[EVAL] Evaluation completed in {eval_seconds:.1f}s but returned no score",
            flush=True,
        )
    else:
        print(f"[EVAL] Evaluation completed in {eval_seconds:.1f}s", flush=True)
    print(
        f"[EVAL] Total time: {time.perf_counter() - start_time:.1f}s | Score: {res.get('score', 'N/A')}",
        flush=True,
    )
    sys.stdout.flush()
    if repair_metadata is not None:
        res = dict(res)
        res["repair"] = repair_metadata
    return res


def evaluate_genotype_serial_cpu(
    genotype: np.ndarray,
    config: EvolMergeConfiguration,
    genome: ModelGenome,
    merge_options: MergeOptions,
    model_storage_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    task_manager: Optional[Any] = None,
):
    return _evaluate_genotype_serial_cpu_impl(
        genotype,
        config,
        genome,
        merge_options,
        model_storage_path=model_storage_path,
        batch_size=batch_size,
        task_manager=task_manager,
    )


class SerialEvaluationStrategy(EvaluationStrategyBase):
    def __init__(
        self,
        *args,
        vllm: bool = False,
        in_memory: bool = False,
        **kwargs,
    ):
        self.vllm = vllm
        self._repair_distiller = None
        self._repair_corpus = None
        if in_memory:
            raise ValueError("In-memory evaluation is not supported for serial mode")
        super().__init__(*args, **kwargs)

    def repair_checkpoint(
        self,
        checkpoint_path: str,
        genotype: np.ndarray,
        eval_config: Optional[EvolMergeConfiguration] = None,
    ) -> Dict[str, Any]:
        config = eval_config or self.config
        repair_config = getattr(config, "repair", None)
        if repair_config is None or not repair_config.enabled:
            return {"repaired": False, "skipped": True}
        if self.num_gpus and self.num_gpus > 0:
            raise ValueError(
                "Gated repair currently supports CPU serial execution only"
            )

        from mergekit.evo.repair import (
            ParentDistiller,
            blend_weights_for_genotype,
            load_repair_corpus,
            repair_checkpoint,
        )

        if self._repair_distiller is None:
            parent_refs = [
                str(model)
                for model in getattr(
                    getattr(self.genome, "definition", None), "models", []
                )
            ]
            self._repair_distiller = ParentDistiller(
                parent_refs,
                tau_distill=repair_config.tau_distill,
                trust_remote_code=self.merge_options.trust_remote_code,
            )
        if self._repair_corpus is None:
            self._repair_corpus = load_repair_corpus(
                repair_config.corpus,
                max_examples=repair_config.batch_size * repair_config.max_steps,
            )

        genotype_array = np.asarray(genotype, dtype=np.float32).reshape(-1)
        genotype_seed = int(hashlib.sha1(genotype_array.tobytes()).hexdigest()[:8], 16)
        base_seed = int(self.merge_options.random_seed or 0)
        return repair_checkpoint(
            checkpoint_path,
            self._repair_distiller,
            self._repair_corpus,
            config=repair_config,
            weights=blend_weights_for_genotype(self.genome, genotype_array),
            seed=(base_seed + genotype_seed) % (2**31),
            trust_remote_code=self.merge_options.trust_remote_code,
        )

    def _evaluate_genotypes_once(
        self, genotypes: List[np.ndarray], eval_config: EvolMergeConfiguration
    ) -> List[dict]:
        import sys

        print(
            f"\n[SERIAL] Evaluating {len(genotypes)} genotypes in serial mode...",
            flush=True,
        )
        sys.stdout.flush()

        if self.num_gpus and self.num_gpus > 0:
            ray = _require_ray()
            print(f"[SERIAL] Using GPU path with {self.num_gpus} GPUs", flush=True)
            sys.stdout.flush()
            candidate_contexts = self._candidate_contexts(genotypes, eval_config)
            remote_evaluator = ray.remote(evaluate_genotype_serial)
            return ray.get(
                [
                    remote_evaluator.remote(
                        x,
                        eval_config,
                        self.genome,
                        self.merge_options,
                        model_storage_path=self.model_storage_path,
                        vllm=self.vllm,
                        tensor_parallel_size=self.tensor_parallel_size,
                        batch_size=self.batch_size,
                        task_manager=self.task_manager,
                        quantization_config=self.quantization_config,
                        context=context,
                    )
                    for x, context in zip(genotypes, candidate_contexts)
                ]
            )
        else:
            # CPU-only path: no GPUs available
            print(f"[SERIAL] Using CPU-only path (no GPUs detected)", flush=True)
            sys.stdout.flush()
            results = []
            total = len(genotypes)
            candidate_contexts = self._candidate_contexts(genotypes, eval_config)
            for idx, (genotype, context) in enumerate(
                zip(genotypes, candidate_contexts), start=1
            ):
                if getattr(self, "run_observer", None) is not None:
                    self.run_observer.record_candidate_start(
                        phase=str(
                            context.get("evaluation_stage")
                            or getattr(self, "current_phase", "ga")
                        ),
                        candidate_index=idx - 1,
                        generation=context.get("generation"),
                        method=context.get("merge_method"),
                    )
                print(
                    f"[SERIAL] Evaluating genotype {idx}/{total} sequentially...",
                    flush=True,
                )
                sys.stdout.flush()
                audit_config = getattr(self.config, "audit", None)
                repair_eval_config = None
                if (
                    getattr(getattr(eval_config, "repair", None), "enabled", False)
                    and audit_config is not None
                    and audit_config.enabled
                ):
                    repair_eval_config = eval_config.model_copy(
                        update={"limit": audit_config.limit, "two_stage": False}
                    )
                result = _evaluate_genotype_serial_cpu_impl(
                    genotype,
                    eval_config,
                    self.genome,
                    self.merge_options,
                    model_storage_path=self.model_storage_path,
                    batch_size=self.batch_size,
                    task_manager=self.task_manager,
                    repair_callback=(
                        self.repair_checkpoint
                        if getattr(
                            getattr(eval_config, "repair", None), "enabled", False
                        )
                        else None
                    ),
                    repair_eval_config=repair_eval_config,
                    retain_repaired_checkpoint=bool(
                        getattr(getattr(eval_config, "repair", None), "reentry", False)
                    ),
                )
                results.append(result)
                if getattr(self, "run_observer", None) is not None:
                    self.run_observer.record_candidate_end(
                        phase=str(
                            context.get("evaluation_stage")
                            or getattr(self, "current_phase", "ga")
                        ),
                        candidate_index=idx - 1,
                        generation=context.get("generation"),
                        method=context.get("merge_method"),
                        score=result.get("score"),
                        failed=result.get("score") is None,
                    )
            print(f"[SERIAL] All {len(genotypes)} evaluations completed!", flush=True)
            sys.stdout.flush()
            return results
