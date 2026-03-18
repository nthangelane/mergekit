# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import asyncio
import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import lm_eval.tasks
import numpy as np
import ray
import ray.util.queue
import ray.util.scheduling_strategies
import torch
import transformers

from mergekit.common import get_torch_accelerator_count
from mergekit.evo.actors import (
    InMemoryMergeEvaluator,
    OnDiskMergeEvaluator,
    OnDiskMergeEvaluatorCPU,
)
from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.genome import ModelGenome
from mergekit.evo.helpers import (
    evaluate_model_cpu,
    evaluate_model_ray,
    evaluate_model_ray_cpu,
    merge_model_ray,
    merge_model_ray_cpu,
    merge_model_with_details,
    merge_model_with_details_ray,
)
from mergekit.evo.ranking import weighted_rank_scores
from mergekit.evo.task_utils import create_task_manager
from mergekit.options import MergeOptions


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
        self.task_manager = create_task_manager(task_search_path)
        self.model_storage_path = model_storage_path
        self.quantization_config = quantization_config
        if self.model_storage_path:
            os.makedirs(self.model_storage_path, exist_ok=True)

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[dict]:
        if not genotypes:
            return []
        if not getattr(self.config, "two_stage", False) or len(genotypes) == 1:
            return self._evaluate_genotypes_once(genotypes, self.config)

        stage1_config = self._stage_config(stage=1)
        stage2_config = self._stage_config(stage=2)
        stage1_results = self._evaluate_genotypes_once(genotypes, stage1_config)

        combined_results: List[dict] = []
        successful_indices: List[int] = []
        for idx, stage1_result in enumerate(stage1_results):
            result = dict(stage1_result)
            result["stage1_score"] = stage1_result.get("score")
            result["stage1_results"] = stage1_result.get("results")
            result["stage1_limit"] = stage1_config.limit
            result["stage1_tasks"] = [task.name for task in stage1_config.tasks]
            result["evaluation_stage"] = "stage1"
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

        return combined_results

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
        self.actor_pool = ray.util.ActorPool(
            [
                (
                    self.actor_cls.options(num_gpus=actor_gpu_request).remote
                    if actor_gpu_request > 0
                    else self.actor_cls.remote
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
                )
                for _ in range(worker_count)
            ]
        )

    def _evaluate_genotypes_once(
        self, genotypes: List[np.ndarray], eval_config: EvolMergeConfiguration
    ) -> List[dict]:
        return list(
            self.actor_pool.map(
                lambda a, x: a.evaluate_genotype.remote(x, eval_config),
                genotypes,
            )
        )


@ray.remote
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
        task_manager: Optional[lm_eval.tasks.TaskManager] = None,
        model_storage_path: Optional[str] = None,
        quantization_config: Optional[transformers.BitsAndBytesConfig] = None,
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

    async def evaluate_genotype(
        self,
        genotype: np.ndarray,
        eval_config: Optional[EvolMergeConfiguration] = None,
    ):
        future_result = asyncio.Future()
        self.input_queue.append((genotype, future_result, eval_config))
        return await future_result

    async def process_queue(self):
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
                    genotype, future_result, eval_config = self.input_queue.pop(0)
                    if self.num_gpus > 0:
                        merging[
                            merge_model_ray.remote(
                                genotype,
                                self.genome,
                                self.model_storage_path,
                                self.merge_options,
                            )
                        ] = (future_result, eval_config)
                    else:
                        merging[
                            merge_model_ray_cpu.remote(
                                genotype,
                                self.genome,
                                self.model_storage_path,
                                self.merge_options,
                            )
                        ] = (future_result, eval_config)

                while merged and len(evaluating) < eval_capacity:
                    future_result, merged_path, eval_config = merged.pop()
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
                        ] = future_result
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
                        ] = future_result

                ready, _ = ray.wait(
                    list(merging.keys()) + list(evaluating.keys()),
                    num_returns=1,
                    fetch_local=False,
                    timeout=1,
                )
                for r in ready:
                    if r in merging:
                        future_result, eval_config = merging.pop(r)
                        merged.append((future_result, r, eval_config))
                    elif r in evaluating:
                        future_result = evaluating.pop(r)
                        future_result.set_result(await r)

                if (
                    not self.input_queue
                    and not merging
                    and not merged
                    and not evaluating
                ):
                    await asyncio.sleep(1)
        except Exception as e:
            logging.error("Error in processing loop", exc_info=e)
            raise

    async def shutdown(self):
        self._shutdown = True


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
        self.actor = BufferedRayEvaluationStrategyActor.options(
            max_concurrency=1000
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
        )
        self.actor.process_queue.remote()

    def _evaluate_genotypes_once(
        self, genotypes: List[np.ndarray], eval_config: EvolMergeConfiguration
    ) -> List[dict]:
        return ray.get(
            [self.actor.evaluate_genotype.remote(x, eval_config) for x in genotypes]
        )


@ray.remote
def evaluate_genotype_serial(
    genotype: np.ndarray,
    config: EvolMergeConfiguration,
    genome: ModelGenome,
    merge_options: MergeOptions,
    model_storage_path: Optional[str] = None,
    vllm: bool = False,
    tensor_parallel_size: int = 1,
    batch_size: Optional[int] = None,
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
    quantization_config: Optional[transformers.BitsAndBytesConfig] = None,
):
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
    merge_info = merge_model_with_details_ray.options(scheduling_strategy=strat).remote(
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
        evaluate_model_ray.options(
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
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
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
    res = evaluate_model_cpu(
        merged_path,
        config.tasks,
        num_fewshot=config.num_fewshot,
        limit=config.limit,
        batch_size=batch_size,
        task_manager=task_manager,
        fitness_mode=getattr(config, "fitness_mode", "weighted_sum"),
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
    )
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
    return res


@ray.remote
def evaluate_genotype_serial_cpu(
    genotype: np.ndarray,
    config: EvolMergeConfiguration,
    genome: ModelGenome,
    merge_options: MergeOptions,
    model_storage_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
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
        if in_memory:
            raise ValueError("In-memory evaluation is not supported for serial mode")
        super().__init__(*args, **kwargs)

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
            print(f"[SERIAL] Using GPU path with {self.num_gpus} GPUs", flush=True)
            sys.stdout.flush()
            return ray.get(
                [
                    evaluate_genotype_serial.remote(
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
                    )
                    for x in genotypes
                ]
            )
        else:
            # CPU-only path: no GPUs available
            print(f"[SERIAL] Using CPU-only path (no GPUs detected)", flush=True)
            sys.stdout.flush()
            results = []
            total = len(genotypes)
            for idx, genotype in enumerate(genotypes, start=1):
                print(
                    f"[SERIAL] Evaluating genotype {idx}/{total} sequentially...",
                    flush=True,
                )
                sys.stdout.flush()
                results.append(
                    _evaluate_genotype_serial_cpu_impl(
                        genotype,
                        eval_config,
                        self.genome,
                        self.merge_options,
                        model_storage_path=self.model_storage_path,
                        batch_size=self.batch_size,
                        task_manager=self.task_manager,
                    )
                )
            print(f"[SERIAL] All {len(genotypes)} evaluations completed!", flush=True)
            sys.stdout.flush()
            return results
