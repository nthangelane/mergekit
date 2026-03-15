# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import asyncio
import logging
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
from mergekit.evo.task_utils import create_task_manager
from mergekit.options import MergeOptions


class EvaluationStrategyBase(ABC):
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        num_gpus: Optional[int] = None,
        num_workers: Optional[int] = None,
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
        self.task_manager = create_task_manager(task_search_path)
        self.model_storage_path = model_storage_path
        self.quantization_config = quantization_config
        if self.model_storage_path:
            os.makedirs(self.model_storage_path, exist_ok=True)

    @abstractmethod
    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[dict]:
        pass

    @abstractmethod
    def evaluate_genotype(self, genotype: np.ndarray) -> dict:
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
            self.num_gpus
            if (self.num_gpus and self.num_gpus > 0)
            else (self.num_workers or 1)
        )
        self.actor_pool = ray.util.ActorPool(
            [
                self.actor_cls.remote(
                    self.config,
                    self.genome,
                    self.merge_options,
                    model_storage_path=self.model_storage_path,
                    vllm=vllm,
                    batch_size=self.batch_size,
                    task_manager=self.task_manager,
                    quantization_config=self.quantization_config,
                )
                for _ in range(worker_count)
            ]
        )

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[dict]:
        return list(
            self.actor_pool.map(
                lambda a, x: a.evaluate_genotype.remote(x),
                genotypes,
            )
        )

    def evaluate_genotype(self, genotype: np.ndarray) -> dict:
        return self.evaluate_genotypes([genotype])[0]


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
        self.input_queue = []
        self.batch_size = batch_size
        self.task_manager = task_manager
        self.model_storage_path = model_storage_path
        self.quantization_config = quantization_config
        self._shutdown = False

    async def evaluate_genotype(self, genotype: np.ndarray):
        future_result = asyncio.Future()
        self.input_queue.append((genotype, future_result))
        return await future_result

    async def process_queue(self):
        merging: Dict[ray.ObjectRef, asyncio.Future] = {}
        merged: List[Tuple[asyncio.Future, ray.ObjectRef]] = []
        evaluating: Dict[ray.ObjectRef, asyncio.Future] = {}

        logging.info("Starting processing loop")

        try:
            while not self._shutdown:
                capacity = self.num_gpus if self.num_gpus > 0 else self.num_workers
                while self.input_queue and (len(merging) + len(merged) < capacity):
                    genotype, future_result = self.input_queue.pop(0)
                    if self.num_gpus > 0:
                        merging[
                            merge_model_ray.remote(
                                genotype,
                                self.genome,
                                self.model_storage_path,
                                self.merge_options,
                            )
                        ] = future_result
                    else:
                        merging[
                            merge_model_ray_cpu.remote(
                                genotype,
                                self.genome,
                                self.model_storage_path,
                                self.merge_options,
                            )
                        ] = future_result

                while merged and len(evaluating) < capacity:
                    future_result, merged_path = merged.pop()
                    kwargs = {}
                    if self.quantization_config is not None:
                        kwargs["quantization_config"] = self.quantization_config
                    if self.num_gpus > 0:
                        evaluating[
                            evaluate_model_ray.remote(
                                merged_path,
                                self.config.tasks,
                                num_fewshot=self.config.num_fewshot,
                                limit=self.config.limit,
                                vllm=self.vllm,
                                batch_size=self.batch_size,
                                task_manager=self.task_manager,
                                apply_chat_template=self.config.apply_chat_template,
                                fewshot_as_multiturn=self.config.fewshot_as_multiturn,
                                **kwargs,
                            )
                        ] = future_result
                    else:
                        evaluating[
                            evaluate_model_ray_cpu.remote(
                                merged_path,
                                self.config.tasks,
                                num_fewshot=self.config.num_fewshot,
                                limit=self.config.limit,
                                batch_size=self.batch_size,
                                task_manager=self.task_manager,
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
                        future_result = merging.pop(r)
                        merged.append((future_result, r))
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
            task_manager=self.task_manager,
            batch_size=self.batch_size,
            quantization_config=self.quantization_config,
        )
        self.actor.process_queue.remote()

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[dict]:
        return ray.get([self.actor.evaluate_genotype.remote(x) for x in genotypes])

    def evaluate_genotype(self, genotype: np.ndarray) -> dict:
        return ray.get(self.actor.evaluate_genotype.remote(genotype))


@ray.remote
def evaluate_genotype_serial(
    genotype: np.ndarray,
    config: EvolMergeConfiguration,
    genome: ModelGenome,
    merge_options: MergeOptions,
    model_storage_path: Optional[str] = None,
    vllm: bool = False,
    batch_size: Optional[int] = None,
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
    quantization_config: Optional[transformers.BitsAndBytesConfig] = None,
):
    pg = ray.util.placement_group([{"CPU": 1, "GPU": 1}], strategy="STRICT_PACK")
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
    res = ray.get(
        evaluate_model_ray.options(scheduling_strategy=strat).remote(
            merged_path,
            config.tasks,
            num_fewshot=config.num_fewshot,
            limit=config.limit,
            vllm=vllm,
            batch_size=batch_size,
            task_manager=task_manager,
            apply_chat_template=config.apply_chat_template,
            fewshot_as_multiturn=config.fewshot_as_multiturn,
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

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[dict]:
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
                        self.config,
                        self.genome,
                        self.merge_options,
                        model_storage_path=self.model_storage_path,
                        vllm=self.vllm,
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
                        self.config,
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

    def evaluate_genotype(self, genotype: np.ndarray) -> dict:
        return self.evaluate_genotypes([genotype])[0]
