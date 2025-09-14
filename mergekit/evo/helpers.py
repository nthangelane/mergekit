# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Union

import lm_eval
import lm_eval.api.model
import lm_eval.models.huggingface
import lm_eval.tasks
import ray
import ray.util.queue
import ray.util.scheduling_strategies
import torch

from mergekit.evo.config import TaskConfiguration
from mergekit.evo.genome import InvalidGenotypeError, ModelGenome
from mergekit.evo.monkeypatch import monkeypatch_lmeval_vllm
from mergekit.merge import run_merge
from mergekit.options import MergeOptions


def _eval_model(
    model: Union[str, lm_eval.api.model.LM],
    tasks: List[TaskConfiguration],
    model_args: Optional[Dict[str, Any]] = None,
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
    **kwargs,
) -> Dict[str, Any]:
    results = lm_eval.simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=list(set([task.name for task in tasks])),
        log_samples=False,
        verbosity="WARNING",
        task_manager=task_manager,
        **kwargs,
    )

    logging.info(results["results"])
    res = 0
    for task in tasks:
        if task.name not in results["results"]:
            logging.warning(f"Task {task.name} not found in results")
            continue
            
        task_results = results["results"][task.name]
        metric_value = None
        
        # Try the exact metric first
        if task.metric in task_results:
            metric_value = task_results[task.metric]
        else:
            # Auto-detect common metric alternatives
            metric_alternatives = {
                "ppl,none": ["word_perplexity,none", "perplexity,none", "byte_perplexity,none"],
                "acc,none": ["acc,none", "acc_norm,none", "accuracy,none"],
                "acc_norm,none": ["acc_norm,none", "acc,none", "accuracy,none"],
            }
            
            alternatives = metric_alternatives.get(task.metric, [])
            for alt_metric in alternatives:
                if alt_metric in task_results:
                    metric_value = task_results[alt_metric]
                    logging.info(f"Auto-detected metric for {task.name}: {alt_metric} instead of {task.metric}")
                    break
            
            # If still not found, try pattern matching
            if metric_value is None:
                available_metrics = list(task_results.keys())
                if "ppl" in task.metric or "perplexity" in task.metric:
                    # Look for any perplexity metric
                    for metric in available_metrics:
                        if "perplexity" in metric.lower() and "stderr" not in metric:
                            metric_value = task_results[metric]
                            logging.info(f"Pattern-matched perplexity metric for {task.name}: {metric}")
                            break
                elif "acc" in task.metric:
                    # Look for any accuracy metric
                    for metric in available_metrics:
                        if "acc" in metric.lower() and "stderr" not in metric:
                            metric_value = task_results[metric]
                            logging.info(f"Pattern-matched accuracy metric for {task.name}: {metric}")
                            break
        
        # Handle invalid values
        if metric_value is None:
            logging.error(f"Could not find metric {task.metric} for task {task.name}. Available: {list(task_results.keys())}")
            continue
        
        # Handle NaN values
        import math
        if isinstance(metric_value, float) and math.isnan(metric_value):
            logging.warning(f"NaN result for {task.name}:{task.metric}, skipping")
            continue
            
        res += metric_value * task.weight
    return {"score": res, "results": results["results"]}


def evaluate_model(
    merged_path: str,
    tasks: List[TaskConfiguration],
    num_fewshot: Optional[int],
    limit: Optional[int],
    vllm: bool,
    batch_size: Optional[int] = None,
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> dict:
    # monkeypatch_tqdm()
    monkeypatch_lmeval_vllm()
    try:
        model_args = {
            "pretrained": merged_path,
            "dtype": "bfloat16",
            **(model_kwargs or {}),
        }
        if vllm:
            model_args["gpu_memory_utilization"] = 0.8
            model_args["tensor_parallel_size"] = 1
            model_args["batch_size"] = "auto"
            model_args["max_model_len"] = 4096
        else:
            model_args["use_cache"] = True

        res = _eval_model(
            "vllm" if vllm else "huggingface",
            tasks,
            model_args,
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=batch_size,
            task_manager=task_manager,
            **kwargs,
        )
        return res
    finally:
        shutil.rmtree(merged_path)


evaluate_model_ray = ray.remote(num_cpus=1, num_gpus=1.0)(evaluate_model)


def evaluate_model_cpu(
    merged_path: str,
    tasks: List[TaskConfiguration],
    num_fewshot: Optional[int],
    limit: Optional[int],
    batch_size: Optional[int] = None,
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
) -> dict:
    """CPU-only evaluation using HuggingFace backend and float32."""
    monkeypatch_lmeval_vllm()
    try:
        model_args = {
            "pretrained": merged_path,
            "dtype": "float32",
            "device": "cpu",
            "use_cache": True,
        }
        res = _eval_model(
            "huggingface",
            tasks,
            model_args,
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=batch_size,
            task_manager=task_manager,
        )
        return res
    finally:
        shutil.rmtree(merged_path)


evaluate_model_ray_cpu = ray.remote(num_cpus=1)(evaluate_model_cpu)


def merge_model(
    genotype: torch.Tensor,
    genome: ModelGenome,
    model_storage_path: str,
    merge_options: MergeOptions,
) -> str:
    # monkeypatch_tqdm()
    try:
        cfg = genome.genotype_merge_config(genotype)
    except InvalidGenotypeError as e:
        logging.error("Invalid genotype", exc_info=e)
        return None
    os.makedirs(model_storage_path, exist_ok=True)
    res = tempfile.mkdtemp(prefix="merged", dir=model_storage_path)
    run_merge(cfg, out_path=res, options=merge_options)
    return res


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
