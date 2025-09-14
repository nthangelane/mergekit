# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import wandb
except ImportError:
    wandb = None

try:
    import mlflow
except (ImportError, ValueError) as e:
    # Handle both ImportError and ValueError (binary compatibility issues)
    mlflow = None
    _mlflow_error = str(e)


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking systems."""

    @abstractmethod
    def initialize(self, project_name: str, config: Dict[str, Any], **kwargs) -> None:
        """Initialize the tracking system."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log scalar metrics."""
        pass

    @abstractmethod
    def log_population_stats(self, results: List[Dict], step: int) -> None:
        """Log population-level statistics."""
        pass

    @abstractmethod
    def log_best_individual(
        self, genotype: np.ndarray, score: float, step: int, genome=None
    ) -> None:
        """Log the best individual found so far."""
        pass

    @abstractmethod
    def log_artifact(self, file_path: str, artifact_name: str) -> None:
        """Log an artifact (file)."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """Clean up and finish tracking."""
        pass


class WandBTracker(ExperimentTracker):
    """Weights & Biases experiment tracker."""

    def __init__(self):
        self.run = None

    def initialize(
        self,
        project_name: str,
        config: Dict[str, Any],
        entity: Optional[str] = None,
        **kwargs,
    ) -> None:
        if not wandb:
            raise RuntimeError(
                "wandb is not installed. Install with: pip install wandb"
            )

        self.run = wandb.init(
            project=project_name, entity=entity, config=config, **kwargs
        )
        logging.info(f"Initialized W&B tracking: {project_name}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.run:
            self.run.log(metrics, step=step, commit=False)

    def log_population_stats(self, results: List[Dict], step: int) -> None:
        if not self.run:
            return

        try:
            score_vals = [r["score"] for r in results if r["score"] is not None]
            if score_vals:
                self.run.log(
                    {
                        "population/score_mean": float(np.mean(score_vals)),
                        "population/score_std": float(np.std(score_vals)),
                        "population/score_min": float(np.min(score_vals)),
                        "population/score_max": float(np.max(score_vals)),
                    },
                    commit=False,
                    step=step,
                )

            # Log per-task stats
            if results and results[0].get("results"):
                for task in results[0]["results"]:
                    for metric in results[0]["results"][task]:
                        values = [
                            r["results"][task][metric]
                            for r in results
                            if r.get("results", {}).get(task, {}).get(metric)
                            is not None
                        ]
                        if not values or all(isinstance(v, str) for v in values):
                            continue
                        metric_pretty = metric.replace(",none", "")
                        if metric_pretty.endswith("_stderr"):
                            continue
                        self.run.log(
                            {
                                f"population/{task}_{metric_pretty}_mean": float(
                                    np.mean(values)
                                ),
                                f"population/{task}_{metric_pretty}_max": float(
                                    np.max(values)
                                ),
                                f"population/{task}_{metric_pretty}_min": float(
                                    np.min(values)
                                ),
                            },
                            commit=False,
                            step=step,
                        )
        except Exception as e:
            logging.warning("Failed to log population metrics to wandb", exc_info=e)

    def log_best_individual(
        self, genotype: np.ndarray, score: float, step: int, genome=None
    ) -> None:
        if not self.run:
            return

        try:
            metrics = {"best_score": float(score)}

            if genome:
                best_params = genome.genotype_to_param_arrays(genotype)
                metrics["best_genome"] = wandb.Table(data=pd.DataFrame(best_params))

            self.run.log(metrics, commit=True, step=step)
        except Exception as e:
            logging.warning("Failed to log best individual to wandb", exc_info=e)

    def log_artifact(self, file_path: str, artifact_name: str) -> None:
        if self.run:
            try:
                artifact = wandb.Artifact(artifact_name, type="config")
                artifact.add_file(file_path)
                self.run.log_artifact(artifact)
            except Exception as e:
                logging.warning(
                    f"Failed to log artifact {artifact_name} to wandb", exc_info=e
                )

    def finish(self) -> None:
        if self.run:
            self.run.finish()
            self.run = None


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker."""

    def __init__(self):
        self.run_id = None
        self.experiment_id = None

    def initialize(
        self,
        project_name: str,
        config: Dict[str, Any],
        tracking_uri: Optional[str] = None,
        **kwargs,
    ) -> None:
        if not mlflow:
            if "_mlflow_error" in globals():
                raise RuntimeError(
                    f"mlflow import failed: {_mlflow_error}. Try: pip install --upgrade pyarrow mlflow"
                )
            else:
                raise RuntimeError(
                    "mlflow is not installed. Install with: pip install mlflow"
                )

        # Set tracking URI with environment variable fallback
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "file://./mlruns"
        mlflow.set_tracking_uri(uri)

        # Use environment variable for experiment name if not provided
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", project_name)

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(project_name)
            else:
                self.experiment_id = experiment.experiment_id
        except Exception:
            self.experiment_id = mlflow.create_experiment(project_name)

        # Start run
        mlflow.start_run(experiment_id=self.experiment_id)
        self.run_id = mlflow.active_run().info.run_id

        # Log configuration
        for key, value in config.items():
            try:
                if isinstance(value, (dict, list)):
                    mlflow.log_param(key, str(value))
                else:
                    mlflow.log_param(key, value)
            except Exception as e:
                logging.warning(f"Failed to log param {key}: {e}")

        logging.info(
            f"Initialized MLflow tracking: {project_name} (run: {self.run_id})"
        )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logging.warning(f"Failed to log metrics to mlflow: {e}")

    def log_population_stats(self, results: List[Dict], step: int) -> None:
        try:
            score_vals = [r["score"] for r in results if r["score"] is not None]
            if score_vals:
                metrics = {
                    "population/score_mean": float(np.mean(score_vals)),
                    "population/score_std": float(np.std(score_vals)),
                    "population/score_min": float(np.min(score_vals)),
                    "population/score_max": float(np.max(score_vals)),
                }
                self.log_metrics(metrics, step=step)

            # Log per-task stats
            if results and results[0].get("results"):
                task_metrics = {}
                for task in results[0]["results"]:
                    for metric in results[0]["results"][task]:
                        values = [
                            r["results"][task][metric]
                            for r in results
                            if r.get("results", {}).get(task, {}).get(metric)
                            is not None
                        ]
                        if not values or all(isinstance(v, str) for v in values):
                            continue
                        metric_pretty = metric.replace(",none", "")
                        if metric_pretty.endswith("_stderr"):
                            continue
                        task_metrics.update(
                            {
                                f"population/{task}_{metric_pretty}_mean": float(
                                    np.mean(values)
                                ),
                                f"population/{task}_{metric_pretty}_max": float(
                                    np.max(values)
                                ),
                                f"population/{task}_{metric_pretty}_min": float(
                                    np.min(values)
                                ),
                            }
                        )
                self.log_metrics(task_metrics, step=step)

        except Exception as e:
            logging.warning("Failed to log population metrics to mlflow", exc_info=e)

    def log_best_individual(
        self, genotype: np.ndarray, score: float, step: int, genome=None
    ) -> None:
        try:
            self.log_metrics({"best_score": float(score)}, step=step)

            if genome:
                # Save genome as artifact
                best_params = genome.genotype_to_param_arrays(genotype)
                df = pd.DataFrame(best_params)

                # Save to temporary file and log as artifact
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False
                ) as f:
                    df.to_csv(f.name, index=False)
                    mlflow.log_artifact(f.name, "genomes")
                    os.unlink(f.name)  # Clean up temp file

        except Exception as e:
            logging.warning("Failed to log best individual to mlflow", exc_info=e)

    def log_artifact(self, file_path: str, artifact_name: str) -> None:
        try:
            mlflow.log_artifact(file_path, artifact_name)
        except Exception as e:
            logging.warning(
                f"Failed to log artifact {artifact_name} to mlflow", exc_info=e
            )

    def finish(self) -> None:
        if self.run_id:
            mlflow.end_run()
            self.run_id = None


def create_tracker(tracker_type: str) -> ExperimentTracker:
    """Factory function to create experiment trackers."""
    if tracker_type == "wandb":
        return WandBTracker()
    elif tracker_type == "mlflow":
        return MLflowTracker()
    elif tracker_type == "none":
        return NoOpTracker()
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")


class NoOpTracker(ExperimentTracker):
    """No-operation tracker for when no tracking is desired."""

    def initialize(self, project_name: str, config: Dict[str, Any], **kwargs) -> None:
        logging.info("No experiment tracking enabled")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        pass

    def log_population_stats(
        self,
        generation: int,
        population_size: int,
        mean_fitness: float,
        best_fitness: float,
        **kwargs,
    ) -> None:
        pass

    def log_best_individual(
        self, generation: int, individual: Dict[str, Any], fitness: float, genome=None
    ) -> None:
        pass

    def log_artifact(self, file_path: str, artifact_name: str) -> None:
        pass

    def finish(self) -> None:
        pass
