from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import click
import yaml

from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.finetune import FineTuneBaselineConfiguration
from mergekit.evo.progress import iter_progress_events


@dataclass(frozen=True)
class CampaignJob:
    preset: str
    seed: int
    runner: str
    config_path: Path
    run_dir: Path
    command: List[str]
    resume: bool = False
    skip: bool = False


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_preset(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Campaign preset must be a YAML mapping: {config_path}")
    return payload


def _campaign_metadata(payload: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    metadata = payload.get("campaign")
    if not isinstance(metadata, dict):
        raise ValueError(f"Preset is missing campaign metadata: {config_path}")
    runner = metadata.get("runner")
    if runner not in {"evolve_ga", "finetune_baseline"}:
        raise ValueError(f"Unknown campaign runner {runner!r} in {config_path}")
    seeds = metadata.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"Preset must pin at least one campaign seed: {config_path}")
    return metadata


def validate_preset(config_path: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    payload = _read_preset(config_path)
    metadata = _campaign_metadata(payload, config_path)
    runner = metadata["runner"]
    if int(metadata.get("budget_fevals", 0)) != 192:
        raise ValueError(f"Preset must pin campaign.budget_fevals=192: {config_path}")
    if list(metadata.get("two_stage_limits") or []) != [2, 3, 6]:
        raise ValueError(
            f"Preset must pin campaign.two_stage_limits=[2, 3, 6]: {config_path}"
        )

    if runner == "evolve_ga":
        config_payload = dict(payload)
        config_payload.pop("campaign", None)
        config = EvolMergeConfiguration.model_validate(config_payload)
        if config.provenance != "warn":
            raise ValueError(f"Preset must set provenance: warn: {config_path}")
        if config.stop is None or config.stop.max_fevals != 192:
            raise ValueError(f"Preset must set stop.max_fevals: 192: {config_path}")
        observed_limits = [
            config.stage1_limit,
            config.stage2_top_k,
            config.stage2_limit,
        ]
        if observed_limits != [2, 3, 6] or config.limit != 12:
            raise ValueError(
                "Preset must use stage1_limit=2, stage2_top_k=3, "
                f"stage2_limit=6, and limit=12: {config_path}"
            )
    else:
        config = FineTuneBaselineConfiguration.model_validate(payload)
        if config.provenance != "warn":
            raise ValueError(f"Preset must set provenance: warn: {config_path}")
        if config.budget_steps != 192:
            raise ValueError(f"Preset must set budget_steps: 192: {config_path}")
        if config.search_limit != 6 or config.export_limit != 12:
            raise ValueError(
                f"Fine-tuning preset must use search_limit=6 and export_limit=12: {config_path}"
            )
    return payload, metadata


def _successful_run(run_dir: Path, runner: str) -> bool:
    if not run_dir.is_dir():
        return False
    success_message = (
        "run_finished" if runner == "evolve_ga" else "finetune_run_finished"
    )
    try:
        terminal_status = None
        for event in iter_progress_events(str(run_dir)):
            if event.get("msg") == success_message:
                terminal_status = event.get("status")
        return terminal_status == "success"
    except (FileNotFoundError, ValueError):
        return False


def _parse_seeds(value: Optional[str], defaults: Iterable[Any]) -> List[int]:
    raw_values = list(defaults) if value is None else value.split(",")
    seeds = []
    for raw in raw_values:
        seed = int(str(raw).strip())
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def build_job(
    preset: str,
    config_path: Path,
    output_root: Path,
    *,
    seed: int,
    python_executable: str = sys.executable,
) -> CampaignJob:
    _payload, metadata = validate_preset(config_path)
    runner = str(metadata["runner"])
    run_dir = output_root / preset / f"seed-{seed}"
    if _successful_run(run_dir, runner):
        return CampaignJob(
            preset=preset,
            seed=seed,
            runner=runner,
            config_path=config_path,
            run_dir=run_dir,
            command=[],
            skip=True,
        )

    if runner == "evolve_ga":
        random_search = metadata.get("random_search")
        state_path = run_dir / "ga_state.json"
        resume = state_path.is_file() and random_search is None
        command = [
            python_executable,
            "-m",
            "mergekit.scripts.evolve_ga",
            str(config_path),
            "--resume" if resume else "--storage-path",
            str(run_dir),
            "--random-seed",
            str(seed),
            "--strategy",
            str(metadata.get("strategy", "serial")),
            "--device",
            str(metadata.get("device", "cpu")),
            "--num-gpus",
            "0",
            "--no-vllm",
            "--no-merge-cuda",
            "--max-disk-gb-min",
            str(metadata.get("max_disk_gb_min", 5)),
            "--save-final-model",
        ]
        if random_search is not None:
            command.extend(["--random-search", str(int(random_search))])
        return CampaignJob(
            preset=preset,
            seed=seed,
            runner=runner,
            config_path=config_path,
            run_dir=run_dir,
            command=command,
            resume=resume,
        )

    command = [
        python_executable,
        "-m",
        "mergekit.scripts.finetune_baseline",
        "--config",
        str(config_path),
        "--output-dir",
        str(run_dir),
        "--seed",
        str(seed),
        "--device",
        str(metadata.get("device", "cpu")),
    ]
    return CampaignJob(
        preset=preset,
        seed=seed,
        runner=runner,
        config_path=config_path,
        run_dir=run_dir,
        command=command,
    )


@click.command("mergekit-run-campaign")
@click.argument("presets", nargs=-1, required=True)
@click.option(
    "--seeds",
    type=str,
    default=None,
    help="Comma-separated seed override; otherwise each preset's pinned seeds apply.",
)
@click.option(
    "--config-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--output-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--continue-on-error", is_flag=True, default=False)
def main(
    presets: tuple[str, ...],
    seeds: Optional[str],
    config_dir: Optional[Path],
    output_root: Optional[Path],
    dry_run: bool,
    continue_on_error: bool,
) -> None:
    root = repository_root()
    config_dir = (config_dir or root / "experiments" / "configs").resolve()
    output_root = (output_root or root / "workspace" / "thesis" / "campaigns").resolve()
    failures = []

    for preset in presets:
        config_path = config_dir / f"{preset}.yaml"
        if not config_path.is_file():
            raise click.ClickException(f"Unknown campaign preset: {preset}")
        try:
            _payload, metadata = validate_preset(config_path)
            selected_seeds = _parse_seeds(seeds, metadata["seeds"])
        except (TypeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

        for seed in selected_seeds:
            job = build_job(
                preset,
                config_path,
                output_root,
                seed=seed,
            )
            if job.skip:
                click.echo(f"SKIP {preset} seed={seed}: completed at {job.run_dir}")
                continue
            click.echo(
                f"{'RESUME' if job.resume else 'START'} {preset} seed={seed}: "
                f"{job.run_dir}"
            )
            click.echo(shlex.join(job.command))
            if dry_run:
                continue
            job.run_dir.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(job.command, cwd=root, check=True)
            except subprocess.CalledProcessError as exc:
                failures.append((preset, seed, int(exc.returncode)))
                if not continue_on_error:
                    raise click.ClickException(
                        f"Campaign run failed: {preset} seed={seed} "
                        f"(exit {exc.returncode})"
                    ) from exc
            except KeyboardInterrupt as exc:
                raise click.ClickException(
                    f"Campaign interrupted; rerun the same command to resume {job.run_dir}"
                ) from exc

    if failures:
        summary = ", ".join(
            f"{preset}/seed-{seed}:exit-{code}" for preset, seed, code in failures
        )
        raise click.ClickException(f"Campaign completed with failures: {summary}")


if __name__ == "__main__":
    main()
