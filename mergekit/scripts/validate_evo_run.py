"""Command-line validation for completed evolutionary experiment runs."""

from pathlib import Path

import click

from mergekit.evo.run_validation import EvoRunValidationError, validate_evo_run


@click.command("mergekit-validate-evo-run")
@click.argument(
    "run_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
)
@click.option("--require-baseline/--no-require-baseline", default=True)
def main(run_dir: Path, config_path: Path | None, require_baseline: bool) -> None:
    try:
        result = validate_evo_run(
            run_dir,
            config_path=config_path,
            require_baseline=require_baseline,
        )
    except EvoRunValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"VALID {result.run_dir} mode={result.mode} fevals={result.fevals} "
        f"candidates={result.candidate_rows} successful={result.successful_candidates} "
        f"methods={result.method_rows} invalid_genotypes={result.invalid_genotype_count}"
    )
    for warning in result.warnings:
        click.echo(f"WARNING: {warning}", err=True)


if __name__ == "__main__":
    main()
