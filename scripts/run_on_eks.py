"""Compatibility wrapper for running mergekit-eks from the repository root."""

from mergekit.scripts.run_on_eks import cli


if __name__ == "__main__":
    cli(prog_name="mergekit-eks")
