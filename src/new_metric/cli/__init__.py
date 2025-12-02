"""
MADEX CLI - Command Line Interface for MADEX Analysis Tools

Provides CLI commands for running various MADEX analyses:
- sandbox: Extended scenario analysis
- narrative: Clinical narrative analysis
- bootstrap: Bootstrap statistical validation
- sensitivity: Parameter sensitivity analysis
- analyze-all: Run all analyses sequentially
"""

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="madex",
    help="MADEX (Mean Adjusted Exponent Error) Analysis CLI",
    add_completion=False,
)


@app.command()
def sandbox(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results (default: extended_outcome/)",
    ),
):
    """Run extended scenario analysis comparing Model A and Model B predictions."""
    from new_metric.cli.sandbox import run_sandbox_analysis

    run_sandbox_analysis(output)


@app.command()
def narrative(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results (default: extended_narrative/)",
    ),
):
    """Run clinical narrative analysis with bias pattern assessment."""
    from new_metric.cli.narrative import run_narrative_analysis

    run_narrative_analysis(output)


@app.command()
def bootstrap(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results (default: extended_bootstrap/)",
    ),
    samples: int = typer.Option(
        1000,
        "--samples",
        "-n",
        help="Number of bootstrap samples",
    ),
    confidence: int = typer.Option(
        95,
        "--confidence",
        "-c",
        help="Confidence interval percentage (e.g., 95 for 95%)",
    ),
):
    """Run bootstrap statistical validation analysis."""
    from new_metric.cli.bootstrap import run_bootstrap_analysis

    run_bootstrap_analysis(output, samples, confidence)


@app.command()
def sensitivity(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results (default: extended_sensitivity/)",
    ),
    a_range: Optional[str] = typer.Option(
        None,
        "--a-range",
        help="Range for parameter 'a' as min,max (default: 110,140)",
    ),
    b_range: Optional[str] = typer.Option(
        None,
        "--b-range",
        help="Range for parameter 'b' as min,max (default: 40,80)",
    ),
):
    """Run parameter sensitivity analysis across ranges."""
    from new_metric.cli.sensitivity import run_sensitivity_analysis

    run_sensitivity_analysis(output, a_range, b_range)


@app.command("analyze-all")
def analyze_all(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Base output directory for all results",
    ),
):
    """Run all four analyses sequentially (sandbox, narrative, bootstrap, sensitivity)."""
    from new_metric.cli.bootstrap import run_bootstrap_analysis
    from new_metric.cli.narrative import run_narrative_analysis
    from new_metric.cli.sandbox import run_sandbox_analysis
    from new_metric.cli.sensitivity import run_sensitivity_analysis

    typer.echo("=" * 70)
    typer.echo("MADEX COMPREHENSIVE ANALYSIS")
    typer.echo("Running all analyses sequentially...")
    typer.echo("=" * 70)
    typer.echo()

    # Determine output directories
    if output:
        sandbox_out = output / "sandbox"
        narrative_out = output / "narrative"
        bootstrap_out = output / "bootstrap"
        sensitivity_out = output / "sensitivity"
    else:
        sandbox_out = None
        narrative_out = None
        bootstrap_out = None
        sensitivity_out = None

    # Run sandbox analysis
    typer.echo("[1/4] Running Sandbox Analysis...")
    typer.echo("-" * 50)
    run_sandbox_analysis(sandbox_out)
    typer.echo()

    # Run narrative analysis
    typer.echo("[2/4] Running Narrative Analysis...")
    typer.echo("-" * 50)
    run_narrative_analysis(narrative_out)
    typer.echo()

    # Run bootstrap analysis
    typer.echo("[3/4] Running Bootstrap Analysis...")
    typer.echo("-" * 50)
    run_bootstrap_analysis(bootstrap_out)
    typer.echo()

    # Run sensitivity analysis
    typer.echo("[4/4] Running Sensitivity Analysis...")
    typer.echo("-" * 50)
    run_sensitivity_analysis(sensitivity_out)
    typer.echo()

    typer.echo("=" * 70)
    typer.echo("ALL ANALYSES COMPLETE")
    typer.echo("=" * 70)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
    ),
):
    """MADEX Analysis CLI - Tools for glucose prediction model evaluation."""
    if version:
        typer.echo("madex version 0.1.0")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
