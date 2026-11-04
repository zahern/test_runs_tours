"""
Command-line interface for pyram.

Provides CLI commands for:
- pyram init: create a configuration file
- pyram validate: check configuration and inputs
- pyram-build-pop: build synthetic population
- pyram-run-pipeline: run full pipeline
- pyram-dta-feedback: run DTA feedback loop
- pyram estimate / predict: model estimation and application
- pyram-visualize: launch 3D visualizer
"""

import os
from pathlib import Path
from typing import Optional, List
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from pyram.config import Config, ConfigManager, create_default_config
from pyram.data import load_all_data, build_synthetic_population
from pyram.models import create_all_models
from pyram.pipeline import Pipeline

app = typer.Typer(
    name="pyram",
    help="PYRAM travel demand modeling pipeline",
    add_completion=False,
)

console = Console()


@app.command()
def init(
    output: str = typer.Option("config.yaml", "--output", "-o", help="Output config file path"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
):
    """Create a default configuration file."""
    path = Path(output)
    if path.exists() and not force:
        console.print(f"[yellow]Config file {path} already exists. Use --force to overwrite.[/yellow]")
        return
    
    config = create_default_config(str(path))
    console.print(f"[green]Created default config at {path}[/green]")
    console.print("Edit the config file to set your data paths and model options.")


@app.command()
def validate(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path"),
):
    """Validate configuration and check required files."""
    cfg = Config.from_yaml(config)
    manager = ConfigManager(cfg)
    warnings = manager.validate()
    
    if warnings:
        console.print("[yellow]Validation warnings:[/yellow]")
        for w in warnings:
            console.print(f"  - {w}")
    else:
        console.print("[green]Configuration is valid![/green]")
    
    # Show config summary
    table = Table(title="Configuration Summary")
    table.add_column("Section", style="cyan")
    table.add_column("Key Settings", style="white")
    
    table.add_row("Output Dir", cfg.data.output_dir)
    table.add_row("Zones", f"{cfg.zones.n_zones} zones ({cfg.zones.zone_id_col})")
    table.add_row("Population", f"Overture: {cfg.population.use_overture}, Region: {cfg.population.target_region}")
    table.add_row("Destination", f"Type: {cfg.destination.model_type}")
    table.add_row("Mode", f"Type: {cfg.mode.model_type}, MXL: {cfg.mode.mixed_logit}")
    table.add_row("Parking", f"Enabled: {cfg.parking.enable}")
    table.add_row("DTA Feedback", f"Enabled: {cfg.dta.enable}, Iters: {cfg.dta.iterations}")
    table.add_row("HPC", f"CPUs: {cfg.hpc.stage_ncpus}, Mem: {cfg.hpc.stage_mem}")
    
    console.print(table)


@app.command()
def env(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output .env file"),
):
    """Generate environment variable overrides for PBS/HPC submission."""
    cfg = Config.from_yaml(config)
    manager = ConfigManager(cfg)
    overrides = cfg.get_env_overrides()
    
    if output:
        with open(output, "w") as f:
            for k, v in overrides.items():
                f.write(f"{k}={v}\n")
        console.print(f"[green]Environment variables written to {output}[/green]")
    else:
        console.print("[bold]Environment variable overrides:[/bold]")
        for k, v in sorted(overrides.items()):
            console.print(f"  {k}={v}")


@app.command()
def build_pop(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path"),
    target_pop: Optional[int] = typer.Option(None, "--target", "-t", help="Target population size"),
    use_overture: Optional[bool] = typer.Option(None, "--overture/--no-overture", help="Use Overture Maps"),
    output: str = typer.Option("output/synthetic_population.csv", "--output", "-o", help="Output CSV path"),
):
    """Build synthetic population from HHTS + Overture."""
    cfg = Config.from_yaml(config)
    manager = ConfigManager(cfg)
    manager.create_output_dirs()

    if target_pop:
        console.print(f"[yellow]Target population override: {target_pop:,}[/yellow]")

    console.print("[bold]Loading data...[/bold]")
    data = load_all_data(cfg)
    
    console.print(f"[green]Loaded {len(data['persons']):,} HHTS persons[/green]")
    console.print(f"[green]Loaded {len(data['trips']):,} HHTS trips[/green]")

    use_ov = cfg.population.use_overture if use_overture is None else bool(use_overture)
    target = int(target_pop) if target_pop else int(
        len(data["persons"]) * cfg.population.sample_fraction
    )
    target = max(target, 1)

    overture_gdf = None
    if use_ov:
        overture_path = Path(cfg.data.output_dir) / "overture_population.parquet"
        if overture_path.exists():
            import geopandas as gpd

            overture_gdf = gpd.read_parquet(overture_path)
            console.print(f"  Overture features: {len(overture_gdf):,}")
        else:
            console.print(
                "[yellow]  No overture_population.parquet found - "
                "expanding the HHTS sample only[/yellow]"
            )

    synth = build_synthetic_population(
        hhts_persons=data["persons"],
        hhts_households=data.get("households", pd.DataFrame()),
        zone_crosswalk=data["crosswalk"],
        landuse=data["landuse"],
        target_population=target,
        seed=cfg.seed,
        use_overture=use_ov,
        overture_gdf=overture_gdf,
        zone_boundaries=None,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synth.to_csv(output_path, index=False)

    console.print(f"[green]Generated {len(synth):,} synthetic persons[/green]")
    console.print(f"[green]Written to {output_path}[/green]")


@app.command()
def run(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path"),
    stages: Optional[List[int]] = typer.Option(None, "--stage", "-s", help="Specific stages to run (1-7)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print execution plan without running"),
):
    """Run the full pyram pipeline."""
    cfg = Config.from_yaml(config)
    manager = ConfigManager(cfg)
    manager.setup_environment()
    manager.create_output_dirs()
    
    if dry_run:
        console.print("[bold]DRY RUN - Execution plan:[/bold]")
        console.print("  1. Load data (HHTS, skims, landuse)")
        console.print("  2. Build synthetic population")
        console.print("  3. Stage 1: CDAP (household coordination)")
        console.print("  4. Stage 2: Timing/duration models")
        console.print("  5. Stage 3: Destination choice")
        console.print("  6. Stage 4: Mode choice")
        console.print("  7. Stage 5: Trip chain building")
        console.print("  8. Stage 6: Parking choice")
        console.print("  9. Stage 7: DTA feedback loop")
        console.print(f"\nOutput: {cfg.data.output_dir}")
        console.print(f"Env overrides: {len(cfg.get_env_overrides())} variables")
        return
    
    console.print("[bold]Running pyram pipeline...[/bold]")
    selected = [int(stage) for stage in stages] if stages else None

    pipeline = Pipeline(cfg)
    result = pipeline.run(stages=selected)

    table = Table(title="Pipeline stages")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="white")
    requested = selected or list(range(1, 8))
    for stage in result.stages_completed:
        table.add_row(stage, "[green]completed[/green]")
    missing = [s for s in requested if f"stage{s}" not in " ".join(result.stages_completed)]
    for stage in missing:
        table.add_row(f"stage{stage}", "[yellow]not run[/yellow]")
    console.print(table)

    if not result.success:
        for error in result.errors:
            console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Pipeline complete. Outputs in {cfg.data.output_dir}[/green]")


@app.command()
def dta_feedback(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path"),
    iterations: int = typer.Option(3, "--iters", "-i", help="Number of DTA iterations"),
    engine: str = typer.Option("matsim", "--engine", "-e", help="DTA engine (matsim/sumo/dynust)"),
    input_plan: str = typer.Option("output/stage7_output.csv", "--input", help="Input plan CSV"),
    output_dir: str = typer.Option("output/dta_feedback", "--output", "-o", help="Output directory"),
):
    """Run DTA feedback loop (demand <-> DTA)."""
    cfg = Config.from_yaml(config)
    cfg.dta.enable = True
    cfg.dta.iterations = iterations
    cfg.dta.engine = engine
    
    console.print(f"[bold]Running DTA feedback loop[/bold]")
    console.print(f"  Engine: {engine}")
    console.print(f"  Iterations: {iterations}")
    console.print(f"  Input plan: {input_plan}")
    console.print(f"  Output: {output_dir}")

    plan_path = Path(input_plan)
    if not plan_path.exists():
        console.print(f"[red]Plan CSV not found: {plan_path}[/red]")
        console.print("Run the pipeline first to produce a trip plan.")
        raise typer.Exit(code=1)

    from pyram.dta import DTAFeedbackLoop

    plan = pd.read_csv(plan_path)
    loop = DTAFeedbackLoop(config=cfg.dta, output_dir=output_dir)
    outcome = loop.run(plan)

    if outcome.status not in ("completed", "max_iterations"):
        console.print(f"[yellow]DTA feedback {outcome.status}: {outcome.message}[/yellow]")
        raise typer.Exit(code=2)

    final_path = Path(output_dir) / "final_plan.csv"
    if outcome.plan is not None and len(outcome.plan):
        outcome.plan.to_csv(final_path, index=False)

    console.print(
        f"[green]{engine}: {outcome.iterations} iteration(s), "
        f"{'converged' if outcome.converged else 'not converged'}[/green]"
    )
    for row in outcome.history:
        console.print(f"  iteration {int(row['iteration'])}: skim RMSE {row['skim_rmse']:.5f}")
    if final_path.exists():
        console.print(f"[green]Final plan: {final_path}[/green]")


@app.command()
def estimate(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path"),
    model: str = typer.Option("all", "--model", "-m", help="Model to estimate (destination/mode/parking/trip_dest/all)"),
):
    """Estimate discrete choice models from HHTS data."""
    cfg = Config.from_yaml(config)
    manager = ConfigManager(cfg)
    manager.setup_environment()
    manager.create_output_dirs()

    models_dir = Path(cfg.data.output_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[bold]Loading data...[/bold]")
    data = load_all_data(cfg)
    
    console.print("[bold]Creating models...[/bold]")
    models = create_all_models(cfg)
    
    if model in ("all", "destination"):
        console.print("[cyan]Estimating destination choice...[/cyan]")
        models["destination"].estimate(
            data["trips"], data["persons"], data["landuse"], data["skim_lookup"]
        )
        models["destination"].save(models_dir / "destination_model.pkl")
        console.print("[green]Destination model saved[/green]")
    
    if model in ("all", "mode"):
        console.print("[cyan]Estimating mode choice...[/cyan]")
        models["mode"].estimate(
            data["trips"], data["persons"], data["landuse"], data["skim_lookup"]
        )
        models["mode"].save(models_dir / "mode_model.pkl")
        console.print("[green]Mode model saved[/green]")
    
    if model in ("all", "parking"):
        console.print("[cyan]Estimating parking choice...[/cyan]")
        models["parking"].estimate(
            data["trips"], data["persons"], data["location_table"], data["landuse"], data["skim_lookup"]
        )
        models["parking"].save(models_dir / "parking_model.pkl")
        console.print("[green]Parking model saved[/green]")
    
    if model in ("all", "trip_dest"):
        console.print("[cyan]Estimating trip-level destination...[/cyan]")
        chains_path = Path(cfg.data.output_dir) / "stage5_trip_chains.csv"
        if not chains_path.exists():
            console.print(
                "[yellow]No stage5_trip_chains.csv found - run stages 1-5 first; "
                "trip-level destination estimation skipped[/yellow]"
            )
        else:
            chains = pd.read_csv(chains_path)
            stop_trips = (
                chains[chains["leg"] == "outbound"].copy()
                if "leg" in chains.columns
                else chains.copy()
            )
            stop_trips = stop_trips.rename(
                columns={"OTAZ": "origin_taz", "DTAZ": "dest_taz"}
            )
            stop_trips["anchor_down_taz"] = stop_trips.get(
                "home_zone", stop_trips["dest_taz"]
            )
            models["trip_destination"].estimate(
                stop_trips, data["landuse"], data["skim_lookup"]
            )
            models["trip_destination"].save(
                models_dir / "trip_destination_model.pkl"
            )
            console.print("[green]Trip destination model saved[/green]")
    
    console.print("[green]Estimation complete![/green]")


@app.command()
def predict(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path"),
    input_plan: str = typer.Option("output/synthetic_population.csv", "--input", "-i", help="Input population/plan CSV"),
    output: str = typer.Option("output/predicted_plan.csv", "--output", "-o", help="Output predicted plan CSV"),
):
    """Run prediction on synthetic population."""
    cfg = Config.from_yaml(config)
    manager = ConfigManager(cfg)
    manager.setup_environment()
    manager.create_output_dirs()
    
    console.print("[bold]Loading data and models...[/bold]")
    data = load_all_data(cfg)

    population_path = Path(input_plan)
    if not population_path.exists():
        console.print(f"[red]Input population not found: {population_path}[/red]")
        raise typer.Exit(code=1)
    population = pd.read_csv(population_path)

    models = create_all_models(cfg)
    loaded = 0
    for name in ("destination", "mode", "parking", "trip_destination"):
        model_path = Path(cfg.data.output_dir) / "models" / f"{name}_model.pkl"
        if not model_path.exists():
            continue
        try:
            models[name] = type(models[name]).load(model_path)
            loaded += 1
            console.print(f"  Loaded {name} model")
        except Exception as exc:  # noqa: BLE001 - report and keep other models
            console.print(f"[yellow]  Could not load {name} model: {exc}[/yellow]")

    if loaded == 0:
        console.print(
            f"[red]No estimated models in {Path(cfg.data.output_dir) / 'models'}; "
            "run `pyram estimate` first[/red]"
        )
        raise typer.Exit(code=1)

    pipeline = Pipeline(cfg)
    pipeline.data = data
    pipeline.data["synthetic_population"] = population
    pipeline.models = models

    try:
        pipeline._run_stage1_cdap()
        pipeline._run_stage2_timing()
        pipeline._run_stage5_trip_chain()
    except Exception as exc:  # noqa: BLE001 - surface as a CLI failure
        console.print(f"[red]Prediction failed: {exc}[/red]")
        raise typer.Exit(code=1)

    chains = pipeline.data.get("trip_chains")
    if chains is None:
        console.print("[red]Prediction produced no trip chains[/red]")
        raise typer.Exit(code=1)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chains.to_csv(output_path, index=False)

    console.print(f"[green]Wrote {len(chains):,} predicted trips to {output_path}[/green]")


@app.command()
def visualize(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config file path"),
    data_dir: str = typer.Option("output", "--data-dir", "-d", help="Pipeline output directory"),
    port: int = typer.Option(8080, "--port", "-p", help="Server port"),
    host: str = typer.Option("0.0.0.0", "--host", help="Server host"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """Launch the PYRAM 3D visualizer web server."""
    cfg = Config.from_yaml(config)
    
    console.print("[bold]Starting PYRAM 3D Visualizer...[/bold]")
    console.print(f"  Data directory: {data_dir}")
    console.print(f"  Server: http://{host}:{port}")
    
    try:
        from pyram.visualizer import VisualizerServer
        server = VisualizerServer(
            data_dir=data_dir,
            port=port,
            host=host,
            debug=debug,
        )
        server.run()
    except ImportError as e:
        console.print(f"[red]Visualizer dependencies not available: {e}[/red]")
        console.print("Install with: pip install pyram[visualizer]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Visualizer stopped[/yellow]")


if __name__ == "__main__":
    app()
