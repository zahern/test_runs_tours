# pipeline_logger.py
import time
import logging
import functools
import os

os.makedirs('saved_data', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('saved_data/pipeline.log', mode='w'),
    ]
)
logger = logging.getLogger('pipeline')


def timer(label: str = None):
    """Decorator that logs start, end and elapsed time of any function."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            name = label or fn.__name__
            logger.info(f"▶  {name} started")
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                logger.info(f"✓  {name} finished in {_fmt(elapsed)}")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.error(f"✗  {name} FAILED after {_fmt(elapsed)} — {e}")
                raise
        return wrapper
    return decorator


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


class PipelineTimer:
    """Context manager for timing named blocks."""
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self._t0 = time.perf_counter()
        logger.info(f"▶  {self.label} started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._t0
        if exc_type:
            logger.error(f"✗  {self.label} FAILED after {_fmt(elapsed)}")
        else:
            logger.info(f"✓  {self.label} finished in {_fmt(elapsed)}")
        return False


# ---------------------------------------------------------------------------
import pandas as pd
from data_connect import Synth_Connect, get_workstat, get_studying
from stage1_activity_frequency import run_stage1
from stage2_day_duration        import run_stage2
from stage3_activity_duration   import run_stage3
from stage4_schedule_placement  import run_stage4
from stage5_destination_choice  import run_stage5
from stage6_mode_choice         import run_stage6
from plot_activity_plans        import plot_all
from read_skims                 import list_omx_matrices
from ga_hypothesis_testing      import run_ga_all_stages


# ---------------------------------------------------------------------------
# HELPERS — decorated wrappers so every stage is individually timed
# ---------------------------------------------------------------------------

@timer("process_survey + landuse")
def load_data():
    sc      = Synth_Connect()
    out     = sc.process_survey(save=False)
    landuse = sc.get_land()
    return sc, out, landuse


@timer("load synthetic population")
def load_synth():
    synth = pd.read_csv('saved_data/zone_ids_sa2/order_40.csv')
    synth['WORKSTAT'] = synth.apply(get_workstat, axis=1)
    synth['STUDYING'] = synth.apply(get_studying, axis=1)
    return synth


@timer("Stage 1 — activity frequency")
def stage1(out, synth):
    synth_s1, models = run_stage1(out["tours"], out["persons"], synth)
    synth_s1.to_csv('saved_data/stage1_output.csv', index=False)
    return synth_s1, models


@timer("Stage 2 — day duration")
def stage2(out, synth_s1):
    synth_s2, models = run_stage2(out["tours"], out["persons"], synth_s1)
    synth_s2.to_csv('saved_data/stage2_output.csv', index=False)
    return synth_s2, models


@timer("Stage 3 — activity duration")
def stage3(out, synth_s2):
    activities, models = run_stage3(
        out["tours"], out["persons"], synth_s2, trips=out["trips"]
    )
    activities.to_csv('saved_data/stage3_output.csv', index=False)
    return activities, models


@timer("Stage 4 — schedule placement")
def stage4(out, activities):
    scheduled, models = run_stage4(
        out["trips"], out["tours"], out["persons"], activities
    )
    scheduled.to_csv('saved_data/stage4_output.csv', index=False)
    return scheduled, models


@timer("Stage 5 — destination choice")
def stage5(out, scheduled, landuse):
    list_omx_matrices('downcasted_matrices_base.omx')
    scheduled_dest, models = run_stage5(
        out["trips"], out["persons"], scheduled,
        skims='downcasted_matrices_base.omx',
        landuse=landuse,
    )
    scheduled_dest.to_csv('saved_data/stage5_output.csv', index=False)
    return scheduled_dest, models


@timer("Stage 6 — mode choice")
def stage6(out, scheduled_dest, landuse):
    plan, models = run_stage6(
        out["trips"], out["persons"], scheduled_dest, landuse
    )
    plan.to_csv('saved_data/stage6_output.csv', index=False)
    return plan, models


@timer("plot activity plans")
def plot(plan):
    plot_all(plan, n_sample=8, save_dir='saved_data/plots')


# ---------------------------------------------------------------------------
# GA decorated wrappers
# ---------------------------------------------------------------------------

@timer("GA — all stages")
def ga_all(out, landuse):
    results = run_ga_all_stages(
        tours         = out["tours"],
        trips         = out["trips"],
        persons       = out["persons"],
        skims         = 'downcasted_matrices_base.omx',
        landuse       = landuse,
        stages        = [1, 2, 3, 4, 5, 6],
        pop_size      = 30,
        n_generations = 40,
        save_path     = 'saved_data/ga_results.pkl',
    )
    results.summary()
    results.plot_all(save_path='saved_data/plots/ga_convergence.png')
    print("\nBest features per stage:")
    for stg, res in results.results.items():
        print(f"  Stage {stg}: {res.best_features}  fitness={res.best_fitness:.4f}")
    return results


@timer("GA — Stage 1")
def ga_stage1(out):
    from ga_hypothesis_testing import run_ga_stage
    res = run_ga_stage(stage=1, tours=out["tours"], persons=out["persons"],
                       pop_size=30, n_generations=40)
    res.plot_convergence(save_path='saved_data/plots/ga_stage1.png')
    print(f"  best: {res.best_features}  fitness={res.best_fitness:.4f}")
    print(res.feature_importance().to_string())
    return res


@timer("GA — Stage 2")
def ga_stage2(out):
    from ga_hypothesis_testing import run_ga_stage
    res = run_ga_stage(stage=2, tours=out["tours"], persons=out["persons"],
                       pop_size=30, n_generations=40)
    res.plot_convergence(save_path='saved_data/plots/ga_stage2.png')
    print(f"  best: {res.best_features}  fitness={res.best_fitness:.4f}")
    return res


@timer("GA — Stage 3")
def ga_stage3(out):
    from ga_hypothesis_testing import run_ga_stage
    res = run_ga_stage(stage=3, tours=out["tours"], persons=out["persons"],
                       pop_size=30, n_generations=40)
    res.plot_convergence(save_path='saved_data/plots/ga_stage3.png')
    print(f"  best: {res.best_features}  fitness={res.best_fitness:.4f}")
    return res


@timer("GA — Stage 4")
def ga_stage4(out):
    from ga_hypothesis_testing import run_ga_stage
    res = run_ga_stage(stage=4, trips=out["trips"], tours=out["tours"],
                       persons=out["persons"], pop_size=30, n_generations=40)
    res.plot_convergence(save_path='saved_data/plots/ga_stage4.png')
    print(f"  best: {res.best_features}  fitness={res.best_fitness:.4f}")
    return res


@timer("GA — Stage 5 (skim + landuse)")
def ga_stage5(out, landuse):
    from census_bridge import run_ga_extended
    res = run_ga_extended(
        stage=5, trips=out["trips"], persons=out["persons"],
        skims='downcasted_matrices_base.omx', landuse=landuse,
        purpose='work', pop_size=30, n_generations=40,
    )
    res.plot_convergence(save_path='saved_data/plots/ga_stage5.png')
    print(f"  skim:    {res.best_features.get('skim')}")
    print(f"  landuse: {res.best_features.get('landuse')}")
    return res


@timer("GA — Stage 6 (person + accessibility)")
def ga_stage6(out, landuse):
    from census_bridge import run_ga_extended
    res = run_ga_extended(
        stage=6, trips=out["trips"], persons=out["persons"],
        landuse=landuse, purpose='work', pop_size=30, n_generations=40,
    )
    res.plot_convergence(save_path='saved_data/plots/ga_stage6.png')
    print(f"  person: {res.best_features.get('person')}")
    print(f"  acc:    {res.best_features.get('accessibility')}")
    return res


# ---------------------------------------------------------------------------
# MAIN PIPELINES
# ---------------------------------------------------------------------------

def main():
    """Standard pipeline — no GA."""
    sc, out, landuse = load_data()
    synth            = load_synth()

    synth_s1, _      = stage1(out, synth)
    synth_s2, _      = stage2(out, synth_s1)
    activities, _    = stage3(out, synth_s2)
    scheduled, _     = stage4(out, activities)
    scheduled_dest,_ = stage5(out, scheduled, landuse)
    plan, _          = stage6(out, scheduled_dest, landuse)
    plot(plan)

    logger.info("Pipeline complete.")


def GA_main():
    """Run GA across all stages, then run pipeline."""
    sc, out, landuse = load_data()
    synth            = load_synth()

    ga_all(out, landuse)

    synth_s1, _      = stage1(out, synth)
    synth_s2, _      = stage2(out, synth_s1)
    activities, _    = stage3(out, synth_s2)
    scheduled, _     = stage4(out, activities)
    scheduled_dest,_ = stage5(out, scheduled, landuse)
    plan, _          = stage6(out, scheduled_dest, landuse)
    plot(plan)

    logger.info("GA pipeline complete.")


def GA_STAGED():
    """Run GA stage-by-stage, then run full pipeline."""
    sc, out, landuse = load_data()
    synth            = load_synth()

    ga_stage1(out)
    synth_s1, _   = stage1(out, synth)

    ga_stage2(out)
    synth_s2, _   = stage2(out, synth_s1)

    ga_stage3(out)
    activities, _ = stage3(out, synth_s2)

    ga_stage4(out)
    scheduled, _  = stage4(out, activities)

    ga_stage5(out, landuse)
    scheduled_dest, _ = stage5(out, scheduled, landuse)

    ga_stage6(out, landuse)
    plan, _ = stage6(out, scheduled_dest, landuse)

    plot(plan)
    logger.info("GA staged pipeline complete.")


if __name__ == "__main__":
    

    
    main()
    
    GA_main()
    
    GA_STAGED()
 