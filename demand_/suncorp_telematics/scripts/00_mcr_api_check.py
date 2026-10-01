"""Offline compat check for the current PyPI metacountregressor API."""
import inspect
from importlib.metadata import PackageNotFoundError, version

import metacountregressor as mcr

try:
    distribution_version = version("metacountregressor")
except PackageNotFoundError as exc:
    raise SystemExit("metacountregressor must be installed from PyPI") from exc
version_tuple = tuple(int(part) for part in distribution_version.split(".")[:3])
if version_tuple < (1, 0, 159):
    raise SystemExit(
        f"metacountregressor {distribution_version} is older than PyPI floor 1.0.159"
    )
print("distribution version:", distribution_version)
print("module version:", getattr(mcr, "__version__", "?"))
from metacountregressor import ExperimentBuilder, ModelConstraints

sig = inspect.signature(ExperimentBuilder.build_evaluator)
print("build_evaluator params:", sorted(sig.parameters))
print("engine param supported:", "engine" in sig.parameters)
for meth in ("run", "fit_manual_model", "make_manual_spec", "describe"):
    print(f"has {meth}:", hasattr(ExperimentBuilder, meth))
for meth in ("force_include", "no_zi", "no_random", "allow_random",
             "force_fixed", "exclude", "mutual_exclusion"):
    print(f"constraints.{meth}:", hasattr(ModelConstraints, meth))
run_sig = inspect.signature(ExperimentBuilder.run)
print("run params:", sorted(run_sig.parameters))
fit_sig = inspect.signature(ExperimentBuilder.fit_manual_model)
print("fit_manual_model params:", sorted(fit_sig.parameters))
