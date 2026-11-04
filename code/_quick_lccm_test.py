#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python - <<'PY'
"""Quick test of the LC model build with the monkey patch."""
import sys
sys.path.insert(0, "/mnt/hpccs01/home/ahernz/test_runs_tours/code")

# Apply monkey patch first
import larch.model.latent_class as lcl
import numpy as np
from functools import wraps

_original_reflow = lcl.LatentClass.reflow_data_arrays

def patched_reflow(self):
    datatree = self.datatree
    if datatree is None:
        raise ValueError("missing datatree")
    classmodel = self.class_membership_model
    request = classmodel.required_data()
    request.pop("avail_any", None)
    for _kname, kmodel in self._models.items():
        if kmodel._model_subtype in ("class-membership", "latent-class"):
            continue
        kreq = kmodel.required_data()
        for k, v in kreq.items():
            if k not in request:
                request[k] = v
            else:
                if isinstance(request[k], dict):
                    request[k].update(v)
                elif isinstance(request[k], (list | tuple)):
                    request[k] = list(set(request[k]) | set(v))
                else:
                    if request[k] != v:
                        raise ValueError("incompatible requests")
    if isinstance(self.groupid, str):
        request["group_co"] = self.groupid
    from larch.data_arrays import prepare_data
    from larch.folding import fold_dataset
    dataset, self.dataflows = prepare_data(
        datasource=self.datatree,
        request=request,
        float_dtype=np.float32,
        cache_dir=datatree.cache_dir,
        flows=self.dataflows,
    )
    if isinstance(self.groupid, str):
        dataset = fold_dataset(dataset, "group")
    elif self.groupid is not None:
        dataset = fold_dataset(dataset, self.groupid)
    self.dataset = dataset
    for _kname, kmodel in self._models.items():
        if kmodel._model_subtype in ("class-membership", "latent-class"):
            continue
        # FIX: Initialize graph with proper alternative codes BEFORE setting dataset
        try:
            alt_codes = getattr(kmodel, 'choice_co_code', None)
            if alt_codes is None:
                alt_codes = list(range(1, 9))
            kmodel.initialize_graph(alternative_codes=alt_codes)
        except Exception:
            try:
                kmodel.initialize_graph(alternative_codes=list(range(1, 9)))
            except Exception:
                pass
        kmodel.dataset = self.dataset
        kmodel._data_arrays = kmodel.dataset.dc.to_arrays(
            kmodel.graph,
            float_dtype=kmodel.float_dtype,
        )
        kmodel._rebuild_fixed_arrays()
        kmodel._rebuild_work_arrays()
    classmodel_ids = [
        kid
        for (kid, km) in self._models.items()
        if km._model_subtype not in ("latent-class", "class-membership")
    ]
    classmodel_ids = list(classmodel_ids)
    from larch.folding import dissolve_zero_variance
    classmodel_data = lcl.dissolve_zero_variance(
        self.dataset.drop_dims(self.dataset.dc.ALTID).dc.set_altids(classmodel_ids),
        "ingroup",
    )
    try:
        classmodel.initialize_graph(alternative_codes=classmodel_ids)
    except Exception:
        try:
            classmodel.initialize_graph(alternative_codes=list(range(1, 9)))
        except Exception:
            pass
    classmodel.dataset = classmodel_data
    classmodel.reflow_data_arrays()

lcl.LatentClass.reflow_data_arrays = patched_reflow
print("Monkey patch applied")

# Now test with the actual pipeline's data
import stage5_destination_choice as s5
import pandas as pd

# Load the actual HHTS data that the pipeline uses
from data_connect import Synth_Connect
sc = Synth_Connect()
out = sc.process_survey(save=False)
tours = out['tours'].copy()
persons = out['persons'].copy()
landuse = sc.get_land()

print(f"Loaded {len(tours)} tours, {len(persons)} persons")

# Take a small sample for quick test
tours_sample = tours[tours['primary_purpose'] == 'work'].sample(n=200, random_state=42).copy()
persons_sample = persons[persons['person_id'].isin(tours_sample['person_id'])].copy()

print(f"Sample: {len(tours_sample)} work tours")

# Build LC model
from stage5_destination_choice import _build_mode_lccm
print("Building LC model...")
try:
    lc = _build_mode_lccm(tours_sample, 'work', n_classes=2)
    print("SUCCESS: LC model built!")
    print("  Classes:", lc.class_membership_model.dataset.dc.altids())
    for k, m in lc._models.items():
        if m._model_subtype not in ('latent-class', 'class-membership'):
            print(f"  Class {k}: graph={m.graph is not None}, dataset={m.dataset is not None}")
    print("SUCCESS: LC model builds without the graph bug!")
except Exception as e:
    import traceback
    print(f"FAILED: {e}")
    traceback.print_exc()
PY