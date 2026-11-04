#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python - <<'PY'
"""Test that the LCCM build now works after the monkey patch."""
import os
import numpy as np
import pandas as pd
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

import larch.model.latent_class as lcl
lcl.LatentClass.reflow_data_arrays = patched_reflow
print("Monkey patch applied")

# Now test building an LC model
import stage5_destination_choice as s5
import pandas as pd

# Create minimal test data
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'ptour_id': [str(i) for i in range(n)],
    'person_id': [str(i) for i in range(n)],
    'tour_id': np.arange(n),
    'OTAZ': np.random.randint(1, 372, n),
    'DTAZ': np.random.randint(1, 372, n),
    'tour_mode': np.random.choice([1,2,3,4,5], n),
    'main_purpose': np.random.choice(['work', 'edu', 'other'], n),
    'outbound_stops': np.random.randint(0, 3, n),
    'n_shopping_out': np.random.randint(0, 2, n),
    'joint_travel': np.random.randint(0, 2, n),
    'incp': np.random.randint(1, 20, n),
    'indp': np.random.randint(1, 20, n),
    'vehrd': np.random.randint(0, 3, n),
    'STUDYING': np.random.randint(0, 2, n),
    'sexp': np.random.randint(1, 3, n),
    'CARLICENCE': np.random.randint(0, 2, n),
    'occp_group': np.random.randint(0, 5, n),
    'bikes': np.random.randint(0, 3, n),
    'nprd': np.random.randint(1, 4, n),
    'walk_avail': 1,
    'bike_avail': 1,
    'STARTIME': np.random.randint(6*60, 22*60, n),
    'ENDTIME': np.random.randint(6*60, 22*60, n),
    'predicted_duration_min': np.random.randint(10, 120, n),
    'planning_order': np.random.randint(1, 5, n),
    'activity_seq': np.random.randint(1, 5, n),
    'schedule_seq': np.random.randint(1, 5, n),
    'schedule_n_activities': np.random.randint(1, 5, n),
    'sched_prev_end': np.random.randint(6*60, 22*60, n),
    'sched_next_start': np.random.randint(6*60, 22*60, n),
    'sched_prev_taz': np.random.randint(1, 372, n),
    'sched_next_taz': np.random.randint(1, 372, n),
    'start_of_day': np.random.randint(6*60, 9*60, n),
    'end_of_day': np.random.randint(17*60, 23*60, n),
})

print("DataFrame created:", df.shape)

# Apply monkey patch
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

# Now try to build the LC model
from stage5_destination_choice import _build_mode_lccm
import pandas as pd

# Create minimal test data
tours = df.copy()
persons = df[['person_id', 'vehrd', 'CARLICENCE', 'bikes', 'incp', 'age5p', 'sexp', 'nprd']].drop_duplicates('person_id')
landuse = pd.DataFrame({'zone_id': range(1, 372), 'size_term': np.random.randn(371)})

print("Building LC model...")
try:
    lc = _build_mode_lccm(df, 'work', n_classes=2)
    print("LC model built successfully!")
    print("Classes:", lc.class_membership_model.dataset.dc.altids())
    for k, m in lc._models.items():
        if m._model_subtype not in ('latent-class', 'class-membership'):
            print(f"  Class {k}: graph={m.graph is not None}, dataset={m.dataset is not None}")
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
PY