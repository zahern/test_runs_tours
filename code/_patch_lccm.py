#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python - <<'PY'
import larch.model.latent_class as lcl
import larch.model.basemodel as bm
import numpy as np
from functools import wraps

# Original reflow_data_arrays
_original_reflow = lcl.LatentClass.reflow_data_arrays

@wraps(lcl.LatentClass.reflow_data_arrays)
def patched_reflow(self):
    """Patched reflow_data_arrays that initializes graph before setting dataset."""
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
    from .data_arrays import prepare_data
    dataset, self.dataflows = prepare_data(
        datasource=datatree,
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
            # Get alternative codes from the model's choice set
            alt_codes = getattr(kmodel, 'choice_co_code', None)
            if alt_codes is None and hasattr(kmodel, 'choice_co_code'):
                alt_codes = kmodel.choice_co_code
            if alt_codes is None:
                # Fallback: use MODE_MAP values
                alt_codes = list(range(1, 9))  # 1=DA, 2=SR, 3=Walk, 4=Bike, 5=Transit, 6=PT_Walk, 7=PT_Drive, 8=PT_Bike
            kmodel.initialize_graph(alternative_codes=alt_codes)
        except Exception as e:
            # If graph initialization fails, try with default codes
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
    classmodel_data = dissolve_zero_variance(
        self.dataset.drop_dims(self.dataset.dc.ALTID).dc.set_altids(classmodel_ids),
        "ingroup",
    )
    # FIX: Initialize classmodel graph before setting dataset
    try:
        classmodel_ids_list = list(classmodel_ids)
        classmodel.initialize_graph(alternative_codes=classmodel_ids_list)
    except Exception:
        try:
            classmodel.initialize_graph(alternative_codes=list(range(1, 9)))
        except Exception:
            pass
    classmodel.dataset = classmodel_data
    classmodel.reflow_data_arrays()

# Apply the monkey patch
lcl.LatentClass.reflow_data_arrays = patched_reflow
print("Monkey patch applied to LatentClass.reflow_data_arrays")
PY