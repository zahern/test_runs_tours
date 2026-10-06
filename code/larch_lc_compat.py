"""Compatibility shims so ``lx.LatentClass`` works with larch 6.0.46 in this stack.

Root causes found by instrumenting the reflow path (see the investigation in
`_test_lc_real_trace.py` / `_test_lc_instrument*.py`):

Defect 1 (data layer, avoided, not patched)
    ``NestingTree(root_id=0)`` absorbs an alternative whose code is 0:
    ``add_node(0)`` collides with the root node, so a choice set with
    alternatives [0, 1, 2] yields a graph with only two elementals [1, 2].
    Every downstream cascade then mismatches (3-alt choice array vs 2-alt
    graph -> ``ValueError: could not broadcast ...``).  Avoid 0-coded
    alternatives for LatentClass models; QLD destination zone ids never
    include 0, so only synthetic/test data is affected.

Defect 2 (patched here)
    ``LatentClass.reflow_data_arrays`` rebuilds the class membership dataset
    as ``drop_dims(ALTID) -> set_altids(class_keys) -> dissolve_zero_variance(
    "ingroup")``.  ``fold_dataset`` copies ALTID/GROUPID/INGROUP but not
    CASEID, and dissolving ``ingroup`` removes that dim, so ``n_cases`` falls
    through ``sizes[None]`` and ``sizes[GROUPID] * sizes[INGROUP]`` ->
    ``KeyError: None``.  For the dissolved class dataset the groups ARE the
    cases, so ``sizes[GROUPID]`` is the correct count.  Patch:
    ``_GenericFlow.n_cases`` falls back to GROUPID when INGROUP is dissolved.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def apply() -> bool:
    """Idempotently patch larch for LatentClass support. Returns success."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from larch.dataset import flow as _flow
    except Exception as exc:  # pragma: no cover
        logger.warning("larch_lc_compat: larch import failed (%r)", exc)
        return False

    cls = getattr(_flow, "_GenericFlow", None)
    if cls is None:
        logger.warning("larch_lc_compat: _GenericFlow not found")
        return False
    if getattr(cls.n_cases, "_qld_compat", False):
        _PATCHED = True
        return True

    def n_cases(self):
        try:
            return self._obj.sizes[self.CASEID]
        except KeyError:
            pass
        try:
            if self.GROUPID is not None and (
                    self.INGROUP is None
                    or self.INGROUP not in self._obj.sizes):
                # Folded and then dissolved (class-membership dataset):
                # the groups are the cases.
                return self._obj.sizes[self.GROUPID]
            return self._obj.sizes[self.GROUPID] * self._obj.sizes[self.INGROUP]
        except KeyError:
            logging.getLogger().error(
                f"missing {self.CASEID!r} among dims {self._obj.sizes}")
            raise

    n_cases._qld_compat = True
    cls.n_cases = property(n_cases)
    _PATCHED = True
    logger.info("larch_lc_compat: _GenericFlow.n_cases patched for "
                "dissolved-group class datasets")
    return True


def alt_ids_supported(alt_ids) -> bool:
    """LatentClass graphs cannot contain a 0-coded alternative (root id 0)."""
    import numpy as np
    return 0 not in set(int(a) for a in np.asarray(alt_ids).ravel())
