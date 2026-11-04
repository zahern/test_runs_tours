#!/usr/bin/env python
"""Focused tests for Stage 5's mode-conditioned destination choice set."""
import ast
import logging
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


def _load_mask_helper():
    source_path = Path(__file__).with_name('stage5_destination_choice.py')
    tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == '_mode_conditioned_destination_mask'
    )
    namespace = {
        'np': np,
        'pd': pd,
        'os': os,
        'MODE_MAP': {'Walk': 3, 'Bike': 4},
        'MODE_CONDITIONAL_WALK_MAX_KM': 6.0,
        'MODE_CONDITIONAL_BIKE_MAX_KM': 25.0,
        'logger': logging.getLogger('stage5-mode-conditioned-test'),
    }
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(source_path), 'exec'),
        namespace,
    )
    return namespace['_mode_conditioned_destination_mask']


def _load_transit_helpers():
    source_path = Path(__file__).with_name('stage5_destination_choice.py')
    tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    helper_names = {
        '_transit_access_code', '_transit_submode_code',
        '_transit_submode_skim_values', '_transit_submode_features',
        '_fit_transit_submode', '_predict_transit_submode',
        '_predict_transit_access', '_get_tt', '_skim_minutes',
    }
    namespace = {
        'np': np,
        'pd': pd,
        'os': os,
        'logger': logging.getLogger('stage5-transit-test'),
        'TRANSIT_SUBMODE_MAP': {'Bus': 1, 'Train': 2},
        'TRANSIT_SUBMODE_NAMES': {1: 'Bus', 2: 'Train'},
        'TRANSIT_ACCESS_MAP': {'Walk': 1, 'Drive': 2, 'Bike': 3},
        'TRANSIT_ACCESS_NAMES': {1: 'Walk', 2: 'Drive', 3: 'Bike'},
        # The real module reads STAGE5_PT_MODES at import; the legacy binary
        # walk/drive access path (exercised by these tests) runs when it is
        # False. PT_BIKE_FROM_WALK is referenced only in the PT-on branch.
        'STAGE5_PT_MODES': False,
        'PT_BIKE_FROM_WALK': 2.4,
        'MODE_TT_COL': {5: 'WLK_LOC_WLK_TOTIVT__AM'},
        'MODE_TT_CAP': {5: 150.0},
        'MODE_TT_DEFAULT': {5: 25.0},
        'WALK_MAX_KM': 10.0,
        'BIKE_MAX_KM': 25.0,
    }
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), 'exec'), namespace)
    return namespace


def _load_two_stage_helper():
    source_path = Path(__file__).with_name('stage5_destination_choice.py')
    tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == '_compose_two_stage_probs'
    )
    namespace = {'np': np}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(source_path), 'exec'),
        namespace,
    )
    return namespace['_compose_two_stage_probs']


def _load_two_stage_prediction_helper():
    source_path = Path(__file__).with_name('stage5_destination_choice.py')
    tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    helper_names = {
        '_compose_two_stage_probs', '_stable_logsumexp',
        '_softmax_from_utilities', '_two_stage_prediction_probs',
    }
    namespace = {
        'np': np,
        'pd': pd,
        'logger': logging.getLogger('stage5-two-stage-prediction-test'),
    }
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), 'exec'), namespace)
    return namespace


def _load_sa1_refinement_helpers():
    source_path = Path(__file__).with_name('stage5_destination_choice.py')
    tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    helper_names = {
        '_load_sa1_refinement_data', '_sa1_refinement_weights',
        '_refine_sa1_destinations',
    }
    namespace = {
        'np': np,
        'pd': pd,
        'os': os,
        '__file__': str(source_path),
        'logger': logging.getLogger('stage5-sa1-refinement-test'),
    }
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), 'exec'), namespace)
    return namespace


class ModeConditionedDestinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask_helper = staticmethod(_load_mask_helper())
        cls.transit_helpers = _load_transit_helpers()
        cls.two_stage_helper = staticmethod(_load_two_stage_helper())
        cls.two_stage_prediction_helpers = _load_two_stage_prediction_helper()
        cls.sa1_refinement_helpers = _load_sa1_refinement_helpers()

    def test_sa1_refinement_stays_inside_selected_parent_sa2(self):
        helpers = self.sa1_refinement_helpers
        with tempfile.TemporaryDirectory() as temp_dir:
            lookup_path = Path(temp_dir) / 'sa1.csv'
            feature_path = Path(temp_dir) / 'sa1_features.csv'
            pd.DataFrame({
                'SA1_CODE21': [10101000101, 10101000201, 10102000101],
                'SA2_CODE21': [112011236, 112011236, 112011237],
            }).to_csv(lookup_path, index=False)
            pd.DataFrame({
                'SA1_CODE21': [10101000101, 10101000201, 10102000101],
                'n_retail': [10, 1, 100],
            }).to_csv(feature_path, index=False)

            with patch.dict(os.environ, {
                'STAGE5_SA1_LOOKUP': str(lookup_path),
                'STAGE5_SA1_LANDUSE': str(feature_path),
            }, clear=False):
                candidates = helpers['_load_sa1_refinement_data']()
                self.assertEqual(
                    set(candidates[1]['SA1_CODE21']),
                    {10101000101, 10101000201},
                )
                self.assertEqual(set(candidates[2]['SA1_CODE21']), {10102000101})

                weights = helpers['_sa1_refinement_weights'](
                    candidates[1], 'shopping'
                )
                self.assertAlmostEqual(float(weights.sum()), 1.0)

                activities = pd.DataFrame({
                    'DTAZ': [1, 1, 2],
                    'purpose': ['shopping', 'shopping', 'shopping'],
                })
                refined = helpers['_refine_sa1_destinations'](
                    activities, np.random.default_rng(4)
                )

        self.assertTrue(set(refined.loc[refined['DTAZ'] == 1, 'DTAZ_SA1']).issubset(
            {10101000101, 10101000201}
        ))
        self.assertEqual(int(refined.loc[refined['DTAZ'] == 2, 'DTAZ_SA1'].iloc[0]), 10102000101)
        np.testing.assert_array_equal(refined['DTAZ'].to_numpy(), [1, 1, 2])

    def test_missing_sa1_lookup_does_not_change_parent_destination(self):
        helpers = self.sa1_refinement_helpers
        activities = pd.DataFrame({'DTAZ': [7], 'purpose': ['social']})
        with patch.dict(os.environ, {
            'STAGE5_SA1_LOOKUP': os.path.join(
                tempfile.gettempdir(), 'stage5-missing-sa1.csv'
            ),
        }, clear=False):
            refined = helpers['_refine_sa1_destinations'](
                activities, np.random.default_rng(2)
            )

        self.assertNotIn('DTAZ_SA1', refined.columns)
        self.assertEqual(int(refined['DTAZ'].iloc[0]), 7)

    def test_two_stage_probabilities_compose_within_sa3(self):
        compose = self.two_stage_helper
        upper = np.array([
            [0.25, 0.75],
            [0.60, 0.40],
        ])
        lower = {
            10: np.array([[0.80, 0.20], [0.25, 0.75]]),
            20: np.array([[0.10, 0.30, 0.60], [0.50, 0.25, 0.25]]),
        }
        zones = [101, 102, 201, 202, 203]
        zone_to_sa3 = {101: 10, 102: 10, 201: 20, 202: 20, 203: 20}

        probs = compose(upper, lower, [10, 20], zones, zone_to_sa3)

        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        np.testing.assert_allclose(probs[0], [0.20, 0.05, 0.075, 0.225, 0.45])
        np.testing.assert_allclose(probs[1], [0.15, 0.45, 0.20, 0.10, 0.10])
        np.testing.assert_allclose(probs[:, :2].sum(axis=1), upper[:, 0])
        np.testing.assert_allclose(probs[:, 2:].sum(axis=1), upper[:, 1])

    def test_two_stage_prediction_uses_each_sa3_lower_subset(self):
        helpers = self.two_stage_prediction_helpers
        prediction = helpers['_two_stage_prediction_probs']
        helpers['_coef_dict_from_model'] = lambda model: model.coef

        upper_model = types.SimpleNamespace(
            coef={'c_sa3_size': 0.0, 'c_sa3_logsum': 1.0},
            _segment=None,
        )
        lower_models = {
            10: types.SimpleNamespace(coef={}, _segment=None),
            20: types.SimpleNamespace(coef={}, _segment=None),
        }
        utilities = {
            id(lower_models[10]): np.array([[2.0, 1.0]]),
            id(lower_models[20]): np.array([[0.0, 3.0]]),
        }
        calls = []

        def fake_utility(model, _dt, _ls, zones, *_args, **_kwargs):
            calls.append(tuple(zones))
            return utilities[id(model)]

        helpers['_dest_utility_numpy'] = fake_utility
        sa3_features = pd.DataFrame({'size_term': [0.0, 0.0]}, index=[10, 20])
        probs = prediction(
            {
                'dest_model': upper_model,
                '_two_stage_lower': lower_models,
                '_two_stage_zone_map': {101: 10, 102: 10, 201: 20, 202: 20},
                '_two_stage_sa3_features': sa3_features,
                '_two_stage_sa3_list': [10, 20],
            },
            object(), np.zeros((1, 1)), [101, 102, 201, 202], pd.DataFrame(), None,
        )

        self.assertEqual(calls, [(101, 102), (201, 202)])
        expected_upper = np.exp([
            np.log(np.exp(2.0) + np.exp(1.0)),
            np.log(np.exp(0.0) + np.exp(3.0)),
        ])
        expected_upper /= expected_upper.sum()
        expected = np.array([
            expected_upper[0] * np.exp([2.0, 1.0]) / (np.exp(2.0) + np.exp(1.0)),
            expected_upper[1] * np.exp([0.0, 3.0]) / (np.exp(0.0) + np.exp(3.0)),
        ]).ravel()
        np.testing.assert_allclose(probs[0], expected)

    def test_walk_is_local_but_motorized_keeps_full_zone_universe(self):
        rows = pd.DataFrame({'OTAZ': [1, 1], 'HOME_LVL4': [0, 0]})
        skim_lookup = {
            (1, 1): {'DISTWALK': 0.5, 'DISTBIKE': 0.5},
            (1, 2): {'DISTWALK': 2.0, 'DISTBIKE': 2.0},
            (1, 3): {'DISTWALK': 8.0, 'DISTBIKE': 8.0},
        }

        mask = self.mask_helper(
            rows, [1, 2, 3], [3, 1], skim_lookup=skim_lookup
        )

        self.assertEqual(mask.tolist(), [[True, True, False], [True, True, True]])

    def test_bike_uses_its_own_distance_threshold(self):
        rows = pd.DataFrame({'OTAZ': [1], 'HOME_LVL4': [0]})
        skim_lookup = {
            (1, 1): {'DISTBIKE': 0.5},
            (1, 2): {'DISTBIKE': 20.0},
            (1, 3): {'DISTBIKE': 30.0},
        }

        mask = self.mask_helper(
            rows, [1, 2, 3], [4], skim_lookup=skim_lookup
        )

        self.assertEqual(mask.tolist(), [[True, True, False]])

    def test_fine_origin_skim_overrides_sa2_fallback(self):
        fine = {
            'DISTWALK': np.array([[0.5, 7.0]], dtype=np.float32),
            'DISTBIKE': np.array([[0.5, 7.0]], dtype=np.float32),
            'lvl4_to_row': {99: 0},
            'sa2_to_col': {1: 0, 2: 1},
            'order_to_col': {},
        }
        fake_read_skims = types.ModuleType('read_skims')
        fake_read_skims.LVL4_WALK_BIKE_H5 = 'fine.h5'
        fake_read_skims.SA1_WALK_BIKE_H5 = 'sa1.h5'
        fake_read_skims.load_lvl4_walk_skims = lambda: fine
        fake_read_skims.load_sa1_walk_skims = lambda: None
        rows = pd.DataFrame({'OTAZ': [1], 'HOME_LVL4': [99]})
        skim_lookup = {
            (1, 1): {'DISTWALK': 0.5},
            (1, 2): {'DISTWALK': 1.0},
        }

        with patch.dict(sys.modules, {'read_skims': fake_read_skims}):
            with patch('os.path.exists', return_value=True):
                mask = self.mask_helper(
                    rows, [1, 2], [3], skim_lookup=skim_lookup
                )

        self.assertEqual(mask.tolist(), [[True, False]])

    def test_no_distance_source_preserves_full_universe(self):
        rows = pd.DataFrame({'OTAZ': [1], 'HOME_LVL4': [0]})

        mask = self.mask_helper(rows, [1, 2, 3], [3], skim_lookup={})

        self.assertEqual(mask.tolist(), [[True, True, True]])

    def test_transit_labels_normalize_to_bus_and_train(self):
        code = self.transit_helpers['_transit_submode_code']

        self.assertEqual(code('Public Bus'), 1)
        self.assertEqual(code('Light rail'), 2)
        self.assertEqual(code('Train'), 2)
        self.assertEqual(code('Ferry'), 0)

    def test_transit_submode_fit_keeps_observed_train_share(self):
        fit = self.transit_helpers['_fit_transit_submode']
        rows = pd.DataFrame({
            'transit_submode': ['Bus'] * 10 + ['Train'] * 10,
            'OTAZ': [1] * 20,
            'DTAZ': list(range(1, 21)),
            'SOVTOLL_DIST__AM': np.arange(20, dtype=float) + 1,
            'WLK_LOC_WLK_TOTIVT__AM': np.arange(20, dtype=float) + 30,
            'WLK_LOC_WLK_WAIT__AM': np.arange(20, dtype=float) + 5,
            'WLK_LOC_WLK_BOARDS__AM': np.arange(20, dtype=float) / 10 + 1,
        })

        model = fit(rows)

        self.assertEqual(model['n_obs'], 20)
        self.assertAlmostEqual(model['train_share'], 0.5)
        self.assertIn('transit_ivt_hr', model['feature_names'])

    def test_drive_access_requires_vehicle_and_uses_drive_los(self):
        predict_access = self.transit_helpers['_predict_transit_access']
        rows = pd.DataFrame({
            'OTAZ': [1] * 200,
            'DTAZ': [2] * 200,
            'CARLICENCE': [1] * 200,
            'vehrd': [1] * 200,
        })
        lookup = {(1, 2): {
            'WLK_LOC_WLK_TOTIVT__AM': 100.0,
            'WLK_LOC_DRV_TOTIVT__AM': 30.0,
            'WLK_LOC_DRV_DTIM__AM': 5.0,
        }}

        access = predict_access(rows, lookup, np.random.default_rng(7))
        self.assertGreater(float((access == 2).mean()), 0.5)

        rows['CARLICENCE'] = 0
        walk_only = predict_access(rows, lookup, np.random.default_rng(7))
        self.assertTrue(np.all(walk_only == 1))

    def test_drive_access_transit_time_uses_drive_matrix(self):
        get_tt = self.transit_helpers['_get_tt']
        lookup = {(1, 2): {
            'WLK_LOC_WLK_TOTIVT__AM': 90.0,
            'WLK_LOC_DRV_TOTIVT__AM': 35.0,
        }}

        self.assertEqual(get_tt(lookup, 1, 2, 5, 2), 35.0)
        self.assertEqual(get_tt(lookup, 1, 2, 5, 1), 90.0)


if __name__ == '__main__':
    unittest.main()