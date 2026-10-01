#!/usr/bin/env python
"""Focused tests for Overture destination land-use aggregation."""
import unittest

import pandas as pd

from build_zone_landuse_overture import _aggregate_sa1_rows


class OvertureLanduseTests(unittest.TestCase):
    def test_sa1_aggregate_keeps_specific_destination_features(self):
        locations = pd.DataFrame({
            'SA1_CODE21': ['101', '101', '101', '202'],
            'SA2_CODE21': ['10', '10', '10', '20'],
            'land_use': ['recreation', 'health', 'education', 'education'],
            'poi_category': ['park', 'doctors', 'school', 'university'],
            'lu_area': [1000.0, 400.0, 2000.0, 5000.0],
            'is_work': [1, 1, 1, 1],
            'is_school': [0, 0, 1, 1],
            'is_discretionary': [1, 1, 0, 0],
            'is_home': [0, 0, 0, 0],
            'est_persons': [10.0, 5.0, 20.0, 50.0],
            'person_kind': ['worker', 'worker', 'student', 'student'],
        })

        result = _aggregate_sa1_rows(locations).set_index('SA1_CODE21')

        self.assertEqual(result.loc[101, 'SA2_CODE21'], '10')
        self.assertEqual(result.loc[101, 'n_parks'], 1)
        self.assertEqual(result.loc[101, 'n_doctors_clinics'], 1)
        self.assertEqual(result.loc[101, 'n_schools'], 1)
        self.assertEqual(result.loc[101, 'n_universities'], 0)
        self.assertEqual(result.loc[202, 'n_universities'], 1)
        self.assertEqual(result.loc[202, 'n_schools'], 0)


if __name__ == '__main__':
    unittest.main()
