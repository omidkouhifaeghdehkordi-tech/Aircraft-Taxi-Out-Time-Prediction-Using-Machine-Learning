import unittest

import pandas as pd

from data_preparation.build_model_table import MODEL_COLUMNS, build_model_table


class DataPreparationTests(unittest.TestCase):
    def test_builds_leakage_safe_model_table(self):
        source = pd.DataFrame(
            {
                "first_groundspeed_positive_ts_utc": ["2026-05-01T08:15:00Z"],
                "taxi_time_minutes": [12.5],
                "flight_number": ["VY1234"],
                "callsign": ["VLG1234"],
                "aircraft_type": ["A320"],
                "wind_speed_kt": [8.0],
                "wind_direction_deg": [180.0],
                "runway_takeoff": ["24L"],
                "queue_size_estimated": [4],
            }
        )

        result = build_model_table(source)

        self.assertEqual(result.columns.tolist(), MODEL_COLUMNS)
        self.assertEqual(result.loc[0, "flight_prefix"], "VY")
        self.assertEqual(result.loc[0, "aircraft_family"], "A320_family")
        self.assertNotIn("runway_takeoff", result.columns)
        self.assertNotIn("queue_size_estimated", result.columns)

    def test_filters_invalid_target(self):
        source = pd.DataFrame(
            {
                "first_groundspeed_positive_ts_utc": ["2026-05-01T08:15:00Z"],
                "taxi_time_minutes": [94.0],
                "flight_number": ["VY1234"],
                "callsign": ["VLG1234"],
                "aircraft_type": ["A320"],
                "wind_speed_kt": [8.0],
                "wind_direction_deg": [180.0],
            }
        )

        self.assertTrue(build_model_table(source).empty)


if __name__ == "__main__":
    unittest.main()
