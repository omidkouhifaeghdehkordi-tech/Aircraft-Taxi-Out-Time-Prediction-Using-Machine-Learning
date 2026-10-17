import unittest

from training.feature_policy import (
    FORBIDDEN_FEATURES,
    PUBLIC_FEATURES,
    TARGET_COLUMN,
    TIME_COLUMN,
    validate_feature_policy,
)


class FeaturePolicyTests(unittest.TestCase):
    def test_public_features_do_not_contain_forbidden_fields(self):
        self.assertFalse(set(PUBLIC_FEATURES) & set(FORBIDDEN_FEATURES))

    def test_complete_schema_passes(self):
        validate_feature_policy(PUBLIC_FEATURES + [TARGET_COLUMN, TIME_COLUMN])

    def test_missing_required_feature_fails(self):
        incomplete = [column for column in PUBLIC_FEATURES if column != "aircraft_family"]
        with self.assertRaisesRegex(ValueError, "aircraft_family"):
            validate_feature_policy(incomplete + [TARGET_COLUMN, TIME_COLUMN])


if __name__ == "__main__":
    unittest.main()
