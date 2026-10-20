import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from main import load_data, FEATURES, TARGET


class TestMLCaseStudy(unittest.TestCase):
    def test_required_columns_exist(self):
        df = load_data()
        self.assertTrue(set(FEATURES + [TARGET]).issubset(df.columns))

    def test_data_has_no_missing_values_in_required_columns(self):
        df = load_data()
        self.assertFalse(df[FEATURES + [TARGET]].isnull().any().any())

    def test_target_has_two_classes(self):
        df = load_data()
        self.assertEqual(df[TARGET].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
