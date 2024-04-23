import unittest
import sys
import os
from _data_helper import data_gen


import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from nampy.formulas.formulas import FormulaHandler


class TestFormulaHandler(unittest.TestCase):
    def test_formula_simple(self):
        data = data_gen()
        formula = (
            "target ~ -1 + MLP(ContinuousFeature) + RandomFourierNet(IntegerFeature)"
        )

        FH = FormulaHandler()
        (
            feature_names,
            target_name,
            terms,
            intercept,
            feature_information,
        ) = FH._extract_formula_data(formula, data)

        self.assertIn("ContinuousFeature", feature_names)
        self.assertIn("IntegerFeature", feature_names)
        self.assertEqual(target_name, "target")
        self.assertEqual(intercept, False)

    def test_formula_intercept(self):
        data = data_gen()
        formula = (
            "target ~ +1 + MLP(ContinuousFeature) + RandomFourierNet(IntegerFeature)"
        )

        FH = FormulaHandler()
        (
            feature_names,
            target_name,
            terms,
            intercept,
            feature_information,
        ) = FH._extract_formula_data(formula, data)

        self.assertIn("ContinuousFeature", feature_names)
        self.assertIn("IntegerFeature", feature_names)
        self.assertEqual(target_name, "target")
        self.assertEqual(intercept, True)

    def test_formula_interaction(self):
        data = data_gen()
        formula = "target ~ +1 + MLP(ContinuousFeature) + RandomFourierNet(IntegerFeature) + MLP(CategoricalFeature):MLP(ContinuousFeature)"

        FH = FormulaHandler()
        (
            feature_names,
            target_name,
            terms,
            intercept,
            feature_information,
        ) = FH._extract_formula_data(formula, data)

        self.assertIn("ContinuousFeature", feature_names)
        self.assertIn("IntegerFeature", feature_names)
        self.assertIn("CategoricalFeature", feature_names)
        self.assertEqual(target_name, "target")
        self.assertEqual(intercept, True)
        self.assertIn(":", terms[-1])

    def test_formula_networks(self):
        data = data_gen()
        formula = "target ~ -1 + CustomNet(ContinuousFeature) + RandomFourierNet(IntegerFeature) + CubicSPlineNet(CategoricalFeature)"

        FH = FormulaHandler()
        (
            feature_names,
            target_name,
            terms,
            intercept,
            feature_information,
        ) = FH._extract_formula_data(formula, data)

        self.assertIn("ContinuousFeature", feature_names)
        self.assertIn("IntegerFeature", feature_names)
        self.assertIn("CategoricalFeature", feature_names)
        self.assertEqual(target_name, "target")
        self.assertEqual(intercept, False)
        self.assertEqual(feature_information[feature_names[0]]["Network"], "CustomNet")
        self.assertEqual(
            feature_information[feature_names[1]]["Network"], "RandomFourierNet"
        )
        self.assertEqual(
            feature_information[feature_names[2]]["Network"], "CubicSPlineNet"
        )


if __name__ == "__main__":
    unittest.main()
