import unittest
import os
import json
from modules.hooks import write_smt_txt, write_smt_num

class Test_SMT_Text_Outcome(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.outcome_path = "tests/sumatra_outcome.json"

    def tearDown(self):
        if os.path.exists(self.outcome_path):
            os.remove(self.outcome_path)

    def test_create_new_outcome(self):
        write_smt_txt("Awesome", "tests")
        self.assertTrue(os.path.exists(self.outcome_path))

    def test_new_outcome_is_dict(self):
        write_smt_txt("Awesome", "tests")
        with open(self.outcome_path, "r") as outcome_file:
            outcome = json.load(outcome_file)
            self.assertIsInstance(outcome, dict)

    def test_new_outcome_keys(self):
        write_smt_txt("Awesome", "tests")
        with open(self.outcome_path, "r") as outcome_file:
            outcome = json.load(outcome_file)
            self.assertEqual(len(outcome), 2) 
            self.assertIn("text_outcome", outcome)
            self.assertIn("numeric_outcome", outcome)

    def test_new_outcome_content(self):
        write_smt_txt("Awesome", "tests")
        with open(self.outcome_path, "r") as outcome_file:
            outcome = json.load(outcome_file)
            self.assertEqual(outcome["text_outcome"], "| Awesome") 

    def test_appending_content_with_linebreak(self):
        write_smt_txt("Awesome", "tests")
        write_smt_txt("Great", "tests")

        with open(self.outcome_path, "r") as outcome_file:
            outcome = json.load(outcome_file)
            self.assertEqual(outcome["text_outcome"],
                             "| Awesome\n| Great") 

    def test_appending_content_inline(self):
        write_smt_txt("Awesome", "tests")
        write_smt_txt("Great", "tests", inline=True)

        with open(self.outcome_path, "r") as outcome_file:
            outcome = json.load(outcome_file)
            self.assertEqual(outcome["text_outcome"],
                             "| Awesome Great")

    def test_create_with_int_metric(self):
        write_smt_txt(22, "tests", metric="xy")

        with open(self.outcome_path, "r") as outcome_file:
            outcome = json.load(outcome_file)
            self.assertEqual(outcome["text_outcome"],
                             "| xy: 22")

    def test_create_with_float_metric(self):
        write_smt_txt(1.239, "tests", metric="xy")

        with open(self.outcome_path, "r") as outcome_file:
            outcome = json.load(outcome_file)
            self.assertEqual(outcome["text_outcome"],
                             "| xy: 1.24")

    def test_append_with_metric_inline(self):
        write_smt_txt("Awesome", "tests")
        write_smt_txt(22, "tests", metric="xy", inline=True)

        with open(self.outcome_path, "r") as outcome_file:
            outcome = json.load(outcome_file)
            self.assertEqual(outcome["text_outcome"],
                             "| Awesome xy: 22")


class Test_SMT_Numerical_Outcome(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.outcome_path = "tests/sumatra_outcome.json"

    def tearDown(self):
        if os.path.exists(self.outcome_path):
            os.remove(self.outcome_path)

    def test_create(self):
        write_smt_num(x=0, y=42, metric="accuracy",
                      smt_outcome_path=self.outcome_path)
        self.assertTrue(os.path.exists(self.outcome_path))


    def test_new_outcome_keys(self):
        write_smt_num(x=0, y=42, metric="accuracy",
                      smt_outcome_path=self.outcome_path)

        with open(self.outcome_path, "r") as outcome_file:
            outcome = json.load(outcome_file)
            self.assertIn("numeric_outcome", outcome)
            self.assertIn("accuracy", outcome["numeric_outcome"])
            self.assertIn("x", outcome["numeric_outcome"]["accuracy"])
            self.assertIn("y", outcome["numeric_outcome"]["accuracy"])
            self.assertIn("x_label", outcome["numeric_outcome"]["accuracy"])

    def test_new_outcome_values(self):
        write_smt_num(x=0, y=42, metric="accuracy",
                      smt_outcome_path=self.outcome_path)

        with open(self.outcome_path, "r") as outcome_file:
            outcome = json.load(outcome_file)
            self.assertEqual(outcome["numeric_outcome"]["accuracy"]["x"], [0])
            self.assertEqual(outcome["numeric_outcome"]["accuracy"]["y"], [42]) 
            self.assertEqual(outcome["numeric_outcome"]["accuracy"]["x_label"], "") 

    def test_new_outcome_values_with_x_label(self):
        write_smt_num(x=0, y=42, metric="accuracy", x_label="steps",
                      smt_outcome_path=self.outcome_path)

        with open(self.outcome_path, "r") as outcome_file:
            outcome = json.load(outcome_file)
            self.assertEqual(outcome["numeric_outcome"]["accuracy"]["x"], [0])
            self.assertEqual(outcome["numeric_outcome"]["accuracy"]["y"], [42]) 
            self.assertEqual(outcome["numeric_outcome"]["accuracy"]["x_label"],
                             "steps")

    def test_append(self):
        write_smt_num(x=0, y=42, metric="accuracy",
                      smt_outcome_path=self.outcome_path)
        write_smt_num(x=1, y=43, metric="accuracy",
                      smt_outcome_path=self.outcome_path)

        with open(self.outcome_path, "r") as outcome_file:
            outcome = json.load(outcome_file)
            self.assertEqual(outcome["numeric_outcome"]["accuracy"]["x"], [0, 1])
            self.assertEqual(outcome["numeric_outcome"]["accuracy"]["y"], [42, 43]) 
            self.assertEqual(outcome["numeric_outcome"]["accuracy"]["x_label"], "")

    def test_append_with_new_label(self):
        write_smt_num(x=0, y=42, metric="accuracy",
                      smt_outcome_path=self.outcome_path)
        write_smt_num(x=1, y=43, metric="accuracy", x_label="steps",
                      smt_outcome_path=self.outcome_path)

        with open(self.outcome_path, "r") as outcome_file:
            outcome = json.load(outcome_file)
            self.assertEqual(outcome["numeric_outcome"]["accuracy"]["x"], [0, 1])
            self.assertEqual(outcome["numeric_outcome"]["accuracy"]["y"], [42, 43]) 
            self.assertEqual(outcome["numeric_outcome"]["accuracy"]["x_label"],
                             "steps")

    def test_append_new_metric(self):
        write_smt_num(x=0, y=42, metric="accuracy",
                      smt_outcome_path=self.outcome_path)
        write_smt_num(x=1, y=0.1, metric="F1",
                      smt_outcome_path=self.outcome_path)

        with open(self.outcome_path, "r") as outcome_file:
            outcome = json.load(outcome_file)

            self.assertIn("F1", outcome["numeric_outcome"])

            self.assertEqual(outcome["numeric_outcome"]["accuracy"]["x"], [0])
            self.assertEqual(outcome["numeric_outcome"]["accuracy"]["y"], [42]) 
            self.assertEqual(outcome["numeric_outcome"]["accuracy"]["x_label"], "")

            self.assertEqual(outcome["numeric_outcome"]["F1"]["x"], [1])
            self.assertEqual(outcome["numeric_outcome"]["F1"]["y"], [0.1]) 
            self.assertEqual(outcome["numeric_outcome"]["F1"]["x_label"], "")