import unittest
import types
import numpy as np
from modules.models.example_loader import PointExamples

class TestPointExamples(unittest.TestCase):
    """Test some functionalities of the example loader."""

    @classmethod
    def setUpClass(self):
        TRK = "tests/data/fibers.trk"
        NII = "tests/data/dwi_train.nii.gz"
        self.loader = PointExamples(
            NII,
            TRK,
            block_size=3,
            n_incoming=3,
            num_eval_examples=0,
            example_percent=0.25,
        )

        self.generator = self.loader.get_generator()()
        self.example = next(self.generator)

    @classmethod
    def tearDownClass(self):
        pass

    def test_generator(self):
        self.assertIsInstance(self.generator, types.GeneratorType)

    def test_example(self):    
        self.assertIsInstance(self.example, tuple)
        self.assertEqual(len(self.example), 2)

    def test_features(self):
        features = self.example[0]
        self.assertIsInstance(features, dict)
        self.assertEqual(len(features), 2)

        self.assertTrue("blocks" in features)
        self.assertTrue("incoming" in features)

        self.assertEqual(features["blocks"].shape, (3, 3, 3, 15))
        self.assertEqual(features["incoming"].shape, (3, 3))

    def test_target(self):
        target = self.example[1]
        self.assertIsInstance(target, np.ndarray)
        self.assertEqual(target.shape, (3,))

if __name__ == '__main__':
    unittest.main()
