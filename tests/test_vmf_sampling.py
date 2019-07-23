"""THE TESTS IN THIS MODULE ARE STUBSself."""

import unittest
import numpy as np

from modules.models.base import ProbabilisticTracker


class TestUnitCircleSampling(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.n_samples = 50000
        self.samples = ProbabilisticTracker._sample_unif_unit_circle(self.n_samples)

    def test_mean(self):
        mean = np.mean(self.samples, axis=0)
        self.assertTrue(np.allclose(mean, [0,0], atol=10**-2))

    def test_shape(self):
        self.assertTrue(self.samples.shape == (self.n_samples, 2))

    def test_unit_norm(self):
        norm = np.linalg.norm(self.samples, axis=1)
        self.assertTrue(np.allclose(norm, np.ones(self.n_samples)))


class TestFvMSampling(unittest.TestCase):
     
    @classmethod
    def setUpClass(self):
        self.n_samples = 50000
        self.mu = np.array([1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
        self.k = 5.0
        self.samples = ProbabilisticTracker.sample_vMF(
            np.tile(self.mu,(self.n_samples, 1)),
            np.tile(self.k, (self.n_samples, 1))
        )

    def test_unit_norm(self):
        norm = np.linalg.norm(self.samples, axis=1)
        self.assertTrue(np.allclose(norm, np.ones(self.n_samples)))

    def test_shape(self):
        self.assertTrue(self.samples.shape == (self.n_samples, 3))

    def test_mean(self):
        mean = np.mean(self.samples, axis=0)
        self.assertTrue(np.allclose(
            mean, 
            (np.cosh(self.k) / np.sinh(self.k) - 1 / self.k) * self.mu,
            atol=10**-2)
        )


class TestRotation(unittest.TestCase):

    def test_rotation(self):
        vectors = np.asarray([[1, 0, 0]] * 4)
        references = np.asarray([[0, 1, 0]] * 4)
        rot = ProbabilisticTracker._rotation_matrices(vectors, references)


if __name__ == '__main__':
    unittest.main()
