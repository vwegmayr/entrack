"""Tests for the maximum entropy loss function.

The computations have been verified in Wolfram Matematica. The implementation
of the loss function in Mathematica is:

  L[mu_, k_, t_, v_] := - (Cosh[k]/Sinh[k] - 1/k) * Total[mu * v, {2}]
                        - t * (1 - k / Tanh[k] - Log[k / (4 * Pi * Sinh[k])])
"""

import tensorflow as tf
from modules.models.trackers import MaxEntropyTracker


class TestMaxEntropy(tf.test.TestCase):
    """Test the maximum entropy implementation.

    Focus on the loss.
    """

    def test_max_entropy_loss_lists(self):
        """Test computations, from lists."""
        with self.test_session():
            # Test 1
            mu = [[1.0, 0]]
            y = [[2.0, 0]]
            T = float(1)
            k = float(1)
            loss = MaxEntropyTracker.max_entropy_loss(y, mu, k, T)
            self.assertAlmostEqual(loss.eval(), -3.005498894039818)
            # Test 2
            mu2 = [[1.0, 0, 0], [2.0, 0, 0]]
            y2 = [[2.0, 0, 0], [3.0, 0, 0]]
            T2 = float(1)
            k2 = [1.0, 1.0]
            loss = MaxEntropyTracker.max_entropy_loss(y2, mu2, k2, T2)
            self.assertAlmostEqual(loss.eval(), -3.6315694)

    def test_max_entropy_loss_tensors_float64(self):
        """Test computations, from constant tensors type tf.float64."""
        with self.test_session():
            # Test 1
            mu = tf.constant([[1.0, 0]], dtype=tf.float64)
            y = tf.constant([[2.0, 0]], dtype=tf.float64)
            T = tf.constant(1, dtype=tf.float64)
            k = tf.constant(1, dtype=tf.float64)
            loss = MaxEntropyTracker.max_entropy_loss(y, mu, k, T)
            self.assertAlmostEqual(loss.eval(), -3.005498894039818)
            # Test 2
            mu2 = tf.constant([[1.0, 0, 0], [2.0, 0, 0]], dtype=tf.float64)
            y2 = tf.constant([[2.0, 0, 0], [3.0, 0, 0]], dtype=tf.float64)
            T2 = tf.constant(1, dtype=tf.float64)
            k2 = tf.constant([1.0, 1.0], dtype=tf.float64)
            loss = MaxEntropyTracker.max_entropy_loss(y2, mu2, k2, T2)
            self.assertAlmostEqual(loss.eval(), -3.63156946503848)

    def test_max_entropy_loss_tensors_float32(self):
        """Test computations, from constant tensors type tf.float32."""
        with self.test_session():
            # Test 1
            mu = tf.constant([[1.0, 0]], dtype=tf.float32)
            y = tf.constant([[2.0, 0]], dtype=tf.float32)
            T = tf.constant(1, dtype=tf.float32)
            k = tf.constant(1, dtype=tf.float32)
            loss = MaxEntropyTracker.max_entropy_loss(y, mu, k, T)
            self.assertAlmostEqual(loss.eval(), -3.005498894039818)
            # Test 2
            mu2 = tf.constant([[1.0, 0, 0], [2.0, 0, 0]], dtype=tf.float32)
            y2 = tf.constant([[2.0, 0, 0], [3.0, 0, 0]], dtype=tf.float32)
            T2 = tf.constant(1, dtype=tf.float32)
            k2 = tf.constant([1.0, 1.0], dtype=tf.float32)
            loss = MaxEntropyTracker.max_entropy_loss(y2, mu2, k2, T2)
            self.assertAlmostEqual(loss.eval(), -3.6315694)   # Less precise


if __name__ == "__main__":
    tf.test.main()
