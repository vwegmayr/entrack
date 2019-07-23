import numpy as np
import tensorflow as tf
import sklearn as skl

from modules.models.utils import parse_hooks, parse_layers, get_rate
from modules.models.base import DeterministicTracker, ProbabilisticTracker

from tensorflow.python.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY)
from tensorflow.python.estimator.export.export_output import PredictOutput
from tensorflow.python.ops.variable_scope import variable_scope as var_scope


class SimpleTracker(DeterministicTracker):
    """docstring for ExampleTF"""

    def __init__(self, input_fn_config={"shuffle": True}, config={},
                 params={}):  # noqa: E129

        super(SimpleTracker, self).__init__(input_fn_config, config, params)

    def model_fn(self, features, labels, mode, params, config):

        blocks = tf.layers.flatten(features["blocks"])

        incoming = tf.layers.flatten(features["incoming"])

        concat = tf.concat([blocks, incoming], axis=1)

        unnormed = parse_layers(
            inputs=concat,
            layers=params["layers"],
            mode=mode,
            default_summaries=params["default_summaries"])

        normed = tf.nn.l2_normalize(unnormed, dim=1)

        # ================================================================
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"directions": normed},
                export_outputs={
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    PredictOutput({
                        "directions": normed
                    })
                })
        # ================================================================
        loss = -tf.multiply(normed, labels)
        loss = tf.reduce_sum(loss, axis=1)
        loss = tf.reduce_mean(loss)

        optimizer = params["optimizer_class"](
            learning_rate=params["learning_rate"])

        gradients, variables = zip(*optimizer.compute_gradients(loss))

        gradient_global_norm = tf.global_norm(gradients, name="global_norm")

        if "gradient_max_norm" in params:
            gradients, _ = tf.clip_by_global_norm(gradients,
                                                  params["gradient_max_norm"],
                                                  use_norm=gradient_global_norm)

        train_op = optimizer.apply_gradients(grads_and_vars=zip(gradients, variables),
                                             global_step=tf.train.get_global_step())
        # ================================================================

        training_hooks = parse_hooks(params, locals(), self.save_path)
        # ================================================================
        if (mode == tf.estimator.ModeKeys.TRAIN
                or mode == tf.estimator.ModeKeys.EVAL):  # noqa: E129
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=training_hooks)

    def score(self, X):
        pass


class MaxEntropyTracker(ProbabilisticTracker):
    """Implementation of the maximimum entropy probabilistic tracking."""

    def __init__(self, input_fn_config={"shuffle": True}, config={},
                 params={}):  # noqa: E129

        super(MaxEntropyTracker, self).__init__(input_fn_config, config,
                                                params)

    def model_fn(self, features, labels, mode, params, config):

        blocks = tf.layers.flatten(features["blocks"])

        incoming = tf.layers.flatten(features["incoming"])

        concat = tf.concat([blocks, incoming], axis=1)

        shared_layers = parse_layers(
            inputs=concat,
            layers=params["shared_layers"],
            mode=mode,
            default_summaries=params["default_summaries"])

        mu_out = parse_layers(
            inputs=shared_layers,
            layers=params["mu_head"],
            mode=mode,
            default_summaries=params["default_summaries"])

        k_out = parse_layers(
            inputs=shared_layers,
            layers=params["k_head"],
            mode=mode,
            default_summaries=params["default_summaries"])

        # Normalize the mean vectors
        mu_normed = tf.nn.l2_normalize(mu_out, dim=1)

        predictions = {
            'mean': mu_normed,
            'concentration': k_out
        }
        # ================================================================
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    PredictOutput(predictions)
                })
        # ================================================================

        dot_products = tf.reduce_sum(tf.multiply(mu_normed, labels), axis=1)

        W = self.W_stabilized(k_out, 10**-12)

        H = self.H_stabilized(k_out, 10**-12)

        cost = -tf.multiply(W, dot_products)

        T_H = -get_rate(params["temp"]) * H

        loss = cost + T_H

        loss = tf.reduce_mean(loss)

        optimizer = params["optimizer_class"](
            learning_rate=get_rate(params["learning_rate"]))

        gradients, variables = zip(*optimizer.compute_gradients(loss))

        gradient_global_norm = tf.global_norm(gradients, name="global_norm")

        if "gradient_max_norm" in params:
            gradients, _ = tf.clip_by_global_norm(gradients,
                                                  params["gradient_max_norm"],
                                                  use_norm=gradient_global_norm)

        train_op = optimizer.apply_gradients(grads_and_vars=zip(gradients, variables),
                                             global_step=tf.train.get_global_step())

        # ================================================================

        training_hooks = parse_hooks(params, locals(), self.save_path)

        # ================================================================
        if (mode == tf.estimator.ModeKeys.TRAIN
                or mode == tf.estimator.ModeKeys.EVAL):  # noqa: E129
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=training_hooks)

    @staticmethod
    def max_entropy_loss(y, mu, k, T):
        """Compute the maximum entropy loss.

        Args:
            y: Ground-truth fiber direction vectors.
            mu: Predicted mean vectors.
            k: Concentration parameters.
            T: Temperature parameter.

        Returns:
            loss: The maximum entropy loss.

        """
        dot_products = tf.reduce_sum(tf.multiply(mu, y), axis=1)

        W = tf.cosh(k) / (tf.sinh(k))- tf.reciprocal(k)

        cost = -tf.multiply(W, dot_products)

        entropy = 1 - k / tf.tanh(k) - tf.log(k / (4 * np.pi * tf.sinh(k)))

        loss = cost - T * entropy
        loss = tf.reduce_mean(loss)
        return loss

    @staticmethod
    def W_stabilized(k, eps):
        return (k * (1 + tf.exp(-2 * k)) - (1 - tf.exp(-2 * k))) / (eps + k * (1 - tf.exp(-2 * k)))

    @staticmethod
    def H_stabilized(k, eps):
        return (2 * k * tf.exp(-2 * k) / (1 + tf.exp(-2 * k))
                - tf.log(k + eps)
                + tf.log1p(eps - tf.exp(-2 * k))
                + tf.log(float(2 * np.pi * np.exp(1)))
               )


class BayesianTracker(MaxEntropyTracker):

    def model_fn(self, features, labels, mode, params, config):

        blocks = tf.layers.flatten(features["blocks"])

        #incoming = tf.layers.flatten(features["incoming"])
        #incoming = tf.slice(incoming, begin=[0, 0], size=[-1, 3])
        incoming = features["incoming"][:, 0]

        shared_layers = parse_layers(
            inputs=blocks,
            layers=params["shared_layers"],
            mode=mode,
            default_summaries=params["default_summaries"])

        mu_out = parse_layers(
            inputs=shared_layers,
            layers=params["mu_head"],
            mode=mode,
            default_summaries=params["default_summaries"])

        k_out = parse_layers(
            inputs=shared_layers,
            layers=params["k_head"],
            mode=mode,
            default_summaries=params["default_summaries"])

        mu_normed = tf.nn.l2_normalize(mu_out, dim=1)

        mu_tilde = k_out * (mu_normed + incoming)

        k_out = tf.norm(mu_tilde, axis=1, keep_dims=True)
        mean = tf.nn.l2_normalize(mu_tilde, dim=1)

        predictions = {
            'mean': mean,
            'concentration': k_out
        }
        # ================================================================
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    PredictOutput(predictions)
                })
        # ================================================================

        dot_products = tf.reduce_sum(tf.multiply(mean, labels), axis=1)

        W = self.W_stabilized(k_out, 10**-12)

        H = self.H_stabilized(k_out, 10**-12)

        cost = -tf.multiply(W, dot_products)

        T_H = -get_rate(params["temp"]) * H

        loss = cost + T_H

        loss = tf.reduce_mean(loss)

        optimizer = params["optimizer_class"](
            learning_rate=params["learning_rate"])

        gradients, variables = zip(*optimizer.compute_gradients(loss))

        gradient_global_norm = tf.global_norm(gradients, name="global_norm")

        if "gradient_max_norm" in params:
            gradients, _ = tf.clip_by_global_norm(gradients,
                                                  params["gradient_max_norm"],
                                                  use_norm=gradient_global_norm)

        train_op = optimizer.apply_gradients(grads_and_vars=zip(gradients, variables),
                                             global_step=tf.train.get_global_step())

        # ================================================================

        training_hooks = parse_hooks(params, locals(), self.save_path)

        # ================================================================
        if (mode == tf.estimator.ModeKeys.TRAIN
                or mode == tf.estimator.ModeKeys.EVAL):  # noqa: E129
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=training_hooks)
