import numpy as np
import nibabel as nib
import tensorflow as tf

from tensorflow.python.framework import errors
from modules.models.example_loader import PointExamples, Examples
from modules.models.utils import np_placeholder
from copy import deepcopy


def is_mask(X):

    if isinstance(X, str):
        X = nib.load(X).get_data()
    elif (isinstance(X, nib.nifti2.Nifti2Image) or
          isinstance(dwi_loaded, nib.nifti1.Nifti1Image)):
        X = X.get_data()

    assert isinstance(X, np.ndarray)

    if len(X.shape) == 3 and X.max() <= 1 and X.min() >= 0:
        return True
    else:
        return False


def is_nifti(X):
    return False if (X[-3:] != "nii" and X[-6:] != "nii.gz") else True


def is_trk(y):
    return False if y[-3:] != "trk" else True


def is_tracking_data(X, Y):
    for x in X:
        if not is_nifti(x):
            return False
    for y in Y:
        if not is_trk(y):
            return False
    return True


def input(X, y=None, input_fn_config={"shuffle": True}):

    assert isinstance(X, (np.ndarray, dict, list))

    if isinstance(X, np.ndarray):
        train_size = X.shape[0]
        # ============================================================
        feature_spec ={"X": np_placeholder(X)}
        # ============================================================
        assert (y is None or isinstance(y, np.ndarray))

        input_fn = tf.estimator.inputs.numpy_input_fn(x={"X": X},
                                                      y=y,
                                                      **input_fn_config)
    # TODO: X as dict that contains e.g. data file strings
    elif isinstance(X, dict):
        train_size = None
        for key, val in X.items():
            assert isinstance(val, np.ndarray)

            if train_size is not None:
                assert train_size == val.shape[0]
            else:
                train_size = val.shape[0]
                assert isinstance(train_size, int)
        # ============================================================
        assert (y is None or isinstance(y, np.ndarray))

        input_fn = tf.estimator.inputs.numpy_input_fn(x=X,
                                                      y=y,
                                                      **input_fn_config)
        # ============================================================
        feature_spec = {}
        for key, val in X.items():
            feature_spec[key] = np_placeholder(val)

    elif isinstance(X, list):
        assert all(isinstance(x, str) for x in X)

        if is_tracking_data(X, y):
            return tracking_input(X, y, **input_fn_config)
        else:
            raise NotImplementedError("Only tracking data implemented "
                "for input X as list of strings.")

    return input_fn, feature_spec, train_size


def tracking_input(nii_file,
                   trk_file,
                   block_size,
                   n_incoming,
                   batch_size,
                   num_epochs,
		           shuffle=True,
		           buffer_size=10000,
                   min_fiber_length=0,
                   every_n_fibers=None,
                   load_only_n_samples=False):

    examples = PointExamples(nii_file,
                        trk_file,
                        block_size,
                        n_incoming=n_incoming,
                        num_eval_examples=0,
                        min_fiber_length=min_fiber_length,
                        every_n_fibers=every_n_fibers,
                        load_only_n_samples=load_only_n_samples)

    generator = examples.get_generator()

    block_shape = [block_size] * 3 + [examples.brain_data[0].shape[-1]]
    incoming_shape = [n_incoming, 3]

    def input_fn():
        dataset =tf.data.Dataset.from_generator(generator,
            ({"blocks": tf.float32, "incoming": tf.float32}, tf.float32),
            ({"blocks": tf.TensorShape(block_shape),
              "incoming": tf.TensorShape(incoming_shape)},
            tf.TensorShape([3])))
        dataset = dataset.batch(batch_size)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
              
    feature_spec = {}
    feature_spec["blocks"] = np_placeholder(np.ones(tuple(block_shape), dtype=np.float32))
    feature_spec["incoming"] = np_placeholder(np.ones(tuple(incoming_shape), dtype=np.float32))
        
    train_size = min(examples.n_labels) * len(examples.brain_data)

    return input_fn, feature_spec, train_size
    
