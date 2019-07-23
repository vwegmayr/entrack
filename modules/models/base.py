import tensorflow as tf
import multiprocessing
import os
import nibabel as nib
import numpy as np
import subprocess
import json

from sklearn.utils.validation import check_is_fitted
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from termcolor import cprint

from modules.models.utils import (print, save_fibers,
    np_placeholder, make_hdr, flat_percentile, map_to_90deg_range)
from modules.models.example_loader import PointExamples, aff_to_rot
from modules.models.input_fn import input, is_nifti, is_mask
from modules.hooks import write_smt_num, write_smt_txt

from tensorflow.python.estimator.export.export import (
    build_raw_serving_input_receiver_fn as input_receiver_fn)
from tensorflow.python.platform import tf_logging as logging


class BaseTF(ABC, BaseEstimator, TransformerMixin):
    """docstring for BaseTF"""
    lock = multiprocessing.Lock()
    num_instances = 0

    def __init__(self, input_fn_config, config, params):
        super(BaseTF, self).__init__()
        self.input_fn_config = input_fn_config
        self.config = config
        self.params = params

        self._restore_path = None

        with BaseTF.lock:
            self.instance_id = BaseTF.num_instances
            BaseTF.num_instances += 1

    def fit(self, X, y):

        with BaseTF.lock:
            config = self.config
            if BaseTF.num_instances > 1:
                config["model_dir"] = os.path.join(
                    config["model_dir"],
                    "inst-" + str(self.instance_id))

        input_fn, self.feature_spec, train_size = (
            input(X, y, self.input_fn_config))

        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params={**self.params,
                    **self.input_fn_config,
                    "train_size": train_size},
            config=tf.estimator.RunConfig(**config))

        tf.logging.set_verbosity(tf.logging.INFO)
        try:
            self.estimator.train(input_fn=input_fn)
        except KeyboardInterrupt:
            print("\nEarly stop of training, saving model...")
            self.export_estimator()
            return self
        else:
            self.export_estimator()
            return self

    def predict(self, X, head="predictions"):
        check_is_fitted(self, ["_restore_path"])

        predictor = tf.contrib.predictor.from_saved_model(self._restore_path)

        if isinstance(X, np.ndarray):
            return predictor({"X": X})[head]
        elif isinstance(X, dict):
            return predictor(X)[head]

    def predictor(self):
        if self._restore_path is not None:
            return tf.contrib.predictor.from_saved_model(self._restore_path)

        elif self.estimator.latest_checkpoint() is not None:
            return tf.contrib.predictor.from_estimator(
                self.estimator,
                input_receiver_fn(self.feature_spec)
            )

        else:
            print("Neither _restore_path nor latest_checkpoint set. "
                  "Returning None, no predictor.")
            return None

    def predict_proba(self, X):
        return self.predict(X, head="probabs")

    def set_save_path(self, save_path):
        self.save_path = save_path
        if self._restore_path is None:
            self.config["model_dir"] = save_path

    def export_estimator(self):
        receiver_fn = input_receiver_fn(self.feature_spec)
        self._restore_path = self.estimator.export_savedmodel(
            self.save_path,
            receiver_fn)
        print("Model saved to {}".format(self._restore_path))

    @abstractmethod
    def score(self, X, y):
        pass

    @abstractmethod
    def model_fn(self, features, labels, mode, params, config):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()


        def remove_tensorflow(state):
            for key, val in list(state.items()):
                if "tensorflow" in getattr(val, "__module__", "None"):
                    del state[key]
                elif isinstance(val, dict):
                    remove_tensorflow(val)

        remove_tensorflow(state)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class BaseTracker(BaseTF):
    """Exension to BaseTF to enable fiber tracking.

    Extend this class to implement the methods and use for tracking.
    """

    def predict(self, X, args):
        """Generate the tracktography with the current model on the given brain."""
        # Check model
        check_is_fitted(self, ["params"])
        assert isinstance(X, list)
        assert len(X) == 2
        assert all(is_nifti(x) for x in X)

        self.args = args

        # Get brain information
        if is_mask(X[0]):
            wm_mask = nib.load(X[0]).get_data()
            brain_data = nib.load(X[1]).get_data()
            header = make_hdr(X[1])
        else:
            brain_data = nib.load(X[0]).get_data()
            header = make_hdr(X[0])
            wm_mask = nib.load(X[1]).get_data()

        if 'max_length' in args:
            self.max_length = args.max_length
        else:
            self.max_length = 100

        # If no seeds are specified, build them from the wm mask
        if 'seeds' not in self.args:
            if "threshold" in self.args:
                seeds = self._seeds_from_wm_mask(wm_mask, self.args.threshold)
            else:
                seeds = self._seeds_from_wm_mask(wm_mask)
        else:
            seeds = self.args.seeds
            seeds = nib.load(seeds).get_data()
            seeds = np.where(seeds > 0)
            seeds = np.dstack(seeds)[0]
            seeds = [[seed] for seed in seeds]

        predictor = self.predictor()
        if predictor is not None:
            tracks, scalars = self._generate_masked_tractography(
                brain_data,
                wm_mask,
                seeds,
                affine=header["vox_to_ras"],
                predictor=predictor)

            if "file_name" in args:
                if len(args.file_name) >= 4 and args.file_name[-4:] == ".trk":
                    fiber_path = os.path.join(
                        self.save_path, args.file_name)
                else:
                    fiber_path = os.path.join(
                        self.save_path, args.file_name + ".trk")
            else:
                fiber_path = os.path.join(self.save_path, "fibers.trk")

            if "min_length" in self.args:
                min_length = self.args.min_length
            else:
                min_length = 2

            save_fibers(tracks,
                        header,
                        fiber_path,
                        scalars=scalars,
                        min_length=min_length)

    @staticmethod
    def assert_equal_length(tracks, scalars):
        for key in scalars.keys():
            if len(tracks) != len(scalars[key]):
                raise ValueError("Length tracks != Length "
                                 "{} ({} != {})".format(key,
                                                        len(tracks),
                                                        len(scalars[key])))

            for idx, track in enumerate(tracks):
                if len(track) != len(scalars[key][idx]):
                    raise ValueError("Length of track does not match length of "
                        "scalar {} ({} != {})".format(key,
                                                      len(track),
                                                      len(scalars[key][idx]))
                        )


    def _generate_masked_tractography(
            self,
            brain_data,
            wm_mask,
            seeds,
            affine=None,
            predictor=None):
        """Generate the tractography using the white matter mask."""

        tracks = seeds
        scalars = {"concentration": [[] for _ in range(len(seeds))],
                   "fvm_probab": [[] for _ in range(len(seeds))],
                   "angles": [[] for _ in range(len(seeds))],
                   "inverted": [[] for _ in range(len(seeds))],
                   "probab": []}
        ongoing_idx = np.arange(len(seeds))

        while len(ongoing_idx) > 0:
            ongoing_fibers = np.asarray([tracks[i] for i in ongoing_idx])
            predictions = predictor(self._build_next_X(brain_data, ongoing_fibers, affine))
            if "use_mean" in self.args:
                directions = self.get_directions_from_predictions(predictions,
                                                                  affine,
                                                                  use_mean=self.args.use_mean)
            else:
                directions = self.get_directions_from_predictions(predictions,
                                                                  affine)

            next_pts = ongoing_fibers[:, -1, :] + directions * self.args.step_size

            if ongoing_fibers.shape[1] == 1:
                flip = list(map(lambda x: self._is_border(x, wm_mask), next_pts))
                next_pts[flip] = next_pts[flip] - 2 * directions[flip] * self.args.step_size

            not_terminal = list(map(lambda x: not self._is_border(x, wm_mask), next_pts))

            for i in range(len(ongoing_idx)):
                # Note: no need to normalize here.
                v = directions[i] # Affine already applied
                mu = self.apply_affine(predictions["mean"][i], affine)
                k = predictions["concentration"][i][0]
                scalars["concentration"][ongoing_idx[i]].append(k)

                inner_prod = np.clip(np.inner(mu, v), -1.0, 1.0)

                mu = np.sign(inner_prod)*mu
                scalars["inverted"][ongoing_idx[i]].append(1 if inner_prod < 0 else 0)

                fvm_probab = np.log(self.fvm_probab(v, mu, k) + 10**-12)
                scalars["fvm_probab"][ongoing_idx[i]].append(fvm_probab)

                angle = map_to_90deg_range(np.rad2deg(np.arccos(inner_prod)))
                scalars["angles"][ongoing_idx[i]].append(angle)

            if ongoing_fibers.shape[1] >= self.max_length:

                for i, fvm_probab in enumerate(scalars["fvm_probab"]):
                    probab = np.sum(fvm_probab)
                    scalars["probab"].append([probab] * len(fvm_probab))

                self.assert_equal_length(tracks, scalars)
                return tracks, scalars

            for i, next_pt in enumerate(next_pts):
                if not_terminal[i]:
                    tracks[ongoing_idx[i]].append(next_pt)

            ongoing_idx = ongoing_idx[not_terminal]

            cprint(
                "{:6d} / {} fibers going on.".format(len(ongoing_idx),
                                                     len(seeds)),
                "red", "on_grey",
                end="\r", flush=True)

        for i, fvm_probab in enumerate(scalars["fvm_probab"]):
            probab = np.sum(fvm_probab)
            scalars["probab"].append([probab] * len(fvm_probab))

        self.assert_equal_length(tracks, scalars)
        return tracks, scalars


    def _build_next_X(self, brain_data, ongoing_fibers, affine):
        """Builds the next X-batch to be fed to the model.

        The X-batch created continues the streamline based on the outgoing directions obtained at
        the previous step.

        Returns:
            next_X: The next batch of point values (blocks, incoming, centers).
        """
        label_type = "point"
        X = {
            'incoming': [],
            'blocks': []
        }

        for fiber in ongoing_fibers:
            center_point = fiber[-1]
            incoming_point = np.zeros((self.input_fn_config["n_incoming"], 3))
            outgoing = np.zeros(3)
            for i in range(min(self.input_fn_config["n_incoming"], len(fiber)-1)):
                incoming_point[i] = fiber[-i - 2]
            sample = PointExamples.build_datablock(brain_data,
                                                   self.input_fn_config["block_size"],
                                                   center_point,
                                                   incoming_point,
                                                   outgoing,
                                                   label_type,
                                                   affine)
            X_sample = {
                'incoming': sample['incoming'].reshape(-1, 3),
                'blocks': sample['data_block']
            }
            # Add example to examples by appending individual lists
            for key, cur_list in X.items():
                cur_list.append(X_sample[key])

        for key, _ in X.items():
            X[key] = np.array(X[key])

        return X

    @abstractmethod
    def get_directions_from_predictions(self, predictions, affine):
        """Computes fiber directions form the predictions of the network.

        Method to be extended in subclasses. By extending this the outputs of
        different types of networks can be used in the same way.

        Args:
            predictions: The output of the neural network model.
            affine: The affine transformation for the voxel space.
        Returns:
            directions: The fiber directions corresponding to the predictions.
        """
        pass

    def _seeds_from_wm_mask(self, wm_mask, threshold=0.5):
        """Compute the seeds for the streamlining from the white matter mask.

        This is invoked only if no seeds are specified.
        The seeds are selected on the interface between white and gray matter, i.e. they are the
        white matter voxels that have at least one gray matter neighboring voxel.
        These points are furthermore perturbed with some gaussian noise to have a wider range of
        starting points.

        Returns:
            seeds: The list of voxel that are seeds.
        """
        # Take te border voxels as seeds
        seeds = self._find_borders(wm_mask, threshold)
        print("Number of seeds on the white matter mask:", len(seeds))
        print("Number of requested seeds:", self.args.n_fibers)
        new_idxs = np.random.choice(len(seeds), self.args.n_fibers, replace=True)
        new_seeds = [[seeds[i] + np.clip(np.random.normal(0, 0.25, 3), -0.5, 0.5)]
                     for i in new_idxs]
        return new_seeds

    def _find_borders(self, wm_mask, threshold=0.5, order=1):
        """Find the wm-gm interface points.

        Args:
            order: How far from the center voxel to look for differen voxels. Default 1.
        Return:
            seeds: The seeds generated from the white matter mask
        """
        dim = wm_mask.shape
        borders = []

        if wm_mask.dtype != 'int' or wm_mask.dtype == 'int64':
            # If float, check if it is really not boolean
            if np.where(np.abs(wm_mask - 0.5) < 0.5)[0].shape[0] > 0:
                cprint("WARNING: The mask in use might not be binary. \
                        Thresholding at {} will be applied.".format(threshold))

        for x in range(order, dim[0] - order):
            for y in range(order, dim[1] - order):
                for z in range(order, dim[2] - order):
                    if wm_mask[x, y, z] > threshold: # Careful if using non-binary mask!
                        window = wm_mask[x - order:x + 1 + order,
                                              y - order:y + 1 + order,
                                              z - order:z + 1 + order]
                        if not np.all(window):
                            borders.append(np.array([x, y, z]))
        return borders

    def _is_border(self, coord, wm_mask):
        """Check if the voxel is on the white matter border.

        Args:
            coord: Numpy ndarray containing the [x, y, z] coordinates of the point.

        Returns:
            True if the [x, y, z] point is on the border.
        """
        coord = np.round(coord).astype(int)

        lowerbound_condition = coord[0] < 0 or coord[1] < 0 or coord[2] < 0
        upperbound_condition = coord[0] >= wm_mask.shape[0] or \
                               coord[1] >= wm_mask.shape[1] or \
                               coord[2] >= wm_mask.shape[2]

        # Check if out of image dimensions
        if lowerbound_condition or upperbound_condition:
            return True
        # Check if out of white matter area
        return np.isclose(wm_mask[coord[0], coord[1], coord[2]], 0.0)


    @staticmethod
    def apply_affine(directions, affine):
        return aff_to_rot(affine).dot(directions.T).T


class DeterministicTracker(BaseTracker):
    """Base model for deterministic tracking.

    A model does deterministic tracking when its output is the direction of the
    fiber (given the possible different inputs), not a probablity distribution
    (see ProbabilisticTracker).
    """

    def get_directions_from_predictions(self, predictions, affine, use_mean=False):
        """Compute the direction of the fibers from the deterministic predict.
        """
        return self.apply_affine(predictions["directions"], affine)


class ProbabilisticTracker(BaseTracker):
    """Base model for probabilistic tracking.

    This model assumes that the network does not output a direction but a
    probability distribution over the possible directions. In this case, the
    distribution is the Fisher-Von Mises distribution, with parameters [mu, k],
    where mu is the 3-dimensional mean-direciton vector and k is the
    concentration parameter.
    """

    def score(self, trk_file=None, y=None, args=None):

        if isinstance(trk_file, list):
            trk_file = trk_file[0]

        if trk_file is None:
            TM_DATA=["/local/entrack/data/tractometer/125mm/FODl4.nii.gz",
                     "/local/entrack/data/tractometer/125mm/wm_aligned.nii.gz"]
            args.file_name = "tm_fibers.trk"
            self.predict(TM_DATA, args)
            trk_file = os.path.join(self.save_path, args.file_name)

        TM_PATH = ("./.tractometer/ismrm_2015_tractography_challenge_scoring/"
                  "score_tractogram.py")
        SCORING_DATA = ("./.tractometer/ismrm_2015_tractography_challenge_"
                        "scoring/scoring_data/")
        scoring_cmd = "python {command} {tracts} {base} {out}".format(
            command=TM_PATH,
            tracts=trk_file,
            base=SCORING_DATA,
            out=self.save_path)
        subprocess.run(["bash", "-c", "source activate entrack_tm && {}"
                        .format(scoring_cmd)])

        eval_path = os.path.join(self.save_path, "scores", "tm_fibers.json")
        eval_data = json.load(open(eval_path))

        for metric in ["mean_OL", "mean_OR", "VC", "NC", "IC", "VB", "IB", "mean_F1"]:
            write_smt_txt(eval_data[metric],
                          self.save_path,
                          metric=metric,
                          inline=True)

        if "score" in args:
            return eval_data[args.score]


    def get_directions_from_predictions(self, predictions, affine, use_mean=False):
        mu = predictions['mean']
        k = predictions['concentration']

        if not use_mean:
            directions = ProbabilisticTracker.sample_vMF(mu, k)
        else:
            directions = mu

        return self.apply_affine(directions, affine)


    def save_scalars(self,
                     trk_file,
                     nii_file,
                     min_pts_per_fiber=2,
                     every_n_fibers=1,
                     file_name="scalars.trk"):

        scalars, tracks, trk_hdr = self.fvm_scalars(trk_file,
                                                    nii_file,
                                                    min_pts_per_fiber,
                                                    every_n_fibers)

        for metric in scalars.keys():
            for q in [25, 50, 75]:
                write_smt_txt(flat_percentile(scalars[metric], q),
                              self.save_path,
                              metric=metric + "_" + str(q),
                              inline=q>25)

        save_fibers(tracks,
                    trk_hdr,
                    os.path.join(self.save_path, file_name),
                    scalars=scalars)


    def fvm_scalars(self, trk_file, nii_file, min_pts_per_fiber=2, every_n_fibers=1):
        """Produce trk file marked with concentration and fvm_probab.

        fvm_scalars produces concentration and fvm_probab scalars that
        should be passed on to utils.save_fibers

        Args:
            trk_file (str): Path to trk file that contains the fibers to be marked.
            nii_file (str): Path to nifti file with corresponding diffusion data.

        Returns:
            scalars (dict): Dict to lists of shape (n_tracks, ). Currently, there
                            are only two keys: "concentration" and "fvm_probab".
            all_tracks (list): List of unmarked tracks of shape (n_tracks, ).
            trk_hdr: Header of trk_file.

        TODO:
            * Add some kind of skip parameter to reduce computation time.
            * Add min_length parameter.
            * Make fvm_scalars useable with trained model and not only during
              training.
        """
        tracks, trk_hdr = nib.trackvis.read(trk_file, points_space="voxel")
        trk_aff = nib.trackvis.aff_from_hdr(trk_hdr)
        all_tracks = []
        for i, track in enumerate(tracks):
            if len(track[0]) >= min_pts_per_fiber and i % every_n_fibers == 0:
                all_tracks.append(track[0])

        tracks = all_tracks[:]

        nii_file = nib.load(nii_file)
        nii_data = nii_file.get_data()
        nii_hdr = nii_file.header
        nii_aff = nii_file.affine

        assert np.allclose(nii_aff, trk_aff)

        block_size = self.input_fn_config["block_size"]
        n_incoming = self.input_fn_config["n_incoming"]
        n_tracks = len(all_tracks)

        predictor = self.predictor()

        ongoing_idx = list(range(n_tracks))
        track_lengths = list(map(lambda track: len(track), tracks))
        scalars = {"concentration": [[] for _ in range(n_tracks)],
                   "fvm_probab": [[] for _ in range(n_tracks)],
                   "angles": [[] for _ in range(n_tracks)],
                   "inverted": [[] for _ in range(n_tracks)],
                   "probab": []}

        cprint("Marking {} fibers...".format(n_tracks), "green", "on_grey", flush=True)
        while len(ongoing_idx) > 0:
            ongoing_tracks = [tracks[i] for i in ongoing_idx]
            n_ongoing = len(ongoing_tracks)
            batch = self._build_next_X(nii_data, ongoing_tracks, nii_aff)
            predictions = predictor(batch)

            for i, kappa in enumerate(predictions["concentration"]):
                scalars["concentration"][ongoing_idx[i]].append(kappa[0])

            for i in range(n_ongoing):
                k = predictions["concentration"][i][0]
                mu = self.apply_affine(predictions["mean"][i], nii_aff)

                ongoing_len = len(ongoing_tracks[i])
                assert ongoing_len >= 1

                if ongoing_len == len(all_tracks[ongoing_idx[i]]):
                    v = PointExamples.points_to_relative(
                        all_tracks[ongoing_idx[i]][-2],
                        all_tracks[ongoing_idx[i]][-1]
                        )
                else:
                    v = PointExamples.points_to_relative(
                        all_tracks[ongoing_idx[i]][ongoing_len - 1],
                        all_tracks[ongoing_idx[i]][ongoing_len]
                        )

                v = self.apply_affine(v, nii_aff)
                inner_prod = np.clip(np.inner(mu, v), -1.0, 1.0)

                mu = np.sign(inner_prod)*mu
                scalars["inverted"][ongoing_idx[i]].append(1 if inner_prod < 0 else 0)

                fvm_probab = np.log(self.fvm_probab(v, mu, k) + 10**-12)
                scalars["fvm_probab"][ongoing_idx[i]].append(fvm_probab)

                angle = map_to_90deg_range(np.rad2deg(np.arccos(inner_prod)))
                scalars["angles"][ongoing_idx[i]].append(angle)

            tracks = list(map(lambda x: x[:-1], tracks))
            ongoing_idx = list(filter(lambda i: len(tracks[i]) > 0, ongoing_idx))

            cprint(
                "{:6d} / {} fibers going on.".format(len(ongoing_idx),
                                                     n_tracks),
                "red", "on_grey",
                end="\r", flush=True)

        for i, fvm_probab in enumerate(scalars["fvm_probab"]):
            probab = np.sum(fvm_probab)
            scalars["probab"].append([probab] * track_lengths[i])

        self.assert_equal_length(all_tracks, scalars)

        return scalars, all_tracks, trk_hdr


    def transform(self, X, args=None):
        assert isinstance(X, list)
        assert len(X) == 2

        if args is not None:
            args = vars(args)
        else:
            args = {}

        if is_nifti(X[0]):
            self.save_scalars(nii_file=X[0],
                              trk_file=X[1],
                              **args)
        else:
            self.save_scalars(nii_file=X[1],
                              trk_file=X[0],
                              **args)

    @staticmethod
    def fvm_probab(v, mu, k, eps=10**-12):
        if k < eps:
            return 1 / (4 * np.pi)

        C_k = k / (2 * np.pi * (1 - np.exp(-2 * k)) + eps)
        return C_k * np.exp(k * (np.inner(mu, v) - 1))

    @staticmethod
    def sample_vMF(mu, k):
        """Sampe from the von Mises-Fisher distribution.

        See "Numerically stable sampling of the von Mises Fisher distribution
        onS2 (and other tricks)".
        https://www.mendeley.com/viewer/?fileId=1d3bb1ab-8211-60fb-218c-f11e1638
        0bde&documentId=7eb942de-6dd9-3af7-b36c-8a9c37b6b6a6

        Args:
            mu: Mean of the distribution. Shape (N, 3).
            k: Concentration of the distribution. Shape (N, 3).
        Returns:
            samples: Samples from the specified vMF distribution. Ndarray of
                shape (N, 3), where N is the number of different distributions.
                A row of the matrix of index j is a sample from the vMF with
                mean mu[j] and concentration k[j].
        """
        mu = np.asarray(mu)
        k = np.asarray(k)

        # Assert 2D vectors
        assert len(k.shape) == 2
        assert len(mu.shape) == 2
        # Assert correct axis orientation
        assert k.shape[1] == 1
        assert mu.shape[1] == 3
        # Assert same number of samples
        assert mu.shape[0] == k.shape[0]
        # Assert all unit vectors
        assert np.allclose(
            np.linalg.norm(mu, axis=1),
            np.ones(mu.shape[0])
        )

        n_samples = mu.shape[0]

        V = ProbabilisticTracker._sample_unif_unit_circle(n_samples)
        W = ProbabilisticTracker._sample_W_values(k)

        omega = np.multiply(np.sqrt(1 - np.square(W)), V)
        omega = np.hstack((omega, W))

        # Now apply the rotation to change the mean
        # i.e. rotate from the direction of the z-axis to the mean direction
        reference = np.asarray([[0, 0, 1]] * omega.shape[0])
        rotation = ProbabilisticTracker._rotation_matrices(reference, mu)
        samples = np.matmul(rotation, omega[:, :, np.newaxis])[:, :, 0]
        return samples

    @staticmethod
    def _rotation_matrices(vectors, references):
        """Compute all the rotation matrices from the vectors to the references.

        Args:
            vectors: Array of vectors that have to be rotated to match the
                references.
            references: Array of reference vectors.
        Returns:
            rotations: Array of matrices. Each matrix is the rotation form the
                vector of corresponding index to its reference.
        """
        # TODO: Fix the inefficiency of the for loop to compute the rotation
        # matrices
        rotations_list = []
        for idx in range(vectors.shape[0]):
            rot_mat = ProbabilisticTracker._to_rotation(vectors[idx, :],
                                                        references[idx, :])
            rotations_list.append(rot_mat)
        rotations = np.asarray(rotations_list)
        return np.asarray(rotations)

    @staticmethod
    def _to_corss_skew_symmetric(vec, ref):
        """Finds the skew-symmetric cross-product matrix."""
        v = np.cross(vec, ref)

        cross_mat = np.zeros(shape=(3, 3))
        cross_mat[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]] = [-v[2],
                                                             v[1],
                                                             v[2],
                                                             -v[0],
                                                             -v[1],
                                                             v[0]]
        return cross_mat

    @staticmethod
    def _to_rotation(vec, ref):
        """Compute rotation matrix from vec to ref.

        NOTE: There must be a better way to do this.
        """
        cross = ProbabilisticTracker._to_corss_skew_symmetric(vec, ref)
        c = np.reshape(np.asarray(np.dot(vec, ref)), newshape=-1)
        square = np.dot(cross, cross)
        R = np.eye(3) + cross + square * (1 / (1 + c))
        return R

    @staticmethod
    def _sample_unif_unit_circle(n_samples):
        """Sample form the uniform distribution on the unit circle.

        Args:
            n_samples: Number of samples required.
        Returns:
            samples: (n_samples,2) ndarray.
        """
        angles = np.random.uniform(high=2*np.pi, size=(n_samples, 1))
        samples_on_unit_circle = np.hstack((np.cos(angles), np.sin(angles)))

        return samples_on_unit_circle

    @staticmethod
    def _sample_W_values(k):
        """Sample the values of W."""
        n_samples = k.shape[0]
        unif = np.random.uniform(size=(n_samples, 1))

        return 1 + np.reciprocal(k) * np.log(unif + (1 - unif) * np.exp(-2 * k))
