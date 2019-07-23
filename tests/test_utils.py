import os
import unittest
import numpy as np
import subprocess
import yaml

from sklearn.externals import joblib
from shutil import rmtree
from modules.models import utils


PATHS = None


def setUpModule():
    global PATHS
    if not os.path.exists("tests/data"):
        os.mkdir("tests/data")
    PATHS = utils.make_rand_sets("tests/data")


def tearDownModule():
    if os.path.exists("tests/data"):     
        rmtree("tests/data")


class Test_make_data_sets(unittest.TestCase):
    global PATHS

    def test_if_pkls_are_created(self):
        for mode in ["train", "test"]:
            for _, file in PATHS[mode].items():
                self.assertTrue(os.path.exists(file))

    def test_train_shapes(self):
        X = joblib.load(PATHS["train"]["X"])
        self.assertEqual(X["blocks"].shape, (100, 3, 3, 3, 15))
        self.assertEqual(X["incoming"].shape, (100, 9))

        y = joblib.load(PATHS["train"]["y"])
        self.assertEqual(y.shape, (100, 3))

    def test_test_shapes(self):
        X = joblib.load(PATHS["test"]["X"])
        self.assertEqual(X["dwi"].shape, (10, 10, 10, 15))
        self.assertEqual(X["mask"].shape, (10, 10, 10))


class Test_Pipeline(unittest.TestCase):
    global PATHS

    @classmethod
    def setUpClass(self):
        config = {
            "class": "modules.models.trackers.SimpleTracker",
            "params": {
                "input_fn_config": {
                    "batch_size": 512,
                    "num_epochs": 3,
                    "shuffle": True,
                    "queue_capacity": 10000,
                    "num_threads": 1
                },
                "config": {
                    "save_summary_steps": 10,
                    "tf_random_seed": 42,
                    "save_checkpoints_steps": 100,
                    "keep_checkpoint_max": 5
                },
                "params": {
                    "learning_rate": 0.0001,
                    "layers": {
                        "dense": {
                            "units": 2048,
                            "activation": "tensorflow.nn.relu"
                        },
                        "dense": {
                            "units": 3,
                            "activation": "tensorflow.identity"
                        },
                        "dropout": {
                          "rate": 0.1
                        }
                    }
                }
            }
        }

        with open("tests/data/.config.yaml", "w") as file:
            yaml.dump(config, file, default_flow_style=False)

    @classmethod
    def tearDownClass(self):
        if (hasattr(self, "save_path") and 
        os.path.exists(self.save_path)):
            rmtree(self.save_path)

    def test_training(self):

        cmd = [
            "export CUDA_VISIBLE_DEVICES=0;",
            "python run.py -C tests/data/.config.yaml",
            "-X", PATHS["train"]["X"],
            "-y", PATHS["train"]["y"],
            "-a fit"
        ]
        cmd = " ".join(cmd)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            shell=True)
        stdout, stderr = proc.communicate()
        returncode = proc.returncode

        self.assertEqual(returncode, 0)

        self.save_path = stdout.split("\n")[-2]


class Test_aff_to_rot(unittest.TestCase):

    def test_if_rotation_is_correct(self):
        # 90 deg rotation
        aff = np.asarray(
            [[0, -2, 0, 10],
             [1,  0, 0, 20],
             [0,  0, 3, 30],
             [0,  0, 0, 1]
            ]
        )
        rot = utils.aff_to_rot(aff)

        self.assertEqual(rot.tolist(), [[0, -1, 0], [1, 0, 0], [0, 0, 1]])


if __name__ == '__main__':
    unittest.main()
