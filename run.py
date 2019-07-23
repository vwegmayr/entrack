"""Scikit runner"""
import numpy as np
import nibabel as nib
import argparse
import os
import sys
import pandas as pd
import csv
import time

from sklearn.externals import joblib
from abc import ABC, abstractmethod
from modules.configparse import (ConfigParser,
    parse_more_args, import_obj_from_string)
from pprint import pprint
from os.path import normpath
from inspect import getfullargspec
from shutil import rmtree
from traceback import print_tb
from sumatra.projects import load_project


def conditional_call_to(func, **kwargs):

    allowed_kwargs = {}
    for key, val in kwargs.items():
        if key in getfullargspec(func).args:
            allowed_kwargs[key] = val

    return func(**allowed_kwargs)


def get_loader_from_extension(file_path):
    extension = file_path.split(".")[-1]

    if extension == "npy":
        loader = np.load
    elif extension == "pkl":
        loader = joblib.load
    else:
        loader = None

    return loader


def load(file_path, loader):
    try:
        data = loader(file_path)
    except FileNotFoundError:
        print("{} not found. "
              "Please download data first.".format(file_path))
        exit()
    except TypeError:
        data = file_path

    return data


def load_data(data_path):
    if isinstance(data_path, list):
        if len(data_path) > 1:
            return list(map(load_data, data_path))
        else:
            return [load_data(data_path[0])]

    elif isinstance(data_path, str):
        loader = get_loader_from_extension(data_path)
        return load(data_path, loader)
    elif data_path is None:
        return None
    else:
        raise ValueError("Expected data_path as string "
        "or list of strings.")


class Action(ABC):
    """Abstract Action class

    Args:
        args (Namespace): Parsed arguments
    """
    def __init__(self, args, more_args):
        self.args = args
        self.more_args = more_args
        self._check_action(args.action)

        self.X = load_data(self.args.X)
        if self.X is None:
            print("No data loaded for X")
        self.y = load_data(self.args.y)
        if self.y is None:
            print("No data loaded for y")
            
        self._mk_save_folder()
        self.X_new, self.y_new = None, None

    @abstractmethod
    def _save(self):
        pass

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _check_action(self):
        pass

    def act(self):
        if self.args.debug:
            try:
                self.model = self._load_model()
                getattr(self, self.args.action)()
                self._save()
                print(self.save_path)
            except Exception as err:
                print_tb(err.__traceback__)
                print(err.__class__.__name__)
                print(err.__str__)
            finally:
                rmtree(self.save_path)
        else:
            self.model = self._load_model()
            getattr(self, self.args.action)()
            self._save()
            print(self.save_path)


    def _mk_save_folder(self):
        if self.args.smt_label != "debug":
            self.time_stamp = self.args.smt_label
        else:
            self.time_stamp = (time.strftime(
                "%Y%m%d-%H%M%S",
                time.gmtime()) + "-debug")

        project = load_project()
        path = os.path.join(project.data_store.root, self.time_stamp)
        os.mkdir(os.path.normpath(path))

        self.save_path = path

    def transform(self):
        self.X_new = conditional_call_to(self.model.transform,
                                         X=self.X,
                                         args=self.more_args)


class ConfigAction(Action):
    """Class to handle config file actions

    Args:
        args (Namespace): Parsed arguments
        config (dict): Parsed config file

    """
    def __init__(self, args, config, more_args=None):
        super(ConfigAction, self).__init__(args, more_args)
        self.raw_config = ConfigParser().parse_raw(config)
        self.config = ConfigParser().parse(config)
        self.pprint_config(self.raw_config)
        self.act()

    def fit(self):
        if "args" in getfullargspec(self.model.fit).args:
            self.model.fit(self.X, self.y, args=self.more_args)
        else:
            self.model.fit(self.X, self.y)

    def fit_transform(self):
        self.fit()
        self.transform()

    def fit_predict(self):
        assert isinstance(self.X, list)
        assert len(self.X) >= 2
        X_train = self.X[:-1]
        if len(X_train) == 1:
            X_train = X_train[0]
        X_test = self.X[-1]
        self.X = X_train
        self.fit()
        self._predict(X_test)

    def _predict(self, X):
        if "args" in getfullargspec(self.model.predict).args:
            self.y_new = self.model.predict(X, args=self.more_args)
        else:
            self.y_new = self.model.predict(X)

    def _save(self):
        class_name = self.config["class"].__name__
        model_path = normpath(os.path.join(self.save_path, class_name+".pkl"))
        joblib.dump(self.model, model_path)

        if isinstance(self.X_new, np.ndarray):
            X_path = normpath(os.path.join(
                self.save_path,
                "X_" + self.time_stamp + ".npy"))
            np.save(X_path, self.X_new)
        elif isinstance(self.X_new, dict):
            X_path = normpath(os.path.join(
                self.save_path,
                "X_" + self.time_stamp + ".pkl"))
            joblib.dump(self.X_new, X_path)

    def _load_model(self):
        if "params" in self.config:
            model = self.config["class"](**self.config["params"])
        else:
            model = self.config["class"]()

        if hasattr(model, "set_save_path"):
            model.set_save_path(self.save_path)

        return model

    def _check_action(self, action):
        if action not in ["fit", "fit_transform", "transform", "fit_predict"]:
            raise RuntimeError("Can only run fit or fit_transform from config,"
                               " got {}.".format(action))

    def pprint_config(self, raw_config):
        print("\n=========== Config ===========")
        pprint(raw_config)
        print("==============================\n")
        sys.stdout.flush()


class ModelAction(Action):
    """Class to model actions

    Args:
        args (Namespace): Parsed arguments
    """
    def __init__(self, args, more_args=None):
        super(ModelAction, self).__init__(args, more_args)
        self.act()

    def predict(self):
        if "args" in getfullargspec(self.model.predict).args:
            self.y_new = self.model.predict(self.X, args=self.more_args)
        else:
            self.y_new = self.model.predict(self.X)

    def predict_proba(self):
        # self.y_new = self.model.predict_proba(self.X)
        conditional_call_to(self.model.predict_proba,
                            X=self.X,
                            args=self.more_args)

    def score(self):
        self.model.score(self.X, self.y, args=self.more_args)

    def _save(self):
        y_path = normpath(os.path.join(
            self.save_path, "y_" + self.time_stamp + ".csv"))
        X_path = normpath(os.path.join(
            self.save_path, "X_" + self.time_stamp))

        if self.X_new is not None:
            if isinstance(self.X_new, np.ndarray):
                X_path += ".npy"
                np.save(X_path, self.X_new)
            elif isinstance(self.X_new,
                            (nib.nifti1.Nifti1Image,
                             nib.nifti2.Nifti2Image)):
                X_path += ".nii"
                nib.save(self.X_new, X_path)

        if self.y_new is not None:
            if isinstance(self.y_new, (np.ndarray, list)):
                with open(y_path, "w") as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerow(["ID", "Prediction"])
                    for id, prediction in enumerate(self.y_new):
                        prediction = np.round(prediction, decimals=6)
                        if len(prediction.shape) > 1:
                            prediction = " ".join(prediction.astype("str"))
                        writer.writerow([id+1, prediction])
            elif isinstance(self.y_new, dict):
                df = pd.DataFrame(self.y_new)
                df.index += 1
                df.index.name = "ID"
                df.to_csv(y_path)

    def _load_model(self):
        try:
            model = joblib.load(self.args.model)
        except FileNotFoundError:
            model = import_obj_from_string(self.args.model)(**vars(self.more_args))

        if hasattr(model, "set_save_path"):
            model.set_save_path(self.save_path)

        return model

    def _check_action(self, action):
        if action not in ["transform", "predict", "score", "predict_proba"]:
            raise RuntimeError("Can only run transform, predict, predict_proba"
                               " or score from model, got {}.".format(action))


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description="Scikit runner.")

    arg_parser.add_argument("-C", "--config", help="config file")
    arg_parser.add_argument("-M", "--model", help="model file")

    arg_parser.add_argument("-X", help="Input data", action="append")
    arg_parser.add_argument("-y", help="Input labels", action="append")

    arg_parser.add_argument("-a", "--action", choices=["transform", "predict",
                            "fit", "fit_transform", "score", "predict_proba",
                            "fit_predict"],
                            help="Action to perform.",
                            required=True)

    arg_parser.add_argument("smt_label", nargs="?", default="debug")
    arg_parser.add_argument("--debug", action="store_true")

    args, more_args = arg_parser.parse_known_args()

    more_args = parse_more_args(more_args)

    if args.config is None:
        ModelAction(args, more_args)
    else:
        ConfigAction(args, args.config, more_args)
