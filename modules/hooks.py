import json
import os
import numpy as np
import run
import subprocess
from termcolor import cprint

from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.platform import tf_logging as logging
from abc import ABC, abstractmethod
from argparse import Namespace


def write_smt_num(x,
                  y,
                  metric,
                  smt_outcome_path,
                  x_label=""):

    if "sumatra_outcome.json" not in smt_outcome_path:
        smt_outcome_path = os.path.join(smt_outcome_path,
                                        "sumatra_outcome.json")

    if isinstance(x, str):
        x = float(x)

    assert isinstance(x, (int, float))
    assert isinstance(y, (int, float))
    assert isinstance(metric, str)
    if x_label is not None:
        assert isinstance(x_label, str)

    if os.path.exists(smt_outcome_path):
        with open(smt_outcome_path, "r+") as smt_outcome_file:
            smt_outcome = json.load(smt_outcome_file)
            assert isinstance(smt_outcome, dict)
            assert "numeric_outcome" in smt_outcome

            if metric not in smt_outcome["numeric_outcome"]:
                smt_outcome["numeric_outcome"][metric] = {"x": [x],
                                                          "y": [y],
                                                          "x_label": x_label}
            else:
                smt_outcome["numeric_outcome"][metric]["x"].append(x)
                smt_outcome["numeric_outcome"][metric]["y"].append(y)

                if (x_label != "" and
                    smt_outcome["numeric_outcome"][metric]["x_label"] == ""):
                   smt_outcome["numeric_outcome"][metric]["x_label"] = x_label
                elif (x_label != "" and
                      smt_outcome["numeric_outcome"][metric]["x_label"] != ""):
                    assert smt_outcome["numeric_outcome"][metric]["x_label"] == x_label

            smt_outcome_file.seek(0)
            json.dump(smt_outcome, smt_outcome_file)
            smt_outcome_file.truncate()
    else:
        with open(smt_outcome_path, "w") as smt_outcome_file:
            smt_outcome = {
                "text_outcome": "",
                "numeric_outcome": {}
            }

            smt_outcome["numeric_outcome"][metric] = {"x": [x],
                                                      "y": [y],
                                                      "x_label": x_label}

            json.dump(smt_outcome, smt_outcome_file)

def write_smt_txt(outcome,
                  smt_outcome_path,
                  metric=None,
                  inline=False):

    if "sumatra_outcome.json" not in smt_outcome_path:
        smt_outcome_path = os.path.join(smt_outcome_path,
                                        "sumatra_outcome.json")

    if isinstance(outcome, int):
        outcome = str(outcome)
    if isinstance(outcome, float):
        outcome = "{:4.2f}".format(outcome)
    if isinstance(outcome, str):
        outcome = outcome.replace("\n", "| \n")
    assert isinstance(outcome, str)

    if metric is not None:
        assert isinstance(metric, str)

    if os.path.exists(smt_outcome_path):
        with open(smt_outcome_path, "r+") as smt_outcome_file:
            smt_outcome = json.load(smt_outcome_file)
            assert isinstance(smt_outcome, dict)
            assert "text_outcome" in smt_outcome

            empty = True if smt_outcome["text_outcome"] == "" else False

            if inline and not empty:
                if smt_outcome["text_outcome"][-1] != " ":
                    smt_outcome["text_outcome"] += " "
            elif not empty:
                smt_outcome["text_outcome"] += "\n| "

            if metric is None:
                smt_outcome["text_outcome"] += outcome
            else:
                smt_outcome["text_outcome"] += (metric + ": " + outcome)

            smt_outcome_file.seek(0)
            json.dump(smt_outcome, smt_outcome_file)
            smt_outcome_file.truncate()
    else:
        with open(smt_outcome_path, "w") as smt_outcome_file:
            smt_outcome = {
                "text_outcome": "",
                "numeric_outcome": {}
            }

            if metric is not None:
                smt_outcome["text_outcome"] = "| {}: {}".format(metric, outcome)
            else:
                smt_outcome["text_outcome"] = "| {}".format(outcome)

            json.dump(smt_outcome, smt_outcome_file)


def humanize_time(secs):
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02d' % (hours, mins, secs)


def validate_every_n(steps, secs):
    if ((steps is None) and (secs is None)):
        raise ValueError(
            "exactly one of every_n_steps and every_n_secs "
            "must be provided.")
    if steps is not None and steps <= 0:
        raise ValueError("invalid every_n_steps=%s." % steps)


class BaseHook(SessionRunHook, ABC):
    """docstring for BaseHook"""
    def __init__(
            self,
            every_n_steps=None,
            every_n_secs=None):

        super(BaseHook, self).__init__()

        validate_every_n(every_n_steps, every_n_secs)

        self._timer = SecondOrStepTimer(
            every_secs=every_n_secs,
            every_steps=every_n_steps)

    def begin(self):
        self._timer.reset()
        self._iter_count = 0

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._should_trigger = self._timer.should_trigger_for_step(
            self._iter_count)

    def after_run(self, run_context, run_values):
        _ = run_context
        _ = run_values
        if self._should_trigger:
            self._triggered_action()
        self._iter_count += 1

    @abstractmethod
    def _triggered_action(self):
        pass


class LogTotalSteps(BaseHook):
    """docstring for LogTotalSteps"""
    def __init__(
            self,
            batch_size=None,
            train_size=None,
            epochs=None,
            every_n_steps=None,
            every_n_secs=None):

        super(LogTotalSteps, self).__init__(
            every_n_steps,
            every_n_secs)

        self.batch_size = batch_size
        self.train_size = train_size
        self.epochs = epochs

        self.total_steps = self.train_size // self.batch_size * self.epochs

    def _triggered_action(self):
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)
        elapsed_secs, elapsed_steps = self._timer.update_last_triggered_step(
            self._iter_count)

        steps_to_go = self.total_steps - self._iter_count
        if elapsed_steps is not None and elapsed_secs is not None:
            steps_per_sec = elapsed_steps / elapsed_secs
            ETA = steps_to_go / steps_per_sec

            cprint("Steps to go: {:,}/{:,} ({:,} done) ETA: {}".format(
                   steps_to_go, self.total_steps, self._iter_count, humanize_time(ETA)),
                   "yellow", "on_grey", flush=True)
        else:
            cprint("Steps to go: {:,}".format(steps_to_go),
                   "yellow", "on_grey", flush=True)

        np.set_printoptions(**original)


class FiberTrackingHook(BaseHook):
    """docstring for FiberTrackingHook"""
    def __init__(
            self,
            tracker=None,
            every_n_steps=None,
            every_n_secs=None,
            start_at_step=1,
            test_set=None,
            n_fibers=1000,
            step_size=1,
            threshold=0.5,
            min_length=20,
            max_length=300,
            use_mean=False):

        super(FiberTrackingHook, self).__init__(
            every_n_steps,
            every_n_secs)

        self.tracker = tracker
        self.test_set = test_set

        self.args = Namespace(**{
            "n_fibers": n_fibers,
            "step_size": step_size,
            "threshold": threshold,
            "min_length": min_length,
            "max_length": max_length,
            "use_mean": use_mean})

        self.start_at_step = start_at_step

    def _triggered_action(self):
        self._timer.update_last_triggered_step(
            self._iter_count)

        if self._iter_count >= self.start_at_step:
            original = np.get_printoptions()
            np.set_printoptions(suppress=True)
            self.args.file_name = str(self._iter_count)
            cprint("Saving Fibers...", "green", "on_grey", flush=True)
            self.tracker.predict(self.test_set, self.args)
            self.save_png()
            np.set_printoptions(**original)

    def save_png(self):
        """Save png of current tracking"""
        subprocess.run([
            "track_vis",
            os.path.join(
                self.tracker.save_path,
                self.args.file_name + ".trk"),
            "-sc",
            os.path.join(
                self.tracker.save_path,
                "fibers_" + self.args.file_name)])

class TMScoringHook(FiberTrackingHook):
    """docstring for TMScoringHook"""
    def __init__(
            self,
            tracker=None,
            every_n_steps=None,
            every_n_secs=None,
            start_at_step=1,
            test_set=None,
            n_fibers=1000,
            step_size=1,
            threshold=0.5,
            min_length=20,
            max_length=300,
            use_mean=False,
            tm_data_dir=None):

        super(TMScoringHook, self).__init__(
            tracker,
            every_n_steps,
            every_n_secs,
            start_at_step,
            test_set,
            n_fibers,
            step_size,
            threshold,
            min_length,
            max_length,
            use_mean)

        if tm_data_dir is None:
            raise ValueError("The data directory for the tractometer tool must \
                              be specified in the config as 'tm_data_dir'.")
        self.tm_data_dir = os.path.normpath(tm_data_dir)

    def end(self, session):
        self._triggered_action(is_end=True)

    def _triggered_action(self, is_end=False):
        super(TMScoringHook, self)._triggered_action()

        if self._iter_count >= self.start_at_step:
            # Score the produced tracts
            trk_file = os.path.join(self.tracker.save_path,
                                    self.args.file_name + ".trk")
            cprint("Tractometer scoring running",
                   "yellow", "on_grey", flush=True)

            TM_PATH = os.path.join(self.tm_data_dir, "score_tractogram.py")
            SCORING_DATA = os.path.join(self.tm_data_dir, "scoring_data")

            scoring_cmd = "python {command} {tracts} {base} {out}".format(
                command=TM_PATH,
                tracts=trk_file,
                base=SCORING_DATA,
                out=self.tracker.save_path)
            subprocess.run(["bash", "-c", "source activate entrack_tm && {}"
                            .format(scoring_cmd)])

            eval_path = os.path.join(self.tracker.save_path,
                                     "scores",
                                     self.args.file_name + ".json")
            eval_data = json.load(open(eval_path))

            for metric in ["mean_OL", "mean_OR", "VC", "NC", "IC", "VB", "IB", "mean_F1"]:
                write_smt_num(x=self._iter_count,
                              y=eval_data[metric],
                              metric=metric,
                              x_label="steps",
                              smt_outcome_path=self.tracker.save_path)

            if is_end:
                for metric in ["mean_OL", "mean_OR", "VC", "NC", "IC", "VB", "IB", "mean_F1"]:
                    write_smt_txt(eval_data[metric],
                                  self.tracker.save_path,
                                  metric=metric,
                                  inline=True)


class ScalarMarkerHook(BaseHook):
    """docstring for FiberTrackingHook"""
    def __init__(self,
                 tracker=None,
                 trk_file=None,
                 nii_file=None,
                 every_n_steps=None,
                 every_n_secs=None,
                 start_at_step=1,
                 min_pts_per_fiber=2,
                 every_n_fibers=1):

        super(ScalarMarkerHook, self).__init__(
            every_n_steps,
            every_n_secs)

        self.tracker = tracker
        self.start_at_step = start_at_step

        self.trk_file = trk_file
        self.nii_file = nii_file

        self.min_pts_per_fiber = min_pts_per_fiber
        self.every_n_fibers = every_n_fibers

    def _triggered_action(self):

        self._timer.update_last_triggered_step(
            self._iter_count)

        if self._iter_count >= self.start_at_step:
            file_name = str(self._iter_count) + "_scalars.trk"
            self.tracker.save_scalars(self.trk_file,
                                      self.nii_file,
                                      file_name=file_name,
                                      min_pts_per_fiber=self.min_pts_per_fiber,
                                      every_n_fibers=self.every_n_fibers)
