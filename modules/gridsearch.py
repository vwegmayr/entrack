from sklearn.model_selection import GridSearchCV
import pandas as pd
import os
from os.path import normpath


class GridSearchCV(GridSearchCV):
    """docstring for GridSearchCV"""
    def __init__(self, est_class, est_params, param_grid, cv=None, n_jobs=1,
                 error_score="raise", save_path=None, **kwargs):
        self.est_class = est_class
        self.est_params = est_params
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.estimator = est_class(est_params)
        self.set_save_path(save_path)
        self.cv = cv
        if cv is not None and type(cv) is not int:
            self.cv_obj = cv["class"](**cv["params"])
        elif type(cv) is int:
            self.cv_obj = cv
        else:
            self.cv_obj = None
        super(GridSearchCV, self).__init__(self.estimator, param_grid,
                                           cv=self.cv_obj,
                                           n_jobs=n_jobs,
                                           refit=True,
                                           error_score=error_score,
                                           return_train_score=False,
                                           **kwargs)

    def fit(self, X, y=None, groups=None, **fit_params):
        super(GridSearchCV, self).fit(X, y, groups, **fit_params)

        if self.save_path is not None:
            print("Best params: {}".format(self.best_params_))
            print(self.save_path)
            data = {
                "best_params_": self.best_params_,
                "mean_test_score": self.cv_results_["mean_test_score"],
                "std_test_score": self.cv_results_["std_test_score"],
            }
            df = pd.DataFrame.from_dict(pd.io.json.json_normalize(data))
            df.to_csv(normpath(os.path.join(
                self.save_path, "GridSearchCV.csv")))

        return self

    def set_save_path(self, save_path):
        self.save_path = save_path

        if hasattr(self.estimator, "set_save_path"):
            self.estimator.set_save_path(save_path)

        if (hasattr(self, "best_estimator_") and
            hasattr(self.best_estimator_, "set_save_path")):  # noqa: E129
            self.best_estimator_.set_save_path(save_path)
