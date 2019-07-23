import nibabel as nib
import numpy as np

from hashlib import sha1
from sklearn.base import BaseEstimator, TransformerMixin
from modules.models.utils import make_train_set, make_test_set
from modules.models.example_loader import PointExamples
from modules.models.input_fn import is_nifti


class DataTransformer(TransformerMixin):
    """docstring for DataTransformer"""
    def __init__(self):
        super(DataTransformer, self).__init__()

    def set_save_path(self, save_path):
        self.save_path = save_path


class TrainDataTransformer(DataTransformer):
    """Convert dwi.nii and fiber.trk into a pickle.pkl

    This is a wrapper for utils.make_train_set, to enable
    data tracking with sumatra.
    
    """
    def __init__(
        self,
        block_size=3,
        samples_percent=1.0,
        n_samples=None,
        min_fiber_length=0,
        n_incoming=1):

        super(TrainDataTransformer, self).__init__()

        self.block_size = block_size
        self.samples_percent = samples_percent
        self.n_samples = n_samples
        self.min_fiber_length = min_fiber_length
        self.n_incoming = n_incoming

    def transform(self, X, y=None):

        assert isinstance(X, list)
        assert len(X) == 2

        if "trk" in X[0] and "nii" in X[1]:
            X = X[::-1]

        make_train_set(
            dwi_file = X[0],
            trk_file = X[1],
            save_path = self.save_path,
            block_size = self.block_size,
            samples_percent = self.samples_percent,
            n_samples = self.n_samples,
            min_fiber_length = self.min_fiber_length,
            n_incoming = self.n_incoming
        )


class TestDataTransformer(DataTransformer):
    """docstring for TestDataTransformer"""

    def transform(self, X, y=None):
        
        assert isinstance(X, list)
        assert len(X) == 2

        if "mask" in X[0]:
            X = X[::-1]

        make_test_set(
            dwi_file=X[0],
            mask_file=X[1],
            save_path=self.save_path)


class LabelTransformer(DataTransformer):
    """docstring for LabelTransformer"""
    def __init__(self, n_incoming):
        super(LabelTransformer, self).__init__()
        self.n_incoming = n_incoming
        
    def transform(self, X):

        if isinstance(X, list):
            X = X[0]

        examples = PointExamples(trk_file=X,
                                 n_incoming=self.n_incoming)

        return {"labels": examples.train_labels,
                "affine": examples.affine,
                "fiber_header": examples.fiber_header}

class ROIGenerator(DataTransformer):
    """docstring for ROIGenerator"""
    def __init__(self, roi_center, roi_size=1):
        super(ROIGenerator, self).__init__()
        self.roi_center = roi_center
        self.roi_size = roi_size

    def transform(self, X):
        assert isinstance(X, list)
        assert len(X) == 1
        X = X[0]
        assert is_nifti(X)

        file = nib.load(X)

        data = file.get_data()
        data = data - data # Turn data into an array of zeros

        roi_center = np.array([int(x) for  x in self.roi_center.split(",")])

        data[roi_center[0] - (self.roi_size // 2) : roi_center[0] + (self.roi_size // 2) + 1,
             roi_center[1] - (self.roi_size // 2) : roi_center[1] + (self.roi_size // 2) + 1,
             roi_center[2] - (self.roi_size // 2) : roi_center[2] + (self.roi_size // 2) + 1] = 1

        if isinstance(file, nib.nifti2.Nifti2Image):
            img = nib.nifti2.Nifti2Image(data, file.affine, header=file.header)
        else:
            img = nib.nifti1.Nifti1Image(data, file.affine, header=file.header)

        return img