import unittest
import numpy as np
import nibabel as nib
from sklearn.externals import joblib
from modules.models.utils import make_hdr
from modules.models.example_loader import PointExamples

class TestMaxEntropyTracker(unittest.TestCase):
	"""docstring for TestMaxEntropyTracker"""
	
	def test_prediction(self):
		
		#model = joblib.load("/local/entrack/data/20180130-104945/"
		#				    "MaxEntropyTracker.pkl")

		model = joblib.load("/local/entrack/data/20180131-103637/"
						    "MaxEntropyTracker.pkl")

		predictor = model.predictor()

		file = "/local/entrack/data/HCP/979984/T1w/Diffusion/FOD/FODl4.nii.gz"

		header = make_hdr(file)

		FODl4=nib.load(file).get_data()

		#incoming = np.array([[[1, 0, 0], [1, 0, 0],
		#					 [1, 0, 0], [1, 0, 0]]])
		#incoming = np.array([[[0, 1, 0], [0, 1, 0],
		#					 [0, 1, 0], [0, 1, 0]]])
		incoming = np.array([[[0, 0, 0], [0, 0, 0],
							 [0, 0, 0], [0, 0, 0]]])
		#incoming = np.array([[[0, 0, 1], [0, 0, 1],
		#					 [0, 0, 1], [0, 0, 1]]])


		blocks = np.array([FODl4[71:74, 93:96, 75:78, :]])

		#blocks = np.array([FODl4[60:63, 86:89, 101:104, :]])

		input = {"incoming": incoming,
				 "blocks": blocks}

		output = predictor(input)

		directions = model.get_directions_from_predictions(output,
			affine=header["vox_to_ras"])

		print(output)
		print(directions)
	

	"""
	def test_affine(self):
		loader = PointExamples(
		    "/local/entrack/data/HCP/979984/T1w/Diffusion/FOD/FODl4.nii.gz",
		    "/local/entrack/data/HCP/978578/T1w/Diffusion/FOD/iFOD2.trk",
		    block_size=3,
		    n_incoming=3,
		    num_eval_examples=0,
		    example_percent=0.25,
		)

		for label in loader.train_labels:
			if np.linalg.norm(label["center"] - [0, -8.5, 23]) < 0.1:
				print(label)
	"""
	"""
	def test_fibers(self):
		trk_file = "/local/entrack/data/HCP/978578/T1w/Diffusion/FOD/iFOD2.trk"
		fibers, fiber_header = nib.trackvis.read(trk_file, points_space="rasmm")

		fibers = [fiber[0] for fiber in fibers]

		for fiber in fibers:
			for idx, point in enumerate(fiber):
				if np.linalg.norm(point - np.array([0, -8.5, 23])) < 0.5:
					print(fiber[idx-1] - point)
					print(fiber[idx+1] - point)
	"""