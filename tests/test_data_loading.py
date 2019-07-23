import unittest
import numpy as np
from run import load_data

class TestDataLoading(unittest.TestCase):

	def test_file_list(self):
		data = load_data(["fibers.trk", "data.nii"])
		self.assertEqual(data, ["fibers.trk", "data.nii"])

	def test_single_file(self):
		data = load_data(["fibers.trk"])
		self.assertEqual(data, ["fibers.trk"])