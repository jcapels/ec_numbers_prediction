

from unittest import TestCase

import pandas as pd

from ec_number_prediction.data_processing_pipeline.monte_carlo_folds import MonteCarloFoldSplit
from tests import TEST_DIR 

import os

class TestMonteCarloFolds(TestCase):

    def setUp(self):
        
        self.data = os.path.join(TEST_DIR, "data", "test_sample.csv")

    def test_monte_carlo_splits(self):

        MonteCarloFoldSplit(input_file=self.data).run()