import unittest

from ec_number_prediction.predictions import make_blast_prediction


class TestBlast(unittest.TestCase):

    def test_make_predictions(self):
        make_blast_prediction("/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                              "sequence",
                              "id",
                              "/home/jcapela//ec_numbers_prediction/data/all_data/",
                              "all_data_database",
                              "test_blast_predictions.csv",
                              True)
