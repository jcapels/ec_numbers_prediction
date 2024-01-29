import unittest

from ec_number_prediction.predictions import make_blast_prediction, predict_with_blast, predict_with_blast_from_fasta
from ec_number_prediction._utils import _download_blast_database_to_cache


class TestBlast(unittest.TestCase):

    def test_download_blast_database_to_cache(self):
        _download_blast_database_to_cache("BLAST all data")

    def test_make_predictions(self):
        make_blast_prediction("/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                              "sequence",
                              "id",
                              "/home/jcapela//ec_numbers_prediction/data/all_data/",
                              "all_data_database",
                              "test_blast_predictions.csv",
                              True)
    
    def test_predict_with_blast(self):
        predict_with_blast(database_name="BLAST all data",
                            dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                            output_path="test_blast_predictions.csv",
                            ids_field="id",
                            sequences_field="sequence")
    
    def test_predict_with_blast_train_plus_validation(self):
        predict_with_blast(database_name="BLAST train plus validation",
                            dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                            output_path="test_blast_predictions.csv",
                            ids_field="id",
                            sequences_field="sequence")
        
    def test_predict_with_blast_from_fasta(self):
        predict_with_blast_from_fasta(database_name="BLAST all data",
                            fasta_path="/home/jcapela/ec_numbers_prediction/data/test.fasta",
                            output_path="test_blast_predictions.csv")
    
    def test_with_blast_plants(self):
        make_blast_prediction("/home/jcapela/ec_numbers_prediction/data/plants_test_data.csv",
                                "sequence",
                                "id",
                                "/home/jcapela/ec_numbers_prediction/data/all_data/",
                                "all_data_database",
                                "test_blast_predictions.csv",
                                True)