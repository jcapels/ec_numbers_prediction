import time
import tracemalloc
import unittest

from ec_number_prediction.predictions import make_ensemble_prediction, make_predictions_with_model, predict_with_ensemble, predict_with_ensemble_from_fasta, \
    predict_with_model, predict_with_model_from_fasta, make_blast_predictions_from_fasta_file
from ec_number_prediction._utils import _download_pipeline_to_cache
from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder


class TestPipelines(unittest.TestCase):

    def test_download_pipeline_to_cache(self):
        _download_pipeline_to_cache("DNN ProtBERT all data")

    def test_predict_with_ensemble(self):
        predict_with_ensemble(dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                            output_path="predictions_ensemble.csv",
                            ids_field="id",
                            sequences_field="sequence",
                            device="cuda:3")
        
    def test_make_ensemble_prediction(self):
        make_ensemble_prediction(pipelines = ["/home/jcapela/ec_numbers_prediction/models/DNN_ESM1b_all_data",
                                                    "/home/jcapela/ec_numbers_prediction/models/DNN_ProtBERT_all_data",
                                                    "/home/jcapela/ec_numbers_prediction/models/DNN_ESM2_3B_all_data"],
                                    all_data=True,
                                    ids_field="id",
                                    sequences_field="sequence",
                                    dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                                    output_path="predictions_ensemble.csv",
                                    device="cpu", 
                                    blast_database_folder_path="/home/jcapela/ec_numbers_prediction/data/all_data/", 
                                    blast_database = "all_data_database")
        
    def test_make_ensemble_prediction_plants(self):
        make_ensemble_prediction(pipelines = ["/home/jcapela/ec_numbers_prediction/models/DNN_ESM1b_all_data",
                                                    "/home/jcapela/ec_numbers_prediction/models/DNN_ProtBERT_all_data",
                                                    "/home/jcapela/ec_numbers_prediction/models/DNN_ESM2_3B_all_data"],
                                    all_data=True,
                                    ids_field="id",
                                    sequences_field="sequence",
                                    dataset_path="/home/jcapela/ec_numbers_prediction/data/plants_test_data.csv",
                                    output_path="predictions_ensemble_plants_test_data.csv",
                                    device="cpu", 
                                    blast_database_folder_path="/home/jcapela/ec_numbers_prediction/data/all_data/", 
                                    blast_database = "all_data_database")
        
    def test_all_models(self):
        make_predictions_with_model(pipeline_path="/home/jcapela/ec_numbers_prediction"
                                                  "/models/DNN_ESM2_3B_all_data",
                                    all_data=True,
                                    ids_field="id",
                                    sequences_field="sequence",
                                    dataset_path="/home/jcapela/ec_numbers_prediction/data/plants_test_data.csv",
                                    output_path="predictions_esm2_3b.csv",
                                    device="cpu")
        
        make_predictions_with_model(pipeline_path="/home/jcapela/ec_numbers_prediction"
                                                    "/models/DNN_ProtBERT_all_data",
                                        all_data=True,
                                        ids_field="id",
                                        sequences_field="sequence",
                                        dataset_path="/home/jcapela/ec_numbers_prediction/data/plants_test_data.csv",
                                        output_path="predictions_prot_bert.csv",
                                        device="cpu")
        
        make_predictions_with_model(pipeline_path="/home/jcapela/ec_numbers_prediction"
                                                    "/models/DNN_ESM1b_all_data",
                                        all_data=True,
                                        ids_field="id",
                                        sequences_field="sequence",
                                        dataset_path="/home/jcapela/ec_numbers_prediction/data/plants_test_data.csv",
                                        output_path="predictions_esm1b.csv",
                                        device="cpu")

    def test_predict_with_model(self):

        # predict_with_model_from_fasta(pipeline="DNN ESM2 3B all data",
        #                     fasta_path="/home/jcapela/ec_numbers_prediction/data/plants_sm_sequences.fasta",
        #                     output_path="predictions_esm2_3b.csv",
        #                     device="cuda", num_gpus=4)
        
        # predict_with_model_from_fasta(pipeline="DNN ProtBERT all data",
        #                     fasta_path="/home/jcapela/ec_numbers_prediction/data/plants_sm_sequences.fasta",
        #                     output_path="predictions_prot_bert.csv",
        #                     device="cuda")

        predict_with_model_from_fasta(pipeline="DNN ESM1b all data",
                            fasta_path="/home/jcapela/ec_numbers_prediction/data/plants_sm_sequences.fasta",
                            output_path="predictions_esm1b.csv",
                            device="cuda")
        
    def test_predict_with_ensemble_from_fasta(self):

        tracemalloc.start()
        start = time.time()

        # predict_with_model_from_fasta(pipeline="DNN ProtBERT all data",
        #                     fasta_path="/home/jcapela/ec_numbers_prediction/data/plants_sm_sequences.fasta",
        #                     output_path="predictions_protbert.csv",
        #                     device="cuda", num_gpus=4)

        # predict_with_model_from_fasta(pipeline="DNN ESM1b all data",
        #                     fasta_path="/home/jcapela/ec_numbers_prediction/data/plants_sm_sequences.fasta",
        #                     output_path="predictions_esm1b.csv",
        #                     device="cuda", num_gpus=4)

        # predict_with_model_from_fasta(pipeline="DNN ESM2 3B all data",
        #                     fasta_path="/home/jcapela/ec_numbers_prediction/data/plants_sm_sequences.fasta",
        #                     output_path="predictions_esm2_3b.csv",
        #                     device="cuda", num_gpus=4)
        
        # predict_with_model_from_fasta(pipeline="DNN ESM2 3B all data",
        #                     fasta_path="/home/jcapela/ec_numbers_prediction/data/test_2.fasta",
        #                     output_path="predictions_esm2_3b.csv",
        #                     device="cpu")
        
        predict_with_ensemble_from_fasta(fasta_path="/home/jcapela/ec_numbers_prediction/data/test_2.fasta",
                            output_path="predictions_ensemble.csv",
                            device="cpu")
        
        # predict_with_ensemble_from_fasta(fasta_path="/home/jcapela/ec_numbers_prediction/data/plants_sm_sequences.fasta",
        #                     output_path="predictions_ensemble.csv",
        #                     device="cuda", num_gpus=4)

        # end = time.time()
        # print("Time: ", end - start)

    def test_make_blast_predictions(self):
        make_blast_predictions_from_fasta_file(fasta_path="/home/jcapela/ec_numbers_prediction/data/plants_sm_sequences.fasta",
                                                output_path="predictions_blast.csv")
        
        
