import unittest

from ec_number_prediction.predictions import make_ensemble_prediction, make_predictions_with_model, predict_with_ensemble, predict_with_ensemble_from_fasta, \
    predict_with_model, predict_with_model_from_fasta
from ec_number_prediction._utils import _download_pipeline_to_cache
from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder


class TestPipelines(unittest.TestCase):

    def test_download_pipeline_to_cache(self):
        _download_pipeline_to_cache("DNN ProtBERT all data")

    def test_pipeline_prediction_esm1b(self):
        make_predictions_with_model(pipeline_path="/home/jcapela/ec_numbers_prediction"
                                                  "/models/DNN_ESM1b_all_data",
                                    all_data=True,
                                    ids_field="id",
                                    sequences_field="sequence",
                                    dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                                    output_path="predictions_esm1b.csv",
                                    device="cuda:3")
        
    def test_pipeline_prediction_esm1b_no_all_data(self):
        make_predictions_with_model(pipeline_path="/home/jcapela/ec_numbers_prediction"
                                                  "/models/DNN_ESM1b_trial_4_train_plus_validation",
                                    all_data=False,
                                    ids_field="id",
                                    sequences_field="sequence",
                                    dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                                    output_path="predictions_esm1b_no_all_data.csv",
                                    device="cpu")
        
    def test_prot_bert_prediction_all_data(self):
        make_predictions_with_model(pipeline_path="/home/jcapela/ec_numbers_prediction"
                                                  "/models/DNN_ProtBERT_all_data",
                                    all_data=True,
                                    ids_field="id",
                                    sequences_field="sequence",
                                    dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                                    output_path="predictions_prot_bert.csv",
                                    device="cuda:3")
        
    def test_prot_bert_prediction_from_fasta(self):
        predict_with_model_from_fasta(pipeline="DNN ProtBERT all data",
                                        fasta_path="/home/jcapela/ec_numbers_prediction/data/test.fasta",
                                        output_path="predictions_prot_bert.csv",
                                        device="cuda:3")
        
    def test_predict_with_prot_bert(self):
        predict_with_model(pipeline="DNN ProtBERT all data",
                            dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                            output_path="predictions_prot_bert.csv",
                            ids_field="id",
                            sequences_field="sequence",
                            device="cuda:1")

    def test_predict_with_ensemble(self):
        predict_with_ensemble(dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                            output_path="predictions_ensemble.csv",
                            ids_field="id",
                            sequences_field="sequence",
                            device="cuda:3")
        
    def test_dnn_esm2_3b_prediction_all_data(self):
        make_predictions_with_model(pipeline_path="/home/jcapela/ec_numbers_prediction"
                                                  "/models/DNN_ESM2_3B_all_data",
                                    all_data=True,
                                    ids_field="id",
                                    sequences_field="sequence",
                                    dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                                    output_path="predictions_esm2_3b.csv",
                                    device="cpu")
    
    def test_dnn_esm2_3b_prediction_all_data_from_cache_folder(self):
        predict_with_model(pipeline="DNN ESM2 3B all data",
                            dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                            output_path="predictions_esm2_3b.csv",
                            ids_field="id",
                            sequences_field="sequence",
                            device="cuda:3")
        
    def test_dnn_esm2_3b_prediction_not_all_data(self):
        make_predictions_with_model(pipeline_path="/home/jcapela/ec_numbers_prediction"
                                                  "/models/DNN_ESM2_3B_trial_2_train_plus_validation",
                                    all_data=False,
                                    ids_field="id",
                                    sequences_field="sequence",
                                    dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                                    output_path="predictions_esm2_3b_no_all_data.csv",
                                    device="cpu")
        
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
        
    def test_predict_with_ensemble_from_fasta(self):
        predict_with_ensemble_from_fasta(fasta_path="/home/jcapela/ec_numbers_prediction/data/test.fasta",
                            output_path="predictions_ensemble.csv",
                            device="cuda:3")
        
        
