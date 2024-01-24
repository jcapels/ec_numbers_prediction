import unittest

from ec_number_prediction.predictions import make_ensemble_prediction, make_predictions_with_model


class TestPipelines(unittest.TestCase):

    def test_pipeline_prediction_esm1b(self):
        make_predictions_with_model(pipeline_path="/home/jcapela/ec_numbers_prediction"
                                                  "/models/DNN_ESM1b_all_data",
                                    all_data=True,
                                    ids_field="id",
                                    sequences_field="sequence",
                                    dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                                    output_path="predictions_esm1b.csv",
                                    device="cpu")
        
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
                                    device="cpu")
        
    def test_dnn_esm2_3b_prediction_all_data(self):
        make_predictions_with_model(pipeline_path="/home/jcapela/ec_numbers_prediction"
                                                  "/models/DNN_ESM2_3B_all_data",
                                    all_data=True,
                                    ids_field="id",
                                    sequences_field="sequence",
                                    dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                                    output_path="predictions_esm2_3b.csv",
                                    device="cpu")
        
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
