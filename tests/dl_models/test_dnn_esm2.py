import profile
import time
from unittest import TestCase
from ec_number_prediction.predictions import make_ensemble_prediction, make_predictions_with_model, predict_with_ensemble, predict_with_ensemble_from_fasta, \
    predict_with_model, predict_with_model_from_fasta

class TestESM2(TestCase):

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
        import tracemalloc
        tracemalloc.start()
        start = time.time()
        predict_with_model(pipeline="DNN ESM2 3B all data",
                            dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                            output_path="predictions_esm2_3b.csv",
                            ids_field="id",
                            sequences_field="sequence",
                            device="cuda", num_gpus=2)
        end = time.time()
        print("Time spent: ", end - start)
        print("Memory needed: ", tracemalloc.get_traced_memory())
        tracemalloc.stop()
        
    def test_dnn_esm2_3b_prediction_not_all_data(self):
        make_predictions_with_model(pipeline_path="/home/jcapela/ec_numbers_prediction"
                                                  "/models/DNN_ESM2_3B_trial_2_train_plus_validation",
                                    all_data=False,
                                    ids_field="id",
                                    sequences_field="sequence",
                                    dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                                    output_path="predictions_esm2_3b_no_all_data.csv",
                                    device="cuda:1")
        
    def test_esm2_3b_prediction_not_all_data(self):
        predict_with_model(pipeline="DNN ESM2 3B trial 2 train plus validation",
                            dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                            output_path="predictions_prot_bert_no_all_data.csv",
                            ids_field="id",
                            all_data=False,
                            sequences_field="sequence",
                            device="cuda:1")
