from unittest import TestCase
from ec_number_prediction.predictions import make_ensemble_prediction, make_predictions_with_model, \
    predict_with_ensemble, predict_with_ensemble_from_fasta, \
    predict_with_model, predict_with_model_from_fasta


class TestESM1b(TestCase):

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

    def test_esm1b_prediction_not_all_data(self):
        predict_with_model(pipeline="DNN ESM1b trial 4 train plus validation",
                           dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                           output_path="predictions_prot_bert_no_all_data.csv",
                           ids_field="id",
                           all_data=False,
                           sequences_field="sequence",
                           device="cuda:3")
