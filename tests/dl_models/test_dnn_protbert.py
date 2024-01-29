from unittest import TestCase
from ec_number_prediction.predictions import make_ensemble_prediction, make_predictions_with_model, predict_with_ensemble, predict_with_ensemble_from_fasta, \
    predict_with_model, predict_with_model_from_fasta

class TestProtBERT(TestCase):

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
        
    def test_prot_bert_prediction_not_all_data(self):
        predict_with_model(pipeline="ProtBERT trial 2 train plus validation",
                            dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                            output_path="predictions_prot_bert_no_all_data.csv",
                            ids_field="id",
                            all_data=False,
                            sequences_field="sequence",
                            device="cuda:3")
