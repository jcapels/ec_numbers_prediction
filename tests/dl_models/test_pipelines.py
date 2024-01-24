import unittest

from ec_number_prediction.predictions import make_predictions_with_model


class TestPipelines(unittest.TestCase):

    def test_pipeline_prediction(self):
        make_predictions_with_model(pipeline_path="/home/joao/Desktop/PHD/ec_numbers_prediction"
                                                  "/models/DNN_ESM1b_all_data",
                                    all_data=True,
                                    ids_field="id",
                                    sequences_field="sequence",
                                    dataset_path="/home/joao/Desktop/PHD/ec_numbers_prediction/data/test_data.csv",
                                    output_path="predictions_esm1b.csv",
                                    device="cpu")
