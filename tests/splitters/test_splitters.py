from unittest import TestCase
import pandas as pd
from ec_number_prediction.data_processing_pipeline.split import StratifiedSplit

class TestSplitters(TestCase):

    def test_splitter(self):

        train_dataset = pd.read_csv("/home/jcapela/ec_numbers_prediction/plants_pipeline/data/train_datasets/train_dataset_0.csv")
        test_dataset = pd.read_csv("/home/jcapela/ec_numbers_prediction/plants_pipeline/data/test_datasets/test_dataset_0.csv")

        splitter = StratifiedSplit()
        X_train = train_dataset.iloc[:, :11]
        y_train = train_dataset.iloc[:, 11:]

        X_test = test_dataset.iloc[:, :11]
        y_test = test_dataset.iloc[:, 11:]

        df_with_stats, table_styled = splitter.generate_stats(y_train, y_test)

        X_train, y_train_, x_test, y_test_ = splitter.correct_split(X_train, y_train, X_test, y_test, \
            df_with_stats, validation = False, compensation=50)