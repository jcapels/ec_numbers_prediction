import luigi
import numpy as np
import pandas as pd

from ec_number_prediction.data_processing_pipeline.multi_label_binarizer import MultiLabelBinarizer


class UnderrepresentedLabelsRemoval(luigi.Task):

    def requires(self):
        return MultiLabelBinarizer()
    
    def input(self):
        return luigi.LocalTarget('dataset_binarized.csv')

    def output(self):
        return luigi.LocalTarget('dataset_binarized_filtered.csv')

    def run(self):
        dataset = pd.read_csv(self.input().path)
        labels = dataset.iloc[:, 8:]
        rest = dataset.iloc[:, :8]

        sum_of_labels = np.sum(labels)

        final_labels = labels[sum_of_labels[sum_of_labels >= 100].index]
        final_dataset_filtered_labels = pd.concat([rest, final_labels], axis=1)

        final_dataset_filtered_labels.to_csv(self.output().path, index=False)