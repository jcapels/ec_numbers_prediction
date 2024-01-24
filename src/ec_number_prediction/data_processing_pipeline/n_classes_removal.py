import luigi
import pandas as pd

from ec_number_prediction.data_processing_pipeline.underrepresented_labels_removal import UnderrepresentedLabelsRemoval


class NClassesRemoval(luigi.Task):

    def requires(self):
        return UnderrepresentedLabelsRemoval()
    
    def output(self):
        return luigi.LocalTarget('dataset_binarized_filtered_without_n.csv')

    def run(self):
        dataset = pd.read_csv(self.input().path)
        labels = dataset.iloc[:, 8:]
        rest = dataset.iloc[:, :8]

        for label in labels.columns:
            if "n" in label:
                labels.drop(label, axis=1, inplace=True)

        final_dataset_filtered_labels = pd.concat([rest, labels], axis=1)

        final_dataset_filtered_labels.to_csv(self.output().path, index=False)