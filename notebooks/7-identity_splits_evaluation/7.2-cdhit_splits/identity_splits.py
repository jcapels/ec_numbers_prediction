
from ec_number_prediction.data_processing_pipeline.cdhit_clusters import ClustersIdentifier
import pandas as pd
import numpy as np

from ec_number_prediction.data_processing_pipeline.monte_carlo_folds import MonteCarloFoldSplit
import os


def perform_splits_based_on_identity(clusters_folder, dataset_path, final_datasets_folder):

    os.makedirs(final_datasets_folder, exist_ok=True)
    
    for identity in [90, 80, 70, 60, 50, 40]:

        identity_folder = os.path.join(final_datasets_folder, f"identity_splits_{identity}")
        os.makedirs(identity_folder, exist_ok=True)

        clusters = ClustersIdentifier.from_files(identity_threshold=identity, folder=clusters_folder, filename='all_sequences')

        merged_dataset = pd.read_csv(dataset_path)

        representatives = []
        for cluster in clusters.cluster_to_members:
            element = np.random.choice(np.array(clusters.cluster_to_members[cluster].members), size=1)
            representatives.append(element[0])

        representatives_dataset = merged_dataset[merged_dataset["accession"].isin(representatives)]

        X = representatives_dataset.loc[:, "accession"]
        y = representatives_dataset.iloc[:, 8:]
        y = y.astype(float).astype(int)

        folds = MonteCarloFoldSplit().apply_stratification_sklearn(X, y, test_size=0.2, n_splits=5)
        rest_of_train_datasets = []
        rest_of_test_datasets = []

        for fold in folds:
            X_train, y_train, X_test, y_test = fold
            rest_of_train_dataset = []
            rest_of_test_dataset = []
            for accession in X_train:
                cluster = clusters.get_cluster_by_member(accession).members
                rest_of_train_dataset.extend(cluster)

            for accession in X_test:
                cluster = clusters.get_cluster_by_member(accession).members
                rest_of_test_dataset.extend(cluster)

            rest_of_train_datasets.append(rest_of_train_dataset)
            rest_of_test_datasets.append(rest_of_test_dataset)

        train_datasets = []
        test_datasets = []
        for i, fold in enumerate(folds): 
            train = merged_dataset[merged_dataset["accession"].isin(rest_of_train_datasets[i])]
            test = merged_dataset[merged_dataset["accession"].isin(rest_of_test_datasets[i])]

            train_datasets.append(train)
            test_datasets.append(test)

            X_train = train.loc[:, "accession"]
            y_train = train.iloc[:, 8:]
            y_train = y_train.astype(float).astype(int)

            X_test = test.loc[:, "accession"]
            y_test = test.iloc[:, 8:]
            y_test = y_test.astype(float).astype(int)

            _, table_styled = MonteCarloFoldSplit().generate_stats(y_train, y_test)
            table_styled.to_html(os.path.join(identity_folder, f"stats_fold_{i}.html"))

            X_train.to_csv(os.path.join(identity_folder, f"train_indexes_{i}.csv"), index=False)

            X_test.to_csv(os.path.join(identity_folder, f"test_indexes_{i}.csv"), index=False)

if __name__ == "__main__":
    perform_splits_based_on_identity("./data/clusters/", "../data/merged_dataset.csv", "./identity_splits/")


