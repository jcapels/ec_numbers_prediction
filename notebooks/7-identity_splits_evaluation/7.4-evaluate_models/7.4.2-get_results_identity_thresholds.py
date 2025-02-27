import os
from joblib import Parallel, delayed
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from plants_sm.io.pickle import read_pickle, write_pickle
from tqdm import tqdm


def _get_results_for_blast(absolute_path_of_current_folder, tool_name, ground_truth, i):
    test_tool_prediction = pd.read_csv(f"{absolute_path_of_current_folder}/test_{tool_name}_predictions_{i}.csv")
    test_tool_prediction.drop_duplicates(subset=["qseqid"], inplace=True)
    # Create a new column with the custom order as a categorical type
    test_tool_prediction['CustomOrder'] = pd.Categorical(test_tool_prediction['qseqid'], 
                                                            categories=ground_truth["accession"], 
                                                            ordered=True)
    test_tool_prediction.sort_values('CustomOrder', inplace=True)
    test_tool_prediction.drop(columns=["CustomOrder"], inplace=True)
    test_tool_prediction.reset_index(drop=True, inplace=True)
    test_tool_predictions = test_tool_prediction.iloc[:, 6:].to_numpy()

    return test_tool_predictions

def _get_results_per_label(y_true, test_tool_predictions):
    f1_scores_ = [] 
    precision_scores_ = []
    recall_scores_ = []

    for i in range(test_tool_predictions.shape[1]):

        f1_score_ = f1_score(y_true[:, i], test_tool_predictions[:, i])
        f1_scores_.append(f1_score_)

        precision_score_ = precision_score(y_true[:, i], test_tool_predictions[:, i])
        precision_scores_.append(precision_score_)

        recall_score_ = recall_score(y_true[:, i], test_tool_predictions[:, i])
        recall_scores_.append(recall_score_)

    return f1_scores_, precision_scores_, recall_scores_


def get_results_and_write_to_df(base_dir, identity):
    import os
    import pandas as pd
    from tqdm import tqdm

    print(f"IDENTITY: {identity}")

    # Create a DataFrame to store the results
    metrics_df = pd.DataFrame(columns=["tool_name", "fold", "label", "precision", "recall", "f1_score", "identity"])

    # Load merged dataset
    merged_dataset = pd.read_csv(f"{base_dir}/data/merged_dataset.csv")
    folders = os.listdir(os.path.join("identity_splits", f"identity_splits_{identity}", "predictions"))
    
    for folder in tqdm(folders):
        tool_name = folder.replace("_predictions", "")
        absolute_path_of_current_folder = os.path.join("identity_splits",  f"identity_splits_{identity}", "predictions", folder)
        
        tool_names = []
        folds = []
        labels_list = []
        precisions = []
        recalls = []
        f1_scores = []
        identities = []

        for i in range(5):
            # Load ground truth and extract labels
            ground_truth = pd.read_csv(os.path.join("identity_splits",  f"identity_splits_{identity}", f"test_indexes_{i}.csv"))
            train_dataset_indexes = pd.read_csv(os.path.join("identity_splits",  f"identity_splits_{identity}", f"train_indexes_{i}.csv"))

            # Filter the merged dataset by the accessions present in ground truth and train dataset
            ground_truth = merged_dataset[merged_dataset["accession"].isin(list(ground_truth.iloc[:, 0].values))]
            train_dataset = merged_dataset[merged_dataset["accession"].isin(list(train_dataset_indexes.iloc[:, 0].values))]
            train_dataset = train_dataset.iloc[:, 8:]
            test_dataset = ground_truth.iloc[:, 8:]

            # Get column indexes where not all values are zero in both training and test sets
            train_dataset_non_zero_columns = train_dataset.loc[:, (train_dataset != 0).any(axis=0)].columns
            test_dataset_non_zero_columns = test_dataset.loc[:, (test_dataset != 0).any(axis=0)].columns

            # Select columns that are non-zero in both train and test datasets
            common_non_zero_columns = list(train_dataset_non_zero_columns.intersection(test_dataset_non_zero_columns))

            # Filter the ground truth and predictions to only include common non-zero columns
            y_true = ground_truth.loc[:, common_non_zero_columns].to_numpy()
            labels = common_non_zero_columns  # Use only the labels that are non-zero in both datasets

            # Load predictions
            if "blast" in folder:
                test_tool_predictions = _get_results_for_blast(absolute_path_of_current_folder, tool_name, ground_truth, i)
            else:
                test_tool_predictions = read_pickle(f"{absolute_path_of_current_folder}/test_{i}_{tool_name}_predictions.pkl")
                test_tool_predictions = (test_tool_predictions >= 0.5).astype(int)

            non_zero_column_indexes = [train_dataset.columns.get_loc(col) for col in common_non_zero_columns]

            # Filter predictions to only include the same non-zero columns
            test_tool_predictions = test_tool_predictions[:, non_zero_column_indexes]

            # Calculate metrics for each label
            f1_scores_, precision_scores_, recall_scores_ = _get_results_per_label(y_true, test_tool_predictions)

            labels_list.extend(labels)
            folds.extend([i]*len(labels))
            tool_names.extend([tool_name]*len(labels))
            precisions.extend(precision_scores_)
            recalls.extend(recall_scores_)
            f1_scores.extend(f1_scores_)
            identities.extend([identity]*len(labels))

            # Add metrics for each label into the DataFrame
        metrics_df = pd.concat((metrics_df, pd.DataFrame({
            "tool_name": tool_names,
            "fold": folds,
            "label": labels_list,
            "precision": precisions,
            "recall": recalls,
            "f1_score": f1_scores,
            "identity": identities
        })), ignore_index=True)

        # Save the results after each fold to avoid data loss in case of interruption
        metrics_df.to_csv(f"metrics_with_labels_summary_{identity}.csv", index=False)

    # Save final DataFrame to CSV
    metrics_df.to_csv(f"metrics_with_labels_summary_{identity}.csv", index=False)

    print(f"Metrics saved to metrics_with_labels_summary_{identity}.csv")


def compute_for_ensembles(base_dir, identity):
    import os
    import pandas as pd
    from tqdm import tqdm

    print(f"IDENTITY: {identity}")

    # Create a DataFrame to store the results
    if os.path.exists(f"metrics_with_labels_summary_{identity}.csv"):
        metrics_df = pd.read_csv(f"metrics_with_labels_summary_{identity}.csv")
    else:
        metrics_df = pd.DataFrame(columns=["tool_name", "fold", "label", "precision", "recall", "f1_score", "identity"])

    # Load merged dataset
    merged_dataset = pd.read_csv(f"{base_dir}/data/merged_dataset.csv")
    folders = ["models_blast_predictions", "models_ensemble_predictions"]
    
    for folder in tqdm(folders):
        tool_name = folder.replace("_predictions", "")
        absolute_path_of_current_folder = os.path.join("identity_splits",  f"identity_splits_{identity}", "predictions", folder)
        
        tool_names = []
        folds = []
        labels_list = []
        precisions = []
        recalls = []
        f1_scores = []
        identities = []

        for i in range(5):
            # Load ground truth and extract labels
            ground_truth = pd.read_csv(os.path.join("identity_splits",  f"identity_splits_{identity}", f"test_indexes_{i}.csv"))
            train_dataset_indexes = pd.read_csv(os.path.join("identity_splits",  f"identity_splits_{identity}", f"train_indexes_{i}.csv"))

            # Filter the merged dataset by the accessions present in ground truth and train dataset
            ground_truth = merged_dataset[merged_dataset["accession"].isin(list(ground_truth.iloc[:, 0].values))]
            train_dataset = merged_dataset[merged_dataset["accession"].isin(list(train_dataset_indexes.iloc[:, 0].values))]
            train_dataset = train_dataset.iloc[:, 8:]
            test_dataset = ground_truth.iloc[:, 8:]

            # Get column indexes where not all values are zero in both training and test sets
            train_dataset_non_zero_columns = train_dataset.loc[:, (train_dataset != 0).any(axis=0)].columns
            test_dataset_non_zero_columns = test_dataset.loc[:, (test_dataset != 0).any(axis=0)].columns

            # Select columns that are non-zero in both train and test datasets
            common_non_zero_columns = list(train_dataset_non_zero_columns.intersection(test_dataset_non_zero_columns))

            # Filter the ground truth and predictions to only include common non-zero columns
            y_true = ground_truth.loc[:, common_non_zero_columns].to_numpy()
            labels = common_non_zero_columns  # Use only the labels that are non-zero in both datasets

            # Load predictions
            test_tool_predictions = read_pickle(f"{absolute_path_of_current_folder}/test_{i}_{tool_name}_predictions.pkl")

            non_zero_column_indexes = [train_dataset.columns.get_loc(col) for col in common_non_zero_columns]

            # Filter predictions to only include the same non-zero columns
            test_tool_predictions = test_tool_predictions[:, non_zero_column_indexes]

            # Calculate metrics for each label
            f1_scores_, precision_scores_, recall_scores_ = _get_results_per_label(y_true, test_tool_predictions)

            labels_list.extend(labels)
            folds.extend([i]*len(labels))
            tool_names.extend([tool_name]*len(labels))
            precisions.extend(precision_scores_)
            recalls.extend(recall_scores_)
            f1_scores.extend(f1_scores_)
            identities.extend([identity]*len(labels))

            # Add metrics for each label into the DataFrame
        metrics_df = pd.concat((metrics_df, pd.DataFrame({
            "tool_name": tool_names,
            "fold": folds,
            "label": labels_list,
            "precision": precisions,
            "recall": recalls,
            "f1_score": f1_scores,
            "identity": identities
        })), ignore_index=True)

        # Save the results after each fold to avoid data loss in case of interruption
        metrics_df.to_csv(f"metrics_with_labels_summary_{identity}.csv", index=False)

    # Save final DataFrame to CSV
    metrics_df.to_csv(f"metrics_with_labels_summary_{identity}.csv", index=False)

    print(f"Metrics saved to metrics_with_labels_summary_{identity}.csv")

def run_parallel_identities(base_dir:str):

    # # Parallelize the BLAST task for each of the 5 datasets
    # Parallel(n_jobs=5, backend='multiprocessing')(
    #     delayed(get_results_and_write_to_df)(base_dir, i) for i in [80,70,60,50,40]
    # )

    # Parallelize the BLAST task for each of the 5 datasets
    Parallel(n_jobs=5, backend='multiprocessing')(
        delayed(compute_for_ensembles)(base_dir, i) for i in [80, 70, 60, 50, 40]
    )

base_dir = "/home/jcapela/ec_number_prediction_version_2/ec_numbers_prediction/"
run_parallel_identities(base_dir)