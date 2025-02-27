import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from plants_sm.io.pickle import read_pickle, write_pickle
from tqdm import tqdm


def _get_results_for_blast(folder, tool_name, ground_truth, i):
    test_tool_prediction = pd.read_csv(f"predictions/{folder}/test_{tool_name}_predictions_{i}.csv")
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


def get_results_and_write_to_df():
    import os
    import pandas as pd
    from tqdm import tqdm

    # Create a DataFrame to store the results
    metrics_df = pd.DataFrame(columns=["tool_name", "fold", "label", "precision", "recall", "f1_score"])

    folders = os.listdir("predictions")

    for folder in tqdm(folders):
        tool_name = folder.replace("_predictions", "")
        tool_names = []
        folds = []
        labels_list = []
        precisions = []
        recalls = []
        f1_scores = []
        identities = []
        for i in range(5):
            # Load ground truth and extract labels
            ground_truth = pd.read_csv(f"monte_carlo_splits/test_{i}.csv")
            y_true = ground_truth.iloc[:, 8:].to_numpy()
            labels = ground_truth.iloc[:, 8:].columns  # Get the labels

            # Load predictions
            if "blast" in folder:
                test_tool_predictions = _get_results_for_blast(folder, tool_name, ground_truth, i)
            else:
                test_tool_predictions = read_pickle(f"predictions/{folder}/test_{i}_{tool_name}_predictions.pkl")
                test_tool_predictions = (test_tool_predictions >= 0.5).astype(int)
            # Calculate metrics for each label
            f1_scores_, precision_scores_, recall_scores_ = _get_results_per_label(y_true, test_tool_predictions)

            labels_list.extend(labels)
            folds.extend([i]*len(labels))
            tool_names.extend([tool_name]*len(labels))
            precisions.extend(precision_scores_)
            recalls.extend(recall_scores_)
            f1_scores.extend(f1_scores_)
            identities.extend([90]*len(labels))

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

    # Save DataFrame to a CSV file
    metrics_df.to_csv("metrics_with_labels_summary.csv", index=False)

    print("Metrics saved to metrics_with_labels_summary.csv")

def get_results_for_ensembles():
    import os
    import pandas as pd
    from tqdm import tqdm

    # Create a DataFrame to store the results
    if os.path.exists("metrics_with_labels_summary.csv"):
        metrics_df = pd.read_csv(f"metrics_with_labels_summary.csv")
    else:
        metrics_df = pd.DataFrame(columns=["tool_name", "fold", "label", "precision", "recall", "f1_score", "identity"])


    metrics_df = pd.DataFrame(columns=["tool_name", "fold", "label", "precision", "recall", "f1_score"])

    folders = ["models_blast_predictions", "models_ensemble_predictions"]

    for folder in tqdm(folders):
        tool_name = folder.replace("_predictions", "")
        tool_names = []
        folds = []
        labels_list = []
        precisions = []
        recalls = []
        f1_scores = []
        identities = []
        for i in range(5):
            # Load ground truth and extract labels
            ground_truth = pd.read_csv(f"monte_carlo_splits/test_{i}.csv")
            y_true = ground_truth.iloc[:, 8:].to_numpy()
            labels = ground_truth.iloc[:, 8:].columns  # Get the labels

            # Load predictions
            test_tool_predictions = read_pickle(f"predictions/{folder}/test_{i}_{tool_name}_predictions.pkl")
            # Calculate metrics for each label
            f1_scores_, precision_scores_, recall_scores_ = _get_results_per_label(y_true, test_tool_predictions)

            labels_list.extend(labels)
            folds.extend([i]*len(labels))
            tool_names.extend([tool_name]*len(labels))
            precisions.extend(precision_scores_)
            recalls.extend(recall_scores_)
            f1_scores.extend(f1_scores_)
            identities.extend([90]*len(labels))

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

    # Save DataFrame to a CSV file
    metrics_df.to_csv("metrics_with_labels_summary.csv", index=False)

    print("Metrics saved to metrics_with_labels_summary.csv")


get_results_and_write_to_df()
get_results_for_ensembles()