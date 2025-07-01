
from sklearn.metrics import f1_score, precision_score, recall_score
from plants_sm.io.pickle import write_pickle
from tqdm import tqdm


def compute_f1_scores_per_ec(level):
    # define the directory where the data is
    data_path = "/home/jcapela/ec_number_prediction_version_2/ec_numbers_prediction/required_data_ec_number_paper/"

    from plants_sm.io.pickle import read_pickle
    import os

    esm1b_predictions = read_pickle(os.path.join(data_path, "predictions", "predictions_esm1b.pkl"))
    esm2_3b_predictions = read_pickle(os.path.join(data_path, "predictions", "predictions_esm2_3b.pkl"))
    prot_bert_predictions = read_pickle(os.path.join(data_path, "predictions", "predictions_prot_bert.pkl"))
    deep_ec_predictions = read_pickle(os.path.join(data_path, "predictions", "predictions_DeepEC.pkl"))
    DSPACE_predictions = read_pickle(os.path.join(data_path, "predictions", "predictions_DSPACE.pkl"))
    models_voting_blast_predictions = read_pickle(os.path.join(data_path, "predictions", "predictions_models_voting_blast.pkl"))
    models_voting_predictions = read_pickle(os.path.join(data_path, "predictions", "predictions_models_voting.pkl"))

    import pandas as pd

    test = pd.read_csv(os.path.join(data_path, "data", "test.csv"))

    test_blast = pd.read_csv(os.path.join(data_path, "test_blast_predictions_right_format.csv"))

    test_blast.drop_duplicates(subset=["qseqid"], inplace=True)
    # Create a new column with the custom order as a categorical type
    test_blast['CustomOrder'] = pd.Categorical(test_blast['qseqid'], categories=test["accession"], ordered=True)
    test_blast.sort_values('CustomOrder', inplace=True)
    test_blast.drop(columns=["CustomOrder"], inplace=True)
    test_blast.reset_index(drop=True, inplace=True)
    test_blast_predictions = test_blast.iloc[:, 6:].to_numpy()

    from collections import defaultdict


    labels = list(test.columns[8:])

    # Dictionary to hold grouped EC numbers
    level_2 = defaultdict(list)
    level_3 = defaultdict(list)
    level_4 = defaultdict(list)

    # Group by the first EC number level and filter by the number of elements
    for i, ec in enumerate(labels):
        split_ec = ec.split('.')
        first_level = split_ec[0]  # Get the first part (EC level 1)
        num_elements = len(split_ec)  # Get the number of elements (levels)
        
        # Only add EC numbers with 2, 3, or 4 elements
        if num_elements==2:
            level_2[first_level].append(i)
        elif num_elements==3:
            level_3[first_level].append(i)
        elif num_elements==4:
            level_4[first_level].append(i)
            

    # Convert defaultdict to a normal dict for easy reading
    if int(level) == 2:
        grouped_level = dict(level_2)
    elif int(level) == 3:
        grouped_level = dict(level_3)
    elif int(level) == 4:
        grouped_level = dict(level_4)
    else: 
        raise ValueError

    y_true_ = test.iloc[:, 8:].to_numpy()
    y_true_ecs = list(test.iloc[:, 8:].columns)

    from sklearn.metrics import f1_score

    # Assuming you have these available
    # y_true_ = test.iloc[:, 8:].to_numpy()  # Ground truth from your test set
    # y_true_ecs = list(test.iloc[:, 8:].columns)  # Column names for EC numbers
    # esm1b_predictions = model predictions for ESM1b

    # Placeholder for multiple models' predictions
    # Replace these with actual model predictions
    model_predictions = {
        "BLASTp": test_blast_predictions,
        "DNN ESM1b": esm1b_predictions,  # Example of one model's predictions
        "DNN ESM2 3B": esm2_3b_predictions,  # Replace with actual model predictions
        "DNN ProtBERT": prot_bert_predictions,
        "D-SPACE EC": DSPACE_predictions,
        "DeepEC CNN3": deep_ec_predictions,
        "Models ensemble": models_voting_predictions,
        "Models + BLASTp": models_voting_blast_predictions
        
    }

    # Initialize DataFrame to store results
    results_dataframe = pd.DataFrame(columns=["Model", "Level", "EC", "F1 score"])

    # Loop through each EC level (1 to 7)
    for level_, ec_indices in tqdm(grouped_level.items()):
        
        # Loop through each model's predictions
        for model_name, y_pred_all in model_predictions.items():
            
            model = []
            f1_scores = []
            ecs = []
            ec_level = []
            
            # Loop through the EC indices for the current level
            for ec_index in ec_indices:
                y_true = y_true_[:, ec_index]  # Ground truth for the EC
                y_pred = y_pred_all[:, ec_index]  # Model prediction for the EC
                
                # Store EC name and calculate F1 score
                ecs.append(y_true_ecs[ec_index])
                ec_f1_score = f1_score(y_true, y_pred)
                model.append(model_name)
                f1_scores.append(ec_f1_score)
                ec_level.append(level_)  # Store the EC level (1, 2, etc.)
            
            # Create a DataFrame for the current results
            results_dataframe_ = pd.DataFrame({
                "Model": model,
                "Level": ec_level,
                "EC": ecs,
                "F1 score": f1_scores
            })
            
            # Concatenate the results
            results_dataframe = pd.concat([results_dataframe, results_dataframe_], ignore_index=True)

            results_dataframe.to_csv(f"f1_score_per_ec_results_{level}.csv")

if __name__ == "__main__":
    import sys

    level = sys.argv[1]
    compute_f1_scores_per_ec(level)