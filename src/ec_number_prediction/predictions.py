import os
import os.path
import re
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from plants_sm.alignments.alignment import BLAST
from plants_sm.data_structures.dataset import SingleInputDataset
from plants_sm.io.pickle import read_pickle
from plants_sm.pipeline.pipeline import Pipeline
from plants_sm.utilities.utils import convert_csv_to_fasta

from ec_number_prediction import SRC_PATH
from ec_number_prediction._utils import get_final_labels

def _make_blast_prediction(dataset_path: str, sequences_field: str,
                            ids_field: str, database_folder: str, database_name: str, 
                            binarised=False):
    current_directory = os.getcwd()
    os.chdir(database_folder)
    convert_csv_to_fasta(dataset_path, sequences_field, ids_field, 'temp.fasta')

    blast = BLAST(database_name)
    blast.run('temp.fasta', 'temp_results_file', 1e-5, 1)
    database = pd.read_csv("database.csv")
    blast.associate_to_ec(database, "temp_results_file")
    results = pd.read_csv("temp_results_file")
    if binarised:
        results = results.astype({'EC1': 'str',
                                  'EC2': 'str',
                                  'EC3': 'str',
                                  'EC4': 'str'})
        results = get_final_labels(results)

    dataset = pd.read_csv(dataset_path)
    dataset_ids = dataset[ids_field]
    results.drop(["accession", "pident", "length", "mismatch", "gapopen", "qstart", "qend",
                  "sstart", "evalue", "bitscore", "name"], axis=1, inplace=True)

    results_ids = results["qseqid"]
    not_in_results = dataset[~dataset_ids.isin(results_ids)]
    not_in_results.drop([sequences_field], axis=1, inplace=True)

    for column in results.columns:
        if column != "qseqid":
            not_in_results[column] = 0.0

    not_in_results.columns = results.columns
    results = pd.concat([results, not_in_results])
    results.drop_duplicates(subset=["qseqid"], inplace=True)
    # Create a new column with the custom order as a categorical type
    results['CustomOrder'] = pd.Categorical(results['qseqid'], categories=dataset[ids_field], ordered=True)
    results.sort_values('CustomOrder', inplace=True)
    results.drop(columns=["CustomOrder"], inplace=True)
    results.reset_index(drop=True, inplace=True)

    os.remove("temp.fasta")
    os.remove("temp_results_file")

    os.chdir(current_directory)
    return results

def make_blast_prediction(dataset_path: str, sequences_field: str,
                          ids_field: str, database_folder: str, database_name: str,
                          output_path: str, binarised=False):
    """
    Make a prediction using BLAST.

    Parameters
    ----------
    dataset_path: str
        Path to the dataset in a csv format.
    sequences_field: str
        Path to the database.
    ids_field: str
        Field containing the ids.
    database_folder: str
        Folder with the database in csv and BLAST database format.
    output_path: str
        Path to the output file.
    binarised: bool
        Binarise input
    """
    results = _make_blast_prediction(dataset_path, sequences_field, ids_field, database_folder,
                                     database_name, output_path, binarised)
    results.to_csv(output_path, index=False)


def get_ec_from_regex_match(match: re.Match) -> Union[str, None]:
    """
    Get the EC number from a regex match.

    Parameters
    ----------
    match: re.Match
        Regex match.

    Returns
    -------
    EC: str
        EC number.
    """
    if match is not None:
        EC = match.group()
        if EC is not None:
            return EC
    return None


def _generate_ec_number_from_model_predictions(ECs: list) -> Tuple[list, list, list, list]:
    """
    Generate EC numbers from model predictions.

    Parameters
    ----------
    ECs: list
        List of EC numbers.

    Returns
    -------
    EC1: list
        List of EC1 numbers.
    EC2: list
        List of EC2 numbers.
    EC3: list
        List of EC3 numbers.
    EC4: list
        List of EC4 numbers.
    """
    EC3 = []
    EC2 = []
    EC1 = []
    EC4 = []
    for EC in ECs:
        new_EC = re.search(r"^\d+.\d+.\d+.n*\d+", EC)
        new_EC = get_ec_from_regex_match(new_EC)
        if isinstance(new_EC, str):
            if new_EC not in EC4:
                EC4.append(new_EC)

        new_EC = re.search(r"^\d+.\d+.\d+", EC)
        new_EC = get_ec_from_regex_match(new_EC)
        if isinstance(new_EC, str):
            if new_EC not in EC3:
                EC3.append(new_EC)

        new_EC = re.search(r"^\d+.\d+", EC)
        new_EC = get_ec_from_regex_match(new_EC)
        if isinstance(new_EC, str):
            if new_EC not in EC2:
                EC2.append(new_EC)

        new_EC = re.search(r"^\d+", EC)
        new_EC = get_ec_from_regex_match(new_EC)
        if isinstance(new_EC, str):
            if new_EC not in EC1:
                EC1.append(new_EC)

    return EC1, EC2, EC3, EC4

def _make_predictions_with_model(dataset, pipeline, device, all_data):
    
    pipeline.steps["place_holder"][-1].device = device
    if "cuda" in device:
        pipeline.steps["place_holder"][-1].num_gpus = 1
    pipeline.models[0].model.to(device)
    pipeline.models[0].device = device
    predictions = pipeline.predict(dataset)
    if all_data:
        path = os.path.join(SRC_PATH, "labels_names_all_data.pkl")
    else:
        path = os.path.join(SRC_PATH, "labels_names.pkl")

    results_dataframe = pd.DataFrame(columns=["accession", "EC1", "EC2", "EC3", "EC4"])
    labels_names = read_pickle(path)
    # get all the column indexes where the value is 1
    indices = [np.where(row == 1)[0].tolist() for row in predictions]
    labels_names = np.array(labels_names)

    ids = dataset.dataframe[dataset.instances_ids_field]
    for i in range(len(indices)):
        label_predictions = labels_names[indices[i]]

        EC1, EC2, EC3, EC4 = _generate_ec_number_from_model_predictions(label_predictions)
        label_predictions = [";".join(EC1)] + [";".join(EC2)] + [";".join(EC3)] + [";".join(EC4)]
        results_dataframe.loc[i] = [ids[i]] + label_predictions

    return results_dataframe


def make_predictions_with_model(pipeline_path: str, dataset_path: str, sequences_field: str,
                                ids_field: str, output_path: str, all_data: bool = True,
                                device: str = "cpu"):
    """
    Make predictions with a model.

    Parameters
    ----------
    pipeline_path: str
        Path to the pipeline.
    dataset_path: str
        Path to the dataset in a csv format.
    sequences_field: str
        Path to the database.
    ids_field: str
        Field containing the ids.
    output_path: str
        Path to the output file.
    all_data: bool
        Use all data from the dataset.
    device: str
        Device to use.
    """
    dataset = SingleInputDataset.from_csv(dataset_path, representation_field=sequences_field,
                                          instances_ids_field=ids_field)
    pipeline = Pipeline.load(pipeline_path)

    results_dataframe = _make_predictions_with_model(dataset, pipeline, device, all_data)

    results_dataframe.to_csv(output_path, index=False)

def determine_ensemble_predictions(threshold=2, *model_predictions):
    model_predictions = list(model_predictions)

    for i, model_prediction in enumerate(model_predictions):
        model_predictions[i] = np.array(model_prediction)


    predictions_voting = np.zeros_like(model_predictions[0])

    for i in range(model_predictions[0].shape[0]):
        # Combine conditions into a single array and sum along the second axis
        combined_conditions = np.sum(np.array([model_predictions[j][i] for j in range(len(model_predictions))]), axis=0)

        # Apply the threshold condition
        predictions_voting[i] = (combined_conditions >= threshold).astype(int)

    # If you want to ensure the resulting array is of integer type
    predictions_voting = predictions_voting.astype(int)
    return predictions_voting


def make_ensemble_prediction(dataset_path: str, pipelines: List[str], sequences_field: str,
                                ids_field: str, output_path: str, blast_database, blast_database_folder_path, 
                                all_data: bool = True,
                                device: str = "cpu"):
    """
    Make an ensemble prediction.

    Parameters
    ----------
    pipelines: List[str]
        List of paths to the pipelines.
    blast_database: str
        Path to the BLAST database.
    all_data: bool
        Use all data from the dataset.
    """
    results_dataframe = pd.DataFrame(columns=["accession", "EC1", "EC2", "EC3", "EC4"])
    results = []
    for pipeline in pipelines:
        
        dataset = SingleInputDataset.from_csv(dataset_path, representation_field=sequences_field,
                                                                   instances_ids_field=ids_field)
        pipeline = Pipeline.load(pipeline)
        pipeline.steps["place_holder"][-1].device = device
        if "cuda" in device:
            pipeline.steps["place_holder"][-1].num_gpus = 1
        pipeline.models[0].model.to(device)
        pipeline.models[0].device = device
        predictions = pipeline.predict(dataset)
        
        results.append(predictions)
    
    blast_results = _make_blast_prediction(dataset_path, sequences_field, ids_field, 
                                           blast_database_folder_path, blast_database)
    
    if all_data:
        path = os.path.join(SRC_PATH, "labels_names_all_data.pkl")
    else:
        path = os.path.join(SRC_PATH, "labels_names.pkl")

    labels_names = read_pickle(path)

    blast_results_array = np.zeros((len(blast_results), len(labels_names)))
    for i, row in blast_results.iterrows():
        EC1 = row["EC1"]
        EC2 = row["EC2"]
        EC3 = row["EC3"]
        EC4 = row["EC4"]
        if isinstance(EC1, float):
            EC1 = ""
        if isinstance(EC2, float):
            EC2 = ""
        if isinstance(EC3, float):
            EC3 = ""
        if isinstance(EC4, float):
            EC4 = ""
        EC1 = EC1.split(";")
        EC2 = EC2.split(";")
        EC3 = EC3.split(";")
        EC4 = EC4.split(";")
        for EC in EC1 + EC2 + EC3 + EC4:
            try:
                index = labels_names.index(EC)
                blast_results_array[i, index] = 1
            except ValueError:
                pass
    
    determined_predictions = determine_ensemble_predictions(2, *results, blast_results_array)
    results_dataframe = pd.DataFrame(columns=["accession", "EC1", "EC2", "EC3", "EC4"])
    labels_names = read_pickle(path)
    # get all the column indexes where the value is 1
    indices = [np.where(row == 1)[0].tolist() for row in determined_predictions]
    labels_names = np.array(labels_names)

    ids = dataset.dataframe[dataset.instances_ids_field]
    for i in range(len(indices)):
        label_predictions = labels_names[indices[i]]

        EC1, EC2, EC3, EC4 = _generate_ec_number_from_model_predictions(label_predictions)
        label_predictions = [";".join(EC1)] + [";".join(EC2)] + [";".join(EC3)] + [";".join(EC4)]
        results_dataframe.loc[i] = [ids[i]] + label_predictions
    
    results_dataframe.to_csv(output_path, index=False)