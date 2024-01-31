import os
import os.path
import re
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from plants_sm.alignments.alignment import BLAST
from plants_sm.data_structures.dataset import SingleInputDataset, Dataset
from plants_sm.io.pickle import read_pickle
from plants_sm.pipeline.pipeline import Pipeline
from plants_sm.utilities.utils import convert_csv_to_fasta
import torch

from ec_number_prediction import SRC_PATH
from ec_number_prediction._utils import get_final_labels, _download_blast_database_to_cache, \
    _download_pipeline_to_cache, convert_fasta_to_csv
from ec_number_prediction.enumerators import BLASTDatabases


def _make_blast_prediction(dataset_path: str, sequences_field: str,
                           ids_field: str, database_folder: str, database_name: str,
                           binarised=False) -> pd.DataFrame:
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
    database_name: str
        Name of the database.
    binarised: bool (default: False)
        Binarise input

    Returns
    -------
    results: pd.DataFrame
        Results of the prediction.
    """
    current_directory = os.getcwd()
    os.chdir(database_folder)
    convert_csv_to_fasta(dataset_path, sequences_field, ids_field, 'temp.fasta')

    database_names = {
        "BLAST all data": BLASTDatabases.BLAST_ALL_DATA.value,
        "BLAST train plus validation": BLASTDatabases.BLAST_TRAIN_VALID.value,
    }

    blast = BLAST(database_names[database_name])
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

    if not_in_results.shape[0] > 0:
        for column in results.columns:
            if column != "qseqid":
                not_in_results.loc[:, column] = 0.0

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
    database_name: str
        Name of the database.
    output_path: str
        Path to the output file.
    binarised: bool
        Binarise input
    """
    results = _make_blast_prediction(dataset_path, sequences_field, ids_field, database_folder,
                                     database_name, binarised)
    results.to_csv(output_path, index=False)


def predict_with_blast(dataset_path: str, sequences_field: str,
                       ids_field: str, database_name: str,
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
    database_name: str
        Name of the database.
    output_path: str
        Path to the output file.
    binarised: bool
        Binarise input
    """
    database_path = _download_blast_database_to_cache(database_name)
    make_blast_prediction(dataset_path, sequences_field, ids_field, database_path,
                          database_name, output_path, binarised)


def predict_with_blast_from_fasta(fasta_path: str, database_name: str,
                                  output_path: str, binarised=False):
    """
    Make a prediction using BLAST.

    Parameters
    ----------
    fasta_path: str
        Path to the fasta file.
    database_name: str
        Name of the database.
    output_path: str
      Path to the output file.
    binarised: bool
      Binarise input
    """
    current_directory = os.getcwd()
    temp_csv = os.path.join(current_directory, "temp.csv")
    database_path = _download_blast_database_to_cache(database_name)
    convert_fasta_to_csv(fasta_path, temp_csv)
    try:
        make_blast_prediction(temp_csv, "sequence", "id", database_path,
                              database_name, output_path, binarised)
        os.remove(temp_csv)
    except Exception as e:
        os.remove(temp_csv)
        raise Exception(e)


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


def _make_predictions_with_model(dataset: Dataset, pipeline: Pipeline, device: str, all_data: bool = True,
                                 num_gpus: int = 1) \
        -> pd.DataFrame:
    """
    Make predictions with a model.

    Parameters
    ----------
    dataset: Dataset
        Dataset.
    pipeline: Pipeline
        Pipeline.
    device: str
        Device to use.
    all_data: bool
        Use all data from the dataset.
    num_gpus: int
        Number of GPUs to use.

    Returns
    -------
    results_dataframe: pd.DataFrame
        Results of the prediction.
    """
    pipeline.steps["place_holder"][-1].device = device
    
    if "cuda" in device:
        if device == "cuda":
            pipeline.steps["place_holder"][-1].num_gpus = num_gpus
        else:
            pipeline.steps["place_holder"][-1].num_gpus = 1

        pipeline.steps["place_holder"][-1].is_ddf = True
    
    if pipeline.steps["place_holder"][-1].__class__.__name__ == "ProtBert":
        pipeline.steps["place_holder"][-1].model.to(device)

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
                                device: str = "cpu", num_gpus: int = 1):
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
    num_gpus: int
        Number of GPUs to use for predicting the ESM embedding.
    """
    dataset = SingleInputDataset.from_csv(dataset_path, representation_field=sequences_field,
                                          instances_ids_field=ids_field)
    pipeline = Pipeline.load(pipeline_path)

    results_dataframe = _make_predictions_with_model(dataset, pipeline, device, all_data, num_gpus=num_gpus)

    results_dataframe.to_csv(output_path, index=False)


def predict_with_model(pipeline: str, dataset_path: str, sequences_field: str,
                       ids_field: str, output_path: str, all_data: bool = True,
                       device: str = "cpu", num_gpus: int = 1):
    """
    Make predictions with a model.

    Parameters
    ----------
    pipeline: str
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
    num_gpus: int
        Number of GPUs to use for predicting the ESM embedding.

    Returns
    -------

    """
    pipeline_path = _download_pipeline_to_cache(pipeline)
    make_predictions_with_model(pipeline_path=pipeline_path,
                                dataset_path=dataset_path,
                                sequences_field=sequences_field,
                                ids_field=ids_field,
                                output_path=output_path,
                                all_data=all_data, device=device,
                                num_gpus=num_gpus)


def predict_with_model_from_fasta(pipeline: str, fasta_path: str,
                                  output_path: str, all_data: bool = True,
                                  device: str = "cpu", num_gpus: int = 1):
    """
    Make predictions with a model.

    Parameters
    ----------
    pipeline: str
        Path to the pipeline.
    fasta_path: str
        Path to the fasta file.
    output_path: str
        Path to the output file.
    all_data: bool
        Use all data from the dataset.
    device: str
        Device to use.
    num_gpus: int
        Number of GPUs to use for predicting the ESM embedding.

    """
    current_directory = os.getcwd()
    temp_csv = os.path.join(current_directory, "temp.csv")
    convert_fasta_to_csv(fasta_path, temp_csv)
    try:
        predict_with_model(pipeline, temp_csv, "sequence", "id", output_path, all_data, device, num_gpus)
        os.remove(temp_csv)
    except Exception as e:
        os.remove(temp_csv)
        raise Exception(e)


def determine_ensemble_predictions(threshold: int = 2, *model_predictions) -> np.ndarray:
    """
    Determine the ensemble predictions.

    Parameters
    ----------
    threshold: int
        Threshold to use.
    model_predictions: list
        List of model predictions.

    Returns
    -------

    """
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
                             device: str = "cpu", num_gpus: int = 1):
    """
    Make an ensemble prediction.

    Parameters
    ----------
    dataset_path: str
        Path to the dataset in a csv format.
    pipelines: List[str]
        List of paths to the pipelines.
    sequences_field: str
        Path to the database.
    ids_field: str
        Field containing the ids.
    output_path: str
        Path to the output file.
    blast_database: str
        Path to the BLAST database.
    blast_database_folder_path: str
        Path to the BLAST database folder.
    all_data: bool
        Use all data from the dataset.
    device: str
        Device to use.
    num_gpus: int
        Number of GPUs to use for predicting the ESM embedding.

    """
    results = []
    for pipeline in pipelines:

        dataset = SingleInputDataset.from_csv(dataset_path, representation_field=sequences_field,
                                              instances_ids_field=ids_field)
        pipeline = Pipeline.load(pipeline)
        pipeline.steps["place_holder"][-1].device = device
        if "cuda" in device:
            if device == "cuda":
                pipeline.steps["place_holder"][-1].num_gpus = num_gpus
            else:
                pipeline.steps["place_holder"][-1].num_gpus = 1

            pipeline.steps["place_holder"][-1].is_ddf = True
    
        if pipeline.steps["place_holder"][-1].__class__.__name__ == "ProtBert":
            pipeline.steps["place_holder"][-1].model.to(device)
        
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


def predict_with_ensemble(dataset_path: str, sequences_field: str,
                          ids_field: str, output_path: str,
                          all_data: bool = True,
                          device: str = "cpu",
                          num_gpus: int = 1):
    """
    Make an ensemble prediction.

    Parameters
    ----------
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
    num_gpus: int
        Number of GPUs to use for predicting the ESM embedding.
    """
    esm2_3b = _download_pipeline_to_cache("DNN ESM2 3B all data")
    prot_bert = _download_pipeline_to_cache("DNN ProtBERT all data")
    esm1b = _download_pipeline_to_cache("DNN ESM1b all data")
    pipelines = [esm2_3b, prot_bert, esm1b]
    blast_database = "BLAST all data"
    blast_database_folder_path = _download_blast_database_to_cache(blast_database)
    make_ensemble_prediction(dataset_path, pipelines, sequences_field, ids_field, output_path, blast_database,
                             blast_database_folder_path, all_data, device, num_gpus)


def predict_with_ensemble_from_fasta(fasta_path: str,
                                     output_path: str,
                                     all_data: bool = True,
                                     device: str = "cpu", 
                                     num_gpus: int = 1):
    """
    Make an ensemble prediction.

    Parameters
    ----------
    fasta_path: str
        Path to the fasta file.
    output_path: str
        Path to the output file.
    all_data: bool
        Use all data from the dataset.
    device: str
        Device to use.
    num_gpus: int
        Number of GPUs to use for predicting the ESM embedding.
    """
    current_directory = os.getcwd()
    temp_csv = os.path.join(current_directory, "temp.csv")
    convert_fasta_to_csv(fasta_path, temp_csv)
    try:
        predict_with_ensemble(temp_csv, "sequence", "id", output_path, all_data, device, num_gpus)
        os.remove(temp_csv)
    except Exception as e:
        os.remove(temp_csv)
        raise Exception(e)
