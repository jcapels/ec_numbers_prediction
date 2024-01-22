import os
import sys

import pandas as pd
from plants_sm.alignments.alignment import BLAST
from plants_sm.data_structures.dataset import SingleInputDataset
from plants_sm.utilities.utils import convert_csv_to_fasta

from ec_number_prediction._utils import get_final_labels


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

    os.remove("temp.fasta")
    os.remove("temp_results_file")

    os.chdir(current_directory)
    results.to_csv(output_path, index=False)


def make_predictions_with_model(model_name: str, dataset_path: str, sequences_field: str,
                                ids_field: str, output_path: str):
    """
    Make predictions with a model.

    Parameters
    ----------
    model_name: str
        Name of the model.
    dataset_path: str
        Path to the dataset in a csv format.
    sequences_field: str
        Path to the database.
    ids_field: str
        Field containing the ids.
    output_path: str
        Path to the output file.
    """
    dataset = SingleInputDataset.from_csv(dataset_path, representation_field=sequences_field, instances_ids_field=ids_field)
    if model_name == "DNN ESM1b":
        pass

