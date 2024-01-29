import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from ec_number_prediction.enumerators import ModelsDownloadPaths, BLASTDownloadPaths
import requests
import zipfile
import os
import shutil

from Bio import SeqIO
import csv

def get_unique_labels_by_level(dataset, level):
    final_dataset_test = dataset.copy()
    final_dataset_test = final_dataset_test.loc[:, level]
    final_dataset_test.fillna("0", inplace=True)
    values = pd.Series(final_dataset_test.values.reshape(-1)).str.split(";")
    list_of_unique_labels = np.unique(values.explode()).tolist()
    if "0" in list_of_unique_labels:
        list_of_unique_labels.remove("0")
    list_of_unique_labels_dict = dict(zip(list_of_unique_labels, range(len(list_of_unique_labels))))
    return list_of_unique_labels_dict


def get_final_labels(dataset):
    unique_EC1 = get_unique_labels_by_level(dataset, "EC1")
    unique_EC2 = get_unique_labels_by_level(dataset, "EC2")
    unique_EC3 = get_unique_labels_by_level(dataset, "EC3")

    array_EC1 = np.zeros((len(dataset), len(unique_EC1)))
    array_EC2 = np.zeros((len(dataset), len(unique_EC2)))
    array_EC3 = np.zeros((len(dataset), len(unique_EC3)))

    unique_EC4 = get_unique_labels_by_level(dataset, "EC4")

    array_EC4 = np.zeros((len(dataset), len(unique_EC4)))

    dataset.fillna("0", inplace=True)

    for i, row in dataset.iterrows():
        for ec in ["EC1", "EC2", "EC3", "EC4"]:
            for EC in row[ec].split(";"):
                if EC != "0":
                    if ec == "EC1":
                        array_EC1[i, unique_EC1[EC]] = 1
                    elif ec == "EC2":
                        array_EC2[i, unique_EC2[EC]] = 1
                    elif ec == "EC3":
                        array_EC3[i, unique_EC3[EC]] = 1
                    elif ec == "EC4":
                        array_EC4[i, unique_EC4[EC]] = 1
    for i, row in dataset.iterrows():
        for EC in row["EC4"].split(";"):
            if EC != "0":
                array_EC4[i, unique_EC4[EC]] = 1
    array_EC1 = pd.DataFrame(array_EC1, columns=unique_EC1.keys())
    array_EC2 = pd.DataFrame(array_EC2, columns=unique_EC2.keys())
    array_EC3 = pd.DataFrame(array_EC3, columns=unique_EC3.keys())
    array_EC4 = pd.DataFrame(array_EC4, columns=unique_EC4.keys())

    dataset = pd.concat((dataset, array_EC1, array_EC2, array_EC3, array_EC4), axis=1)
    return dataset


def get_ec_from_regex_match(match):
    if match is not None:
        EC = match.group()
        if EC is not None:
            return EC
    return None


def get_labels_based_on_list(dataset, labels, all_levels=True):
    array = np.zeros((len(dataset), len(labels)))
    labels_dataframe = pd.DataFrame(array, columns=labels)
    dataset.fillna("0", inplace=True)
    for i, row in dataset.iterrows():
        for label in row["EC4"].split(";"):
            if label != "0" and label in labels:
                labels_dataframe.at[i, label] = 1

        if all_levels:
            for ec in ["EC1", "EC2", "EC3"]:
                for label in row[ec].split(";"):
                    if label != "0" and label in labels:
                        labels_dataframe.at[i, label] = 1

    return pd.concat((dataset, labels_dataframe), axis=1)


def divide_labels_by_EC_level(final_dataset, ec_label):
    EC1_lst = []
    EC2_lst = []
    EC3_lst = []
    EC4_lst = []

    for _, row in final_dataset.iterrows():
        ECs = row[ec_label]
        ECs = ECs.split(";")
        # get the first 3 ECs with regular expression
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

        if len(EC4) == 0:
            EC4_lst.append(np.NaN)
        else:
            EC4_lst.append(";".join(EC4))
        if len(EC3) == 0:
            EC3_lst.append(np.NaN)
        else:
            EC3_lst.append(";".join(EC3))
        if len(EC2) == 0:
            EC2_lst.append(np.NaN)
        else:
            EC2_lst.append(";".join(EC2))
        if len(EC1) == 0:
            EC1_lst.append(np.NaN)
        else:
            EC1_lst.append(";".join(EC1))

    assert None not in EC1_lst
    assert None not in EC2_lst
    assert None not in EC3_lst
    assert None not in EC4_lst

    assert len(EC1_lst) == len(final_dataset)
    assert len(EC2_lst) == len(final_dataset)
    assert len(EC3_lst) == len(final_dataset)
    assert len(EC4_lst) == len(final_dataset)

    final_dataset["EC1"] = EC1_lst
    final_dataset["EC2"] = EC2_lst
    final_dataset["EC3"] = EC3_lst
    final_dataset["EC4"] = EC4_lst

    assert final_dataset["EC1"].isnull().sum() == 0
    print("EC1 is not null")

    return final_dataset

def convert_fasta_to_csv(fasta_file: str, csv_file: str):
    """
    Converts a FASTA file to a CSV file.

    Parameters
    ----------
    fasta_file : str
        Path to the FASTA file.
    csv_file : str
        Path to the CSV file.
    """
    with open(fasta_file, "r") as fasta, open(csv_file, "w", newline="") as csv_out:
        csv_writer = csv.writer(csv_out)
        csv_writer.writerow(["id", "sequence"])  # Writing header
        for record in SeqIO.parse(fasta, "fasta"):
            csv_writer.writerow([record.id, str(record.seq)])



def _download_and_unzip_file_to_cache(url: str, cache_path: str, method_name: str) -> str:
    """
    Downloads a file from a url and unzips it to a given cache path.

    Parameters
    ----------
    url : str
        URL of the file to download.
    cache_path : str
        Path to the cache folder.
    method_name : str
        Name of the method to download.

    Returns
    -------
    str
        Path to the downloaded file.
    """
    
    pipeline_name_for_path = method_name.replace(" ", "_")
    pipeline_cache_file = os.path.join(cache_path, f"{pipeline_name_for_path}.zip")

    if os.path.exists(os.path.join(cache_path, pipeline_name_for_path)):
        print(f"Pipeline {method_name} already in cache.")
        return os.path.join(cache_path, pipeline_name_for_path)

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    print(f"Downloading pipeline {method_name} to cache...")
    response = requests.get(url, stream=True)
    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(pipeline_cache_file, "wb") as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)

    # unzip pipeline
    print(f"Unzipping pipeline {method_name}...")
    with zipfile.ZipFile(pipeline_cache_file, "r") as zip_ref:
        zip_ref.extractall(cache_path)

    os.remove(pipeline_cache_file)

    return os.path.join(cache_path, pipeline_name_for_path)

def _download_blast_database_to_cache(blast_database: str) -> str:
    """
    Downloads a BLAST database to the cache folder.
    
    Parameters
    ----------
    blast_database : str
        Name of the BLAST database to download.
    
    Returns
    -------
    str
        Path to the downloaded BLAST database.
    """

    databases = {
        "BLAST all data": BLASTDownloadPaths.BLAST_ALL_DATA.value,
        "BLAST train plus validation": BLASTDownloadPaths.BLAST_TRAIN_VALID.value,
    }

    if blast_database not in databases:
        raise Exception(f"BLAST database {blast_database} not found.")

    database_url = databases[blast_database]

    database_cache_path = os.path.join(os.path.expanduser("~"), ".ec_number_prediction", "blast_databases")

    return _download_and_unzip_file_to_cache(database_url, database_cache_path, blast_database)
    

def _download_pipeline_to_cache(pipeline: str) -> str:
    """
    Downloads a pipeline to the cache folder.

    Parameters
    ----------
    pipeline : str
        Name of the pipeline to download.
    
    Returns
    -------
    str
        Path to the downloaded pipeline.
    """

    pipelines = {
        "DNN ESM1b all data": ModelsDownloadPaths.DNN_ESM1b_ALL_DATA.value,
        "DNN ProtBERT all data": ModelsDownloadPaths.DNN_PROTBERT_ALL_DATA.value,
        "DNN ESM2 3B all data": ModelsDownloadPaths.DNN_ESM2_3B_ALL_DATA.value,
        "DNN ESM2 3B trial 2 train plus validation": ModelsDownloadPaths.DNN_ESM2_3B_TRAIN_VALID.value,
        "DNN ESM1b trial 4 train plus validation": ModelsDownloadPaths.DNN_ESM1b_TRAIN_VALID.value,
        "ProtBERT trial 2 train plus validation": ModelsDownloadPaths.DNN_PROTBERT_TRAIN_VALID.value,
    }

    if pipeline not in pipelines:
        raise Exception(f"Pipeline {pipeline} not found.")
    
    pipeline_url = pipelines[pipeline]

    pipeline_cache_path = os.path.join(os.path.expanduser("~"), ".ec_number_prediction", "pipelines")
    pipeline_name_for_path = pipeline.replace(" ", "_")

    if os.path.exists(os.path.join(pipeline_cache_path, pipeline_name_for_path)):
        print(f"Pipeline {pipeline} already in cache.")
        return os.path.join(pipeline_cache_path, pipeline_name_for_path)

    return _download_and_unzip_file_to_cache(pipeline_url, pipeline_cache_path, pipeline)