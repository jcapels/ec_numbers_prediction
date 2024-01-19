import os
import sys
from typing import List

from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset
from plants_sm.data_standardization.proteins.padding import SequencePadder
from plants_sm.featurization.encoding.one_hot_encoder import OneHotEncoder
from plants_sm.data_standardization.truncation import Truncator
from plants_sm.data_standardization.x_padder import XPadder
from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder
from plants_sm.featurization.proteins.bio_embeddings.prot_bert import ProtBert
import gc

from plants_sm.transformation.transformer import Transformer


def transform_datasets(transformers: List[Transformer], save_folder: str, dataset_directory: str,
                       batch_size: int = None):
    """
    Transform the train, validation and test datasets using the transformers and save the results in the save_folder.

    Parameters
    ----------
    transformers: List[Transformer]
        List of transformers to be used.
    save_folder: str
        Folder where the results will be saved.
    dataset_directory: str
        Directory where the datasets are located.
    batch_size: str
        Batch size used to load the datasets.

    Returns
    -------

    """
    datasets = ["test", "validation"]
    # dataset_directory = "/scratch/jribeiro/ec_number_prediction/final_data"
    dataset = os.path.join(dataset_directory, "train.csv")
    dataset = SingleInputDataset.from_csv(dataset, instances_ids_field="accession", representation_field="sequence",
                                          labels_field=slice(8, -1), batch_size=batch_size)

    for transformer in transformers:
        transformer.fit(dataset)
        transformer.transform(dataset)

    dataset.save_features(f"{save_folder}/train")

    del dataset
    gc.collect()

    for dataset_name in datasets:
        dataset = os.path.join(dataset_directory, dataset_name)
        dataset = SingleInputDataset.from_csv(f"{dataset}.csv", instances_ids_field="accession",
                                              representation_field="sequence",
                                              labels_field=slice(8, -1), batch_size=batch_size)
        for transformer in transformers:
            transformer.fit(dataset)
            transformer.transform(dataset)

        dataset.save_features(f"{save_folder}/{dataset_name}")
        del dataset
        gc.collect()


def generate_one_hot_encodings(dataset_directory: str, save_folder: str):
    """
    Generate the one hot encodings for the train, validation and test datasets and save the results in the save_folder.

    Parameters
    ----------
    dataset_directory: str
        Directory where the datasets are located.
    save_folder: str
        Folder where the results will be saved.

    Returns
    -------

    """
    transformers = [ProteinStandardizer(), OneHotEncoder(max_length=884)]
    # save_folder = "/scratch/jribeiro/results/one_hot_encoding"
    transform_datasets(transformers, save_folder, dataset_directory=dataset_directory)


def generate_prot_bert_vectors(save_folder: str, dataset_directory: str):
    """
    Generate the ProtBert vectors for the train, validation and test datasets and save the results in the save_folder.

    Parameters
    ----------
    save_folder: str
        Folder where the results will be saved.
    dataset_directory: str
        Directory where the datasets are located.

    Returns
    -------

    """
    transformers = [ProteinStandardizer(), Truncator(max_length=884), ProtBert(device="cuda")]

    # save_folder = "/scratch/jribeiro/results/prot_bert_vectors"
    transform_datasets(transformers, save_folder, dataset_directory=dataset_directory)


def generate_esm_vectors(esm_function: str, save_folder: str, dataset_directory: str):
    """
    Generate the ESM2 vectors for the train, validation and test datasets and save the results in the save_folder.

    Parameters
    ----------
    esm_function: str
        ESM2 function to be used.
    save_folder: str
        Folder where the results will be saved.
    dataset_directory: str
        Directory where the datasets are located.

    Returns
    -------

    """
    transformers = [ProteinStandardizer(), Truncator(max_length=884),
                    ESMEncoder(esm_function=esm_function, batch_size=1, num_gpus=4)]

    # results_name = f"/scratch/jribeiro/results/{esm_function}"
    transform_datasets(transformers, save_folder, dataset_directory=dataset_directory)


if __name__ == "__main__":
    encoding_type = sys.argv[0]

    if encoding_type == "one_hot":
        save_folder = sys.argv[1]
        dataset_directory = sys.argv[2]
        generate_one_hot_encodings(dataset_directory, save_folder)
    elif encoding_type == "prot_bert":
        save_folder = sys.argv[1]
        dataset_directory = sys.argv[2]
        generate_prot_bert_vectors(save_folder, dataset_directory)
    elif encoding_type == "esm":
        esm_function = sys.argv[1]
        save_folder = sys.argv[2]
        dataset_directory = sys.argv[3]
        generate_esm_vectors(esm_function, save_folder, dataset_directory)

    else:
        raise ValueError("Invalid encoding type. They can be one_hot, prot_bert or esm."
                         "If you want to use esm, you need to specify the esm function.")
