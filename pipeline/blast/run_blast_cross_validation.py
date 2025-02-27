from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd

from plants_sm.alignments.alignment import BLAST


def create_database(folder_path: str, database_fasta_file_path: str):
    """
    Create a database for BLAST.

    Parameters
    ----------
    folder_path: str
        Path to the folder where the database will be created.
    database_fasta_file_path:
        Path to the fasta file that will be used to create the database.
    """

    blast = BLAST(folder_path)
    blast.create_database(database_fasta_file_path)

def _process_blast_results_for_test_set(blast_results_path: str, test_set_path: str, i: int):
    """
    Process the BLAST results for the test set.

    Parameters
    ----------
    blast_results_path: str
        Path to the BLAST results.
    test_set_path: str
        Path to the test set.
    """
    test = pd.read_csv(test_set_path)
    predictions = pd.read_csv(blast_results_path)
    predictions_no_duplicates = predictions.drop_duplicates(subset=['qseqid', 'accession'])

    predictions_no_duplicates.drop(columns=['accession', 'pident', 'length', 'mismatch', 'gapopen',
                                            'qstart', 'qend', 'sstart', 'evalue', 'bitscore', 'name'], inplace=True)

    not_in_predictions = test[~test.loc[:, "accession"].isin(predictions_no_duplicates.loc[:, "qseqid"])]
    not_in_predictions.iloc[:, 8:] = 0.0
    not_in_predictions.drop(["sequence", "name"], axis=1, inplace=True)
    not_in_predictions.columns = predictions_no_duplicates.columns

    predictions_no_duplicates = pd.concat([predictions_no_duplicates, not_in_predictions])
    predictions_no_duplicates.sort_values(by=['qseqid'], inplace=True)
    test.sort_values(by=['accession'], inplace=True)
    assert predictions_no_duplicates.loc[:, "qseqid"].tolist() == test.loc[:, "accession"].tolist()


    predictions_no_duplicates.to_csv(blast_results_path, index=False)

def prepare_dataset(dataset_path):
    df = pd.read_csv(dataset_path)

    # Open the output FASTA file
    with open(dataset_path.replace("csv", "faa"), 'w') as fasta_out:
        # Iterate through the rows of the DataFrame
        for index, row in df.iterrows():
            sequence_id = row["accession"]    # Column with sequence ID
            sequence = row["sequence"] # Column with sequence

            # Write to FASTA file
            fasta_out.write(f">{sequence_id}\n")
            fasta_out.write(f"{sequence}\n")

import os
import pandas as pd
from joblib import Parallel, delayed

def process_single_blast_task(i, folder_path, work_path):

    dataset_path = os.path.join(folder_path, f"train_{i}.csv")
    if not os.path.exists(dataset_path.replace("csv", "faa")):
        prepare_dataset(dataset_path)

    train_db_path = os.path.join(work_path, "blast_predictions", f"train_dataset_{i}", f"train_dataset_{i}")
    if not os.path.exists(train_db_path):
        create_database(train_db_path, dataset_path.replace("csv", "faa"))

    test_path = os.path.join(folder_path, f"test_{i}.csv")
    if not os.path.exists(test_path.replace("csv", "faa")):
        prepare_dataset(test_path)

    blast = BLAST(train_db_path)
    blast.run(test_path.replace("csv", "faa"), os.path.join(work_path, "blast_predictions", f"test_{i}_predictions.tsv"), 1e-5, 1)

    # Process the BLAST results
    blast.results = pd.read_csv(os.path.join(work_path, "blast_predictions", f"test_{i}_predictions.tsv"), sep='\t',
                                header=None,
        names=["qseqid", "sseqid", "pident", "length", "mismatch",
                              "gapopen", "qstart", "qend", "sstart", "evalue", "bitscore"])

    # Write the BLAST predictions to a CSV file
    predictions_file = os.path.join(work_path, "blast_predictions", f"test_blast_predictions_{i}.csv")
    database_dataframe = pd.read_csv(dataset_path)
    blast.associate_to_ec(database_dataframe, predictions_file)

    # Process the results for the test set
    _process_blast_results_for_test_set(predictions_file, test_path, i)
    
    return f"Task {i} completed."

def create_databases_and_run_parallel(folder_path: str, work_path: str):
    os.makedirs(os.path.join(work_path, "blast_predictions"), exist_ok=True)

    # Parallelize the BLAST task for each of the 5 datasets
    Parallel(n_jobs=5, backend='multiprocessing')(
        delayed(process_single_blast_task)(i, folder_path, work_path) for i in range(5)
    )

if __name__ == "__main__":
    from plants_sm.alignments.alignment import BLAST

    
    create_databases_and_run_parallel("/home/jcapela/ec_number_prediction_version_2/ec_numbers_prediction/normal_splits_uniref90/monte_carlo_splits"
                             , "/home/jcapela/ec_number_prediction_version_2/ec_numbers_prediction/normal_splits_uniref90")