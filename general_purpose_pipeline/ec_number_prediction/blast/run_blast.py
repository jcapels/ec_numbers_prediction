import pandas as pd
import numpy as np
import os

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


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
    os.makedirs(folder_path, exist_ok=True)

    blast = BLAST(folder_path)
    blast.create_database(database_fasta_file_path)


def _process_blast_results_for_test_set(blast_results_path: str, test_set_path: str):
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
    predictions_no_duplicates.loc[:, ["qseqid", "accession", "pident", "evalue"]].to_csv(
        "predictions_no_duplicates_ident_evalue.csv", index=False)

    predictions_no_duplicates.drop(columns=['accession', 'pident', 'length', 'mismatch', 'gapopen',
                                            'qstart', 'qend', 'sstart', 'evalue', 'bitscore', 'name'], inplace=True)

    not_in_predictions = test[~test.loc[:, "accession"].isin(predictions_no_duplicates.loc[:, "qseqid"])]
    not_in_predictions.iloc[:, 8:] = 0.0
    not_in_predictions.drop(["sequence", "name"], axis=1, inplace=True)
    not_in_predictions.columns = predictions_no_duplicates.columns
    not_in_predictions.loc[:, "qseqid"].to_csv("not_in_predictions.csv", index=False)

    predictions_no_duplicates = pd.concat([predictions_no_duplicates, not_in_predictions])
    predictions_no_duplicates.sort_values(by=['qseqid'], inplace=True)
    test.sort_values(by=['accession'], inplace=True)
    assert predictions_no_duplicates.loc[:, "qseqid"].tolist() == test.loc[:, "accession"].tolist()

    predictions_no_duplicates.to_csv("test_blast_predictions_right_format.csv", index=False)
    test.to_csv("test_right_format.csv", index=False)


def run_blastp(query_file: str, output_file: str, database: str, evalue: float, num_hits: int, database_csv: str):
    """
    Run BLASTP.

    Parameters
    ----------
    query_file: str
        Path to the query file.
    output_file: str
        Path to the output file.
    database: str
        Path to the database.
    evalue: float
        E-value cutoff.
    num_hits: int
        Number of hits.
    database_csv: str
        Path to the database in csv format.
    """

    blast = BLAST(database)
    blast.run(query_file, output_file, evalue, num_hits)

    blast.results = pd.read_csv('test_database.tsv', sep='\t')

    blast.associate_to_ec(database_csv, "test_blast_predictions.csv")
    _process_blast_results_for_test_set("test_blast_predictions.csv", query_file)


def generate_metrics_for_blastp():
    """
    Generate the metrics for BLASTP.
    """

    test = pd.read_csv("test_right_format.csv")
    predictions = pd.read_csv("test_blast_predictions_right_format.csv")

    f1_score_overall = f1_score(test.iloc[:, 8:], predictions.iloc[:, 6:], average='macro')
    f1_score_1 = f1_score(test.iloc[:, 8:8 + 7], predictions.iloc[:, 6:6 + 7], average='macro')
    f1_score_2 = f1_score(test.iloc[:, 8 + 7:84], predictions.iloc[:, 6 + 7:82], average='macro')
    f1_score_3 = f1_score(test.iloc[:, 84:314], predictions.iloc[:, 82:312], average='macro')
    f1_score_4 = f1_score(test.iloc[:, 314:], predictions.iloc[:, 312:], average='macro')

    precision_score_overall = precision_score(test.iloc[:, 8:], predictions.iloc[:, 6:], average='macro')
    precision_score_1 = precision_score(test.iloc[:, 8:8 + 7], predictions.iloc[:, 6:6 + 7], average='macro')
    precision_score_2 = precision_score(test.iloc[:, 8 + 7:84], predictions.iloc[:, 6 + 7:82], average='macro')
    precision_score_3 = precision_score(test.iloc[:, 84:314], predictions.iloc[:, 82:312], average='macro')
    precision_score_4 = precision_score(test.iloc[:, 314:], predictions.iloc[:, 312:], average='macro')

    recall_score_overall = recall_score(test.iloc[:, 8:], predictions.iloc[:, 6:], average='macro')
    recall_score_1 = recall_score(test.iloc[:, 8:8 + 7], predictions.iloc[:, 6:6 + 7], average='macro')
    recall_score_2 = recall_score(test.iloc[:, 8 + 7:84], predictions.iloc[:, 6 + 7:82], average='macro')
    recall_score_3 = recall_score(test.iloc[:, 84:314], predictions.iloc[:, 82:312], average='macro')
    recall_score_4 = recall_score(test.iloc[:, 314:], predictions.iloc[:, 312:], average='macro')

    if os.path.exists("alignment_results.csv"):
        results = pd.read_csv("alignment_results.csv")
    else:
        results = pd.DataFrame(columns=["model", "metric", "train", "valid", "test"])

    alignment_method_name = "blast_train_validation"
    results.loc[results.shape[0] + 1] = [alignment_method_name, "f1_score_macro", np.NaN, np.NaN, f1_score_overall]
    results.loc[results.shape[0] + 2] = [alignment_method_name, "f1_score_macro_level_1", np.NaN, np.NaN, f1_score_1]
    results.loc[results.shape[0] + 3] = [alignment_method_name, "f1_score_macro_level_2", np.NaN, np.NaN, f1_score_2]
    results.loc[results.shape[0] + 4] = [alignment_method_name, "f1_score_macro_level_3", np.NaN, np.NaN, f1_score_3]
    results.loc[results.shape[0] + 5] = [alignment_method_name, "f1_score_macro_level_4", np.NaN, np.NaN, f1_score_4]
    results.loc[results.shape[0] + 6] = [alignment_method_name, "precision_score_macro", np.NaN, np.NaN,
                                         precision_score_overall]
    results.loc[results.shape[0] + 7] = [alignment_method_name, "precision_score_macro_level_1", np.NaN, np.NaN,
                                         precision_score_1]
    results.loc[results.shape[0] + 8] = [alignment_method_name, "precision_score_macro_level_2", np.NaN, np.NaN,
                                         precision_score_2]
    results.loc[results.shape[0] + 9] = [alignment_method_name, "precision_score_macro_level_3", np.NaN, np.NaN,
                                         precision_score_3]
    results.loc[results.shape[0] + 10] = [alignment_method_name, "precision_score_macro_level_4", np.NaN, np.NaN,
                                          precision_score_4]
    results.loc[results.shape[0] + 11] = [alignment_method_name, "recall_score_macro", np.NaN, np.NaN,
                                          recall_score_overall]
    results.loc[results.shape[0] + 12] = [alignment_method_name, "recall_score_macro_level_1", np.NaN, np.NaN,
                                          recall_score_1]
    results.loc[results.shape[0] + 13] = [alignment_method_name, "recall_score_macro_level_2", np.NaN, np.NaN,
                                          recall_score_2]
    results.loc[results.shape[0] + 14] = [alignment_method_name, "recall_score_macro_level_3", np.NaN, np.NaN,
                                          recall_score_3]
    results.loc[results.shape[0] + 15] = [alignment_method_name, "recall_score_macro_level_4", np.NaN, np.NaN,
                                          recall_score_4]

    results.to_csv("alignment_results.csv", index=False)


if __name__ == "__main__":
    from plants_sm.alignments.alignment import BLAST

    create_database("train_database", "database.fasta")

    run_blastp("../test.fasta", "test_database.tsv", database="train_database", evalue=1e-5,
               num_hits=1, database_csv="database.csv")

    generate_metrics_for_blastp()
