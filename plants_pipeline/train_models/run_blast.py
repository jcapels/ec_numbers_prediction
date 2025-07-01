import os
import re
from plants_sm.alignments.alignment import BLAST
from plants_sm.utilities.utils import convert_csv_to_fasta

from ec_number_prediction._utils import get_final_labels
from plants_sm.alignments.alignment import BLAST
import os
import pandas as pd
import numpy as np

base_path = "/home/jcapela/ec_numbers_prediction/plants_pipeline/data/plants/"

def create_database_all_data(base_path):
    os.makedirs(os.path.join(base_path, "blast_database_plants"), exist_ok=True)
    blast = BLAST(os.path.join(base_path, "blast_database_plants/blast_database_plants"))
    convert_csv_to_fasta(f'{base_path}swiss_prot_ec_plants.csv', 'sequence', 'accession', os.path.join(base_path, 'blast_database_plants/swiss_prot_ec_plants.fasta'))
    blast.create_database(os.path.join(base_path, 'blast_database_plants/swiss_prot_ec_plants.fasta'))

def generate_results_for_all_data_case_studies(base_path):
    blast = BLAST(os.path.join(base_path, "blast_database_plants/blast_database_plants"))
    convert_csv_to_fasta(f'{base_path}swiss_prot_ec_plants.csv', 'sequence', 'accession', os.path.join(base_path, 'blast_database_plants/swiss_prot_ec_plants.fasta'))
    
    blast.run(os.path.join(base_path, 'plants_sm_sequences.fasta'), 
              output_file=os.path.join(base_path, 'blast_database_plants/blast_all_data_results.csv'), evalue=1e-5, num_hits=1)
    
    database = pd.read_csv(os.path.join(base_path, 'swiss_prot_ec_plants.csv'))
    results = pd.read_csv(os.path.join(base_path, 'blast_database_plants/blast_all_data_results.csv'), sep='\t')
    blast.results = results
    blast.associate_to_ec(database, "temp_results_file")
    results = pd.read_csv("temp_results_file")
    
    # load fasta file
    from Bio import SeqIO
    dataset_ids = []
    for record in SeqIO.parse(os.path.join(base_path, 'plants_sm_sequences.fasta'), "fasta"):
        dataset_ids.append(record.id)

    results.drop(["accession", "pident", "length", "mismatch", "gapopen", "qstart", "qend",
                    "sstart", "evalue", "bitscore", "name"], axis=1, inplace=True)

    results_ids = results["qseqid"]
    not_in_results = [id_ for id_ in dataset_ids if id_ not in results_ids]
    not_in_results = pd.DataFrame(not_in_results, columns=["qseqid"])

    if not_in_results.shape[0] > 0:
        for column in results.columns:
            if column != "qseqid":
                not_in_results.loc[:, column] = np.NaN

        not_in_results.columns = results.columns
        results = pd.concat([results, not_in_results])
        
    results.drop_duplicates(subset=["qseqid"], inplace=True)
    # Create a new column with the custom order as a categorical type
    os.remove("temp_results_file")
    results.to_csv(os.path.join(base_path, 'blast_database_plants/blast_all_data_results.csv'), index=False)

def create_databases(base_path):
    train_datasets_path = os.path.join(base_path, "train_datasets/")

    for i in range(5):
        print(f"Loading dataset {i}")
        os.makedirs(os.path.join(base_path, f"blast_database_{i}"), exist_ok=True)
        blast = BLAST(os.path.join(base_path, f"blast_database_{i}/blast_database_{i}"))
        

        convert_csv_to_fasta(f'{train_datasets_path}train_dataset_{i}.csv', 'sequence', 'accession', os.path.join(base_path,f'blast_database_{i}/train_dataset_{i}.fasta'))
        blast.create_database(os.path.join(base_path, f'blast_database_{i}/train_dataset_{i}.fasta'))



def generate_results_for_test_set(base_path):
    test_datasets_path = os.path.join(base_path, "test_datasets/")

    for i in range(5):
        blast = BLAST(os.path.join(base_path, f"blast_database_{i}/blast_database_{i}"))
        convert_csv_to_fasta(f'{test_datasets_path}test_dataset_{i}.csv', 'sequence', 'accession', os.path.join(base_path, f'blast_database_{i}/test_dataset_{i}.fasta'))
        blast.run(os.path.join(base_path, f'blast_database_{i}/test_dataset_{i}.fasta'), 
                  output_file=os.path.join(base_path, f'blast_database_{i}/test_dataset_{i}_results.csv'), evalue=1e-5, num_hits=1)

def get_ec_levels(labels):
    level_1 = []
    level_2 = []
    level_3 = []
    level_4 = []
    for i, label in enumerate(labels):
        if re.match(r"^\d+.\d+.\d+.n*\d+$", label):
            level_4.append(i)
        elif re.match(r"^\d+.\d+.\d+$", label):
            level_3.append(i)
        elif re.match(r"^\d+.\d+$", label):
            level_2.append(i)
        elif re.match(r"^\d+$", label):
            level_1.append(i)
    return level_1, level_2, level_3, level_4

def generate_metrics(base_path):

    results_dataframe = pd.DataFrame(columns=["fold", "mf1", "wf1", "mrecall", "wrecall", "mprecision", "wprecision"])
    for i in range(5):

        results = pd.read_csv(os.path.join(base_path, f'blast_database_{i}/test_dataset_{i}_results.csv'), sep='\t')

        blast = BLAST(os.path.join(base_path, f"blast_database_{i}/blast_database_{i}"))
        database = pd.read_csv(os.path.join(base_path, f"train_datasets/train_dataset_{i}.csv"))
        blast.results = results
        blast.associate_to_ec(database, "temp_results_file")
        results = pd.read_csv("temp_results_file")
        results = results.astype({'EC1': 'str',
                                    'EC2': 'str',
                                    'EC3': 'str',
                                    'EC4': 'str'})
        dataset = pd.read_csv(os.path.join(base_path, f"test_datasets/test_dataset_{i}.csv"))
        dataset_ids = dataset["accession"]
        results.drop(["accession", "pident", "length", "mismatch", "gapopen", "qstart", "qend",
                        "sstart", "evalue", "bitscore", "name"], axis=1, inplace=True)

        results_ids = results["qseqid"]
        not_in_results = dataset[~dataset_ids.isin(results_ids)]
        not_in_results.drop(["sequence"], axis=1, inplace=True)

        if not_in_results.shape[0] > 0:
            for column in results.columns:
                if column != "qseqid":
                    not_in_results.loc[:, column] = 0.0

            not_in_results.drop(["name"], axis=1, inplace=True)
            not_in_results.columns = results.columns
            results = pd.concat([results, not_in_results])
            
        results.drop_duplicates(subset=["qseqid"], inplace=True)
        # Create a new column with the custom order as a categorical type
        results['CustomOrder'] = pd.Categorical(results['qseqid'], categories=dataset["accession"], ordered=True)
        results.sort_values('CustomOrder', inplace=True)
        results.drop(columns=["CustomOrder"], inplace=True)
        results.reset_index(drop=True, inplace=True)
        os.remove("temp_results_file")

        test_dataset = pd.read_csv(os.path.join(base_path, f"test_datasets/test_dataset_{i}.csv"))
        y_true = test_dataset.iloc[:, 11:]
        y_pred = results.iloc[:, 9:]
        from sklearn.metrics import f1_score, precision_score, recall_score
        mf1 = f1_score(y_true, y_pred, average='macro')
        wf1 = f1_score(y_true, y_pred, average='weighted')
        mrecall = recall_score(y_true, y_pred, average='macro')
        wrecall = recall_score(y_true, y_pred, average='weighted')
        mprecision = precision_score(y_true, y_pred, average='macro')
        wprecision = precision_score(y_true, y_pred, average='weighted')

        level_1, level_2, level_3, level_4 = get_ec_levels(y_pred.columns)

        level_1_f1 = f1_score(y_true.iloc[:, level_1], y_pred.iloc[:, level_1], average='macro')
        level_2_f1 = f1_score(y_true.iloc[:, level_2], y_pred.iloc[:, level_2], average='macro')
        level_3_f1 = f1_score(y_true.iloc[:, level_3], y_pred.iloc[:, level_3], average='macro')
        level_4_f1 = f1_score(y_true.iloc[:, level_4], y_pred.iloc[:, level_4], average='macro')

        level_1_wf1 = f1_score(y_true.iloc[:, level_1], y_pred.iloc[:, level_1], average='weighted')
        level_2_wf1 = f1_score(y_true.iloc[:, level_2], y_pred.iloc[:, level_2], average='weighted')
        level_3_wf1 = f1_score(y_true.iloc[:, level_3], y_pred.iloc[:, level_3], average='weighted')
        level_4_wf1 = f1_score(y_true.iloc[:, level_4], y_pred.iloc[:, level_4], average='weighted')

        results_dataframe_i = pd.DataFrame({"fold": [i], "mf1": [mf1], "wf1": [wf1], "mrecall": [mrecall], "wrecall": [wrecall], "mprecision": [mprecision], "wprecision": [wprecision],
                                            "level_1_f1": [level_1_f1], "level_2_f1": [level_2_f1], "level_3_f1": [level_3_f1], "level_4_f1": [level_4_f1],
                                            "level_1_wf1": [level_1_wf1], "level_2_wf1": [level_2_wf1], "level_3_wf1": [level_3_wf1], "level_4_wf1": [level_4_wf1]})
        results_dataframe = results_dataframe.append(results_dataframe_i)
    
    results_dataframe.to_csv(os.path.join(base_path, "blast_results_plants.csv"), index=False)

base_path = "/home/jcapela/plants_ec_number_prediction/ec_numbers_prediction/plants_pipeline/data/plants/"
generate_metrics(base_path)

# if __name__ == "__main__":
#     base_path = "/home/jcapela/plants_ec_number_prediction/ec_numbers_prediction/plants_pipeline/data/"
#     create_database_all_data(base_path)
#     generate_results_for_all_data_case_studies(base_path)