from enum import Enum

class ModelsDownloadPaths(Enum):

    DNN_ESM1b_ALL_DATA = "https://zenodo.org/records/11380947/files/DNN_ESM1b_all_data.zip?download=1"
    DNN_PROTBERT_ALL_DATA = "https://zenodo.org/records/11380947/files/DNN_ProtBERT_all_data.zip?download=1"
    DNN_ESM2_3B_ALL_DATA = "https://zenodo.org/records/11380947/files/DNN_ESM2_3B_all_data.zip?download=1"
    DNN_ESM2_3B_TRAIN_VALID = "https://zenodo.org/records/11380947/files/DNN_ESM2_3B_trial_2_train_plus_validation.zip?download=1"
    DNN_ESM1b_TRAIN_VALID = "https://zenodo.org/records/11380947/files/DNN_ESM1b_trial_4_train_plus_validation.zip?download=1"
    DNN_PROTBERT_TRAIN_VALID = "https://zenodo.org/records/11380947/files/DNN_ProtBERT_trial_2_train_plus_validation.zip?download=1"

class BLASTDownloadPaths(Enum):

    BLAST_ALL_DATA = "https://zenodo.org/records/11380947/files/BLAST_all_data.zip?download=1"
    BLAST_TRAIN_VALID = "https://zenodo.org/records/11380947/files/BLAST_train_plus_validation.zip?download=1"

class BLASTDatabases(Enum):

    BLAST_ALL_DATA = "all_data_database"
    BLAST_TRAIN_VALID = "train_database"