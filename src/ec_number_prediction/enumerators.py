from enum import Enum

class ModelsDownloadPaths(Enum):

    DNN_ESM1b_ALL_DATA = "https://nextcloud.bio.di.uminho.pt/s/toT5ExtdLQH2sKT/download/DNN_ESM1b_all_data.zip"
    DNN_PROTBERT_ALL_DATA = "https://nextcloud.bio.di.uminho.pt/s/L9omjLK6jATQi2D/download/DNN_ProtBERT_all_data.zip"
    DNN_ESM2_3B_ALL_DATA = "https://nextcloud.bio.di.uminho.pt/s/5fjawZHiQmojQmc/download/DNN_ESM2_3B_all_data.zip"
    DNN_ESM2_3B_TRAIN_VALID = "https://nextcloud.bio.di.uminho.pt/s/4Mzj3aQD2Mf2kYn/download/DNN_ESM2_3B_trial_2_train_plus_validation.zip"
    DNN_ESM1b_TRAIN_VALID = "https://nextcloud.bio.di.uminho.pt/s/pJiexyb45js8KQx/download/DNN_ESM1b_trial_4_train_plus_validation.zip"
    DNN_PROTBERT_TRAIN_VALID = "https://nextcloud.bio.di.uminho.pt/s/cmQrk44NRwcicCz/download/DNN_ProtBERT_trial_2_train_plus_validation.zip"

class BLASTDownloadPaths(Enum):

    BLAST_ALL_DATA = "https://nextcloud.bio.di.uminho.pt/s/Wka96R8ADGqLjYT/download/BLAST_all_data.zip"
    BLAST_TRAIN_VALID = "https://nextcloud.bio.di.uminho.pt/s/JQZjbsk6cYF7tGc/download/BLAST_train_plus_validation.zip"

class BLASTDatabases(Enum):

    BLAST_ALL_DATA = "all_data_database"
    BLAST_TRAIN_VALID = "train_database"