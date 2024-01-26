from enum import Enum

class ModelsDownloadPaths(Enum):

    DNN_ESM1b_ALL_DATA = "https://nextcloud.bio.di.uminho.pt/s/toT5ExtdLQH2sKT/download/DNN_ESM1b_all_data.zip"
    DNN_PROTBERT_ALL_DATA = "https://nextcloud.bio.di.uminho.pt/s/L9omjLK6jATQi2D/download/DNN_ProtBERT_all_data.zip"
    DNN_ESM2_3B_ALL_DATA = "https://nextcloud.bio.di.uminho.pt/s/5fjawZHiQmojQmc/download/DNN_ESM2_3B_all_data.zip"

class BLASTDownloadPaths(Enum):

    BLAST_ALL_DATA = "https://nextcloud.bio.di.uminho.pt/s/Wka96R8ADGqLjYT/download/BLAST_all_data.zip"

class BLASTDatabases(Enum):

    BLAST_ALL_DATA = "all_data_database"