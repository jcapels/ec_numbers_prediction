import os
import numpy as np
import optuna
import pandas as pd
from plants_sm.hyperparameter_optimization.experiment import Experiment
from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset

from sklearn.metrics import f1_score
from ec_number_prediction.transfer_learning.models import FineTuneModelECNumber, ModelECNumber
from plants_sm.models.lightning_model import InternalLightningModel
from lightning.pytorch.callbacks import EarlyStopping

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def prepare_dataset(dataset_path, features_path):
    dataset = SingleInputDataset.from_csv(dataset_path, instances_ids_field="accession", representation_field="sequence",
                                        labels_field=slice(8, -1))
    dataset.load_features(features_path)
    return dataset

def train_prot_bert(dataset_path, features_path):

    dataset = prepare_dataset(dataset_path, features_path)

    module = ModelECNumber(input_dim=1024, layers=[2560], classification_neurons=5405, 
                metric=f1_macro, learning_rate=0.0001)
            
    model = InternalLightningModel(module=module, max_epochs=30,
            batch_size=64,
            devices=[3],
            accelerator="gpu",
            )

    model.fit(dataset)
    model.save("model_prot_bert_no_plants")

def train_esm1b(dataset_path, features_path):

    dataset = prepare_dataset(dataset_path, features_path)

    module = ModelECNumber(input_dim=1280, layers=[2560, 5120], classification_neurons=5405, 
                metric=f1_macro, learning_rate=0.0001)
            
    model = InternalLightningModel(module=module, max_epochs=30,
            batch_size=64,
            devices=[3],
            accelerator="gpu",
            )

    model.fit(dataset)
    model.save("model_esm1b_no_plants")

def train_esm2_3b(dataset_path, features_path):

    dataset = prepare_dataset(dataset_path, features_path)

    module = ModelECNumber(input_dim=2560, layers=[2560], classification_neurons=5405, 
                metric=f1_macro, learning_rate=0.0001)
            
    model = InternalLightningModel(module=module, max_epochs=30,
            batch_size=64,
            devices=[3],
            accelerator="gpu",
            )

    model.fit(dataset)
    model.save("model_esm2_3b_no_plants")

if __name__ == "__main__":
    #train_prot_bert("embeddings/merged_dataset_no_plants.csv", "embeddings/protbert_vectors/merged")
    # train_esm1b("embeddings/merged_dataset_no_plants.csv", "embeddings/esm1b/merged")
    train_esm2_3b("embeddings/merged_dataset_no_plants.csv", "embeddings/esm2_3b/merged")
