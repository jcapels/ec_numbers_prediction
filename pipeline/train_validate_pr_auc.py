from sklearn.metrics import f1_score
import torch
from torch.nn import BCELoss
from plants_sm.models.lightning_model import InternalLightningModule
from plants_sm.models.fc.fc import DNN
from torch.optim.lr_scheduler import ReduceLROnPlateau

from plants_sm.models.lightning_model import InternalLightningModel
from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset

from plants_sm.io.pickle import write_pickle

from plants_sm.models.ec_number_prediction.d_space import DSPACEModel
from plants_sm.models.ec_number_prediction.deepec import DeepECCNN

from plants_sm.featurization.encoding.one_hot_encoder import OneHotEncoder
from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer

import os

class ModelECNumber(InternalLightningModule):

    def __init__(self, input_dim, layers, classification_neurons, metric=None, learning_rate = 1e-3,  
                 scheduler = False) -> None:

        self.layers = layers
        self.classification_neurons = classification_neurons
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        super().__init__(metric=metric)
        self._create_model()

    def _update_constructor_parameters(self):

        self._contructor_parameters.update({

            "layers": self.layers,
            "classification_neurons": self.classification_neurons,
            "input_dim": self.input_dim,
            "learning_rate": self.learning_rate,
            "scheduler": self.scheduler,

        })

    def _create_model(self):
        self.fc_model = DNN(self.input_dim, self.layers, self.classification_neurons, batch_norm=True, last_sigmoid=True, 
                            dropout=None)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.fc_model.parameters()}], lr=self.learning_rate)

        # Define a custom learning rate scheduler using LambdaLR
        if self.scheduler:
            scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min'), 'monitor': 'val_loss'}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, x):
        return self.fc_model(x)

    def compute_loss(self, logits, y):
        return BCELoss()(logits, y)
    

def prepare_dataset(dataset_path, features_path):

    dataset = SingleInputDataset.from_csv(dataset_path, instances_ids_field="accession", representation_field="sequence",
                                        labels_field=slice(8, -1))
    
    if features_path is not None:
        dataset.load_features(features_path)
    return dataset

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def train_evaluate_pr_auc(datasets_path: str, model: InternalLightningModel, features_path: str, model_name: str, work_path: str):

    os.makedirs(os.path.join(work_path, f"{model_name}_predictions"), exist_ok=True)

    dataset_path = os.path.join(datasets_path, f"train.csv")
    train_dataset = prepare_dataset(dataset_path, features_path)

    model.fit(train_dataset=train_dataset)

    dataset_path = os.path.join(datasets_path, f"validation.csv")
    validation_dataset = prepare_dataset(dataset_path, features_path)

    predictions = model.predict_proba(validation_dataset)

    write_pickle(os.path.join(work_path, f"{model_name}_predictions", f"validation_{model_name}_predictions.pkl"), predictions)
    
    dataset_path = os.path.join(datasets_path, f"test.csv")
    validation_dataset = prepare_dataset(dataset_path, features_path)

    predictions = model.predict_proba(validation_dataset)
    write_pickle(os.path.join(work_path, f"{model_name}_predictions", f"test_{model_name}_predictions.pkl"), predictions)
    

def train_prot_bert(dataset_path, features_path, work_path):

    module = ModelECNumber(input_dim=1024, layers=[2560], classification_neurons=2771, 
                learning_rate=0.0001)
            
    model = InternalLightningModel(module=module, max_epochs=30,
            batch_size=64,
            devices=[3],
            accelerator="gpu",
            )

    train_evaluate_pr_auc(dataset_path, model, features_path, "prot_bert", work_path)


def train_esm1b(dataset_path, features_path, work_path):

    module = ModelECNumber(input_dim=1280, layers=[2560, 5120], classification_neurons=2771, 
                learning_rate=0.0001)
            
    model = InternalLightningModel(module=module, max_epochs=30,
            batch_size=64,
            devices=[3],
            accelerator="gpu",
            )
    
    train_evaluate_pr_auc(dataset_path, model, features_path, "esm1b", work_path)


def train_esm2_3b(dataset_path, features_path, work_path):

    module = ModelECNumber(input_dim=2560, layers=[2560], classification_neurons=2771, 
                learning_rate=0.0001)
            
    model = InternalLightningModel(module=module, max_epochs=30,
            batch_size=64,
            devices=[3],
            accelerator="gpu",
            )

    train_evaluate_pr_auc(dataset_path, model, features_path, "esm2_3B", work_path)

class DSPACEEC(InternalLightningModule):

    def __init__(self, classification_neurons, metric=None, 
                 scheduler = False) -> None:

        self.classification_neurons = classification_neurons
        self.scheduler = scheduler
        super().__init__(metric=metric)
        self._create_model()

    def _update_constructor_parameters(self):

        self._contructor_parameters.update({

            "classification_neurons": self.classification_neurons,
            "scheduler": self.scheduler,

        })

    def _create_model(self):
        self.d_space = DSPACEModel(20, 884, self.classification_neurons)
        
    def configure_optimizers(self):
        optimizer = torch.optim.NAdam([{'params': self.d_space.parameters()}], lr=0.001)

        # Define a custom learning rate scheduler using LambdaLR
        if self.scheduler:
            scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min'), 'monitor': 'val_loss'}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, x):
        return self.d_space(x)

    def compute_loss(self, logits, y):
        return BCELoss()(logits, y)
    
class DeepEC(InternalLightningModule):

    def __init__(self, classification_neurons, metric=None, 
                 scheduler = False) -> None:

        self.classification_neurons = classification_neurons
        self.scheduler = scheduler
        super().__init__(metric=metric)
        self._create_model()

    def _update_constructor_parameters(self):

        self._contructor_parameters.update({

            "classification_neurons": self.classification_neurons,
            "scheduler": self.scheduler,

        })

    def _create_model(self):
        self.deep_ec = DeepECCNN(128, 884, 20, [4, 8, 16], 2, 512, self.classification_neurons)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.deep_ec.parameters()}], lr=0.009999999776482582, 
                                     betas=(0.9, 0.999), eps=1e-7)

        # Define a custom learning rate scheduler using LambdaLR
        if self.scheduler:
            scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min'), 'monitor': 'val_loss'}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, x):
        return self.deep_ec(x)

    def compute_loss(self, logits, y):
        return BCELoss()(logits, y)
    

def train_evaluate_pr_auc_one_hot_encoding(datasets_path: str, model: InternalLightningModel, 
                                  transformers: list, model_name: str, work_path: str):

    os.makedirs(os.path.join(work_path, f"{model_name}_predictions"), exist_ok=True)

    dataset_path = os.path.join(datasets_path, f"train.csv")
    train_dataset = prepare_dataset(dataset_path, None)

    for transformer in transformers:
        transformer.fit_transform(train_dataset)

    model.fit(train_dataset=train_dataset)

    dataset_path = os.path.join(datasets_path, f"test.csv")
    test_dataset = prepare_dataset(dataset_path, None)

    for transformer in transformers:
        transformer.transform(test_dataset)

    predictions = model.predict_proba(test_dataset)

    write_pickle(os.path.join(work_path, f"{model_name}_predictions", f"test_{model_name}_predictions.pkl"), predictions)

    dataset_path = os.path.join(datasets_path, f"validation.csv")
    test_dataset = prepare_dataset(dataset_path, None)

    for transformer in transformers:
        transformer.transform(test_dataset)

    predictions = model.predict_proba(test_dataset)
    write_pickle(os.path.join(work_path, f"{model_name}_predictions", f"validation_{model_name}_predictions.pkl"), predictions)



def train_dspace(dataset_path, work_path):

    module = DSPACEEC(classification_neurons=2771)

    model = InternalLightningModel(module=module, max_epochs=30,
            batch_size=64,
            devices=[3],
            accelerator="gpu",
            )
    
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    transformers = [ProteinStandardizer(), OneHotEncoder(max_length=884, alphabet=amino_acids)]

    train_evaluate_pr_auc_one_hot_encoding(dataset_path, model, transformers, "dspace", work_path)


def train_deepec(dataset_path, work_path):

    module = DeepEC(classification_neurons=2771)

    model = InternalLightningModel(module=module, max_epochs=30,
            batch_size=64,
            devices=[2],
            accelerator="gpu",
            )
    
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    transformers = [ProteinStandardizer(), OneHotEncoder(max_length=884, alphabet=amino_acids)]

    train_evaluate_pr_auc_one_hot_encoding(dataset_path, model, transformers, "deep_ec", work_path)

if __name__ == "__main__":
    base_dir = "/home/jcapela/ec_number_prediction_version_2/ec_numbers_prediction/"
    train_esm2_3b(f"{base_dir}pr_auc_validation/", f"{base_dir}normal_splits_uniref90/features_llms/merged_dataset_features_esm2_3B", f"{base_dir}pr_auc_validation")
    train_prot_bert(f"{base_dir}pr_auc_validation/", f"{base_dir}normal_splits_uniref90/features_llms/merged_dataset_features_prot_bert", f"{base_dir}pr_auc_validation")
    train_esm1b(f"{base_dir}pr_auc_validation/", f"{base_dir}normal_splits_uniref90/features_llms/merged_dataset_features_esm1b",f"{base_dir}pr_auc_validation")
    train_dspace(f"{base_dir}pr_auc_validation/", f"{base_dir}pr_auc_validation")
    train_deepec(f"{base_dir}pr_auc_validation/", f"{base_dir}pr_auc_validation")