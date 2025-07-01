import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.nn import BCELoss
from plants_sm.models.lightning_model import InternalLightningModule
from plants_sm.models.fc.fc import DNN
from torch.optim.lr_scheduler import ReduceLROnPlateau

from plants_sm.models.lightning_model import InternalLightningModel
from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset

from plants_sm.models.ec_number_prediction.d_space import DSPACEModel
from plants_sm.models.ec_number_prediction.deepec import DeepECCNN

from plants_sm.io.pickle import write_pickle, read_pickle

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
        
    def reset_weights(self):
        """
        refs:
            - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
            - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
            - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        """

        @torch.no_grad()
        def weight_reset(m):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self.fc_model.apply(fn=weight_reset)
    

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

def train_predict_model(datasets_path: str, model: InternalLightningModel, features_path: str, model_name: str, work_path: str):

    os.makedirs(os.path.join(work_path, f"{model_name}_predictions"), exist_ok=True)

    for i in range(5):
        dataset_path = os.path.join(datasets_path, f"train_{i}.csv")
        train_dataset = prepare_dataset(dataset_path, features_path)
        
        model.module.reset_weights()
        model.fit(train_dataset=train_dataset)

        dataset_path = os.path.join(datasets_path, f"test_{i}.csv")
        test_dataset = prepare_dataset(dataset_path, features_path)

        predictions = model.predict_proba(test_dataset)

        write_pickle(os.path.join(work_path, f"{model_name}_predictions", f"test_{i}_{model_name}_predictions.pkl"), predictions)
        


def train_prot_bert(dataset_path, features_path, work_path):

    module = ModelECNumber(input_dim=1024, layers=[2560], classification_neurons=2771, 
                learning_rate=0.0001)
            
    model = InternalLightningModel(module=module, max_epochs=30,
            batch_size=64,
            devices=[3],
            accelerator="gpu",
            )

    train_predict_model(dataset_path, model, features_path, "prot_bert", work_path)


def train_esm1b(dataset_path, features_path, work_path):

    module = ModelECNumber(input_dim=1280, layers=[2560, 5120], classification_neurons=2771, 
                learning_rate=0.0001)
            
    model = InternalLightningModel(module=module, max_epochs=30,
            batch_size=64,
            devices=[3],
            accelerator="gpu",
            )
    
    train_predict_model(dataset_path, model, features_path, "esm1b", work_path)


def train_esm2_3b(dataset_path, features_path, work_path):

    module = ModelECNumber(input_dim=2560, layers=[2560], classification_neurons=2771, 
                learning_rate=0.0001)
            
    model = InternalLightningModel(module=module, max_epochs=30,
            batch_size=64,
            devices=[3],
            accelerator="gpu",
            )

    train_predict_model(dataset_path, model, features_path, "esm2_3B", work_path)

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

    def reset_weights(self):
        """
        refs:
            - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
            - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
            - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        """

        @torch.no_grad()
        def weight_reset(m):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self.d_space.apply(fn=weight_reset)
        
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

    def reset_weights(self):
        """
        refs:
            - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
            - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
            - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        """

        @torch.no_grad()
        def weight_reset(m):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self.deep_ec.apply(fn=weight_reset)
        
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
    

def _train_one_hot_encoding_model(datasets_path: str, model: InternalLightningModel, 
                                  transformers: list, model_name: str, work_path: str):

    os.makedirs(os.path.join(work_path, f"{model_name}_predictions"), exist_ok=True)

    for i in range(5):
        dataset_path = os.path.join(datasets_path, f"train_{i}.csv")
        train_dataset = prepare_dataset(dataset_path, None)

        for transformer in transformers:
            transformer.fit_transform(train_dataset)

        model.module.reset_weights()

        model.fit(train_dataset=train_dataset)

        dataset_path = os.path.join(datasets_path, f"test_{i}.csv")
        test_dataset = prepare_dataset(dataset_path, None)

        for transformer in transformers:
            transformer.transform(test_dataset)

        predictions = model.predict_proba(test_dataset)

        write_pickle(os.path.join(work_path, f"{model_name}_predictions", f"test_{i}_{model_name}_predictions.pkl"), predictions)
        


def train_dspace(dataset_path, work_path):

    module = DSPACEEC(classification_neurons=2771)

    model = InternalLightningModel(module=module, max_epochs=30,
            batch_size=64,
            devices=[3],
            accelerator="gpu",
            )
    
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    transformers = [ProteinStandardizer(), OneHotEncoder(max_length=884, alphabet=amino_acids)]

    _train_one_hot_encoding_model(dataset_path, model, transformers, "dspace", work_path)


def train_deepec(dataset_path, work_path):

    module = DeepEC(classification_neurons=2771)

    model = InternalLightningModel(module=module, max_epochs=30,
            batch_size=64,
            devices=[2],
            accelerator="gpu",
            )
    
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    transformers = [ProteinStandardizer(), OneHotEncoder(max_length=884, alphabet=amino_acids)]

    _train_one_hot_encoding_model(dataset_path, model, transformers, "deep_ec", work_path)

import pandas as pd

def _get_results_for_blast(folder, tool_name, ground_truth, i, ):
    test_tool_prediction = pd.read_csv(os.path.join(folder, f"blast_predictions/test_{tool_name}_predictions_{i}.csv"))
    test_tool_prediction.drop_duplicates(subset=["qseqid"], inplace=True)
    # Create a new column with the custom order as a categorical type
    test_tool_prediction['CustomOrder'] = pd.Categorical(test_tool_prediction['qseqid'], 
                                                            categories=ground_truth["accession"], 
                                                            ordered=True)
    test_tool_prediction.sort_values('CustomOrder', inplace=True)
    test_tool_prediction.drop(columns=["CustomOrder"], inplace=True)
    test_tool_prediction.reset_index(drop=True, inplace=True)
    test_tool_predictions = test_tool_prediction.iloc[:, 6:].to_numpy()

    return test_tool_predictions

def _determine_ensemble_predictions(model_predictions):
    predictions_voting = np.zeros_like(model_predictions[0])

    for i, model_prediction in enumerate(model_predictions):
        model_predictions[i] = np.array(model_prediction)

    for i in range(model_predictions[0].shape[0]):
        # Combine conditions into a single array and sum along the second axis
        combined_conditions = np.sum(np.array([model_predictions[j][i] for j in range(len(model_predictions))]), axis=0)

        # Apply the threshold condition
        predictions_voting[i] = (combined_conditions >= 2).astype(int)

    # If you want to ensure the resulting array is of integer type
    predictions_voting = predictions_voting.astype(int)
    return predictions_voting

def get_results_ensembles(predictions_folder, work_path):
    # start with model ensembles

    models_ensemble_predictions_folder = os.path.join(predictions_folder, "models_ensemble_predictions")
    os.makedirs(models_ensemble_predictions_folder, exist_ok=True)

    models_blast_predictions_folder = os.path.join(predictions_folder, "models_blast_predictions")
    os.makedirs(models_blast_predictions_folder, exist_ok=True)

    for j in range(5):
        esm1b_predictions = read_pickle(os.path.join(predictions_folder, "esm1b_predictions", f"test_{j}_esm1b_predictions.pkl"))
        esm2_3b_predictions = read_pickle(os.path.join(predictions_folder, "esm2_3B_predictions", f"test_{j}_esm2_3B_predictions.pkl"))
        protbert_predictions = read_pickle(os.path.join(predictions_folder, "prot_bert_predictions", f"test_{j}_prot_bert_predictions.pkl"))
        
        model_predictions = [esm1b_predictions, esm2_3b_predictions, protbert_predictions]
        for i, _ in enumerate(model_predictions):
            model_predictions[i] = (model_predictions[i] >= 0.5).astype(int)

        ensemble_predictions =_determine_ensemble_predictions(model_predictions)

        write_pickle(os.path.join(models_ensemble_predictions_folder,f"test_{j}_models_ensemble_predictions.pkl"), ensemble_predictions)

        ground_truth = pd.read_csv(os.path.join(work_path, f"monte_carlo_splits/test_{j}.csv"))
        
        blast_predictions = _get_results_for_blast(predictions_folder, "blast", ground_truth, j)

        model_predictions.append(blast_predictions)
        ensemble_predictions =_determine_ensemble_predictions(model_predictions)

        write_pickle(os.path.join(models_blast_predictions_folder,f"test_{j}_models_blast_predictions.pkl"), ensemble_predictions)



if __name__ == "__main__":
    base_dir = "/home/jcapela/ec_number_prediction_version_2/ec_numbers_prediction/"
    train_esm2_3b(f"{base_dir}normal_splits_uniref90/monte_carlo_splits/", f"{base_dir}normal_splits_uniref90/features_llms/merged_dataset_features_esm2_3B", f"{base_dir}normal_splits_uniref90")
    train_prot_bert(f"{base_dir}normal_splits_uniref90/monte_carlo_splits/", f"{base_dir}normal_splits_uniref90/features_llms/merged_dataset_features_prot_bert", f"{base_dir}normal_splits_uniref90")
    train_esm1b(f"{base_dir}normal_splits_uniref90/monte_carlo_splits", f"{base_dir}normal_splits_uniref90/features_llms/merged_dataset_features_esm1b",f"{base_dir}normal_splits_uniref90")
    train_dspace(f"{base_dir}normal_splits_uniref90/monte_carlo_splits",f"{base_dir}normal_splits_uniref90")
    train_deepec(f"{base_dir}normal_splits_uniref90/monte_carlo_splits",f"{base_dir}normal_splits_uniref90")
    get_results_ensembles(f"{base_dir}normal_splits_uniref90/predictions/", f"{base_dir}normal_splits_uniref90/")