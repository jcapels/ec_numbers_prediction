from logging import Logger
import os
import sys
from typing import Union

import numpy as np
from plants_sm.models.ec_number_prediction.clean import CLEANSupConH
from plants_sm.models.ec_number_prediction.d_space import DSPACE
import torch
from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset
from plants_sm.models.fc.fc import DNN
from plants_sm.models.pytorch_model import PyTorchModel
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, coverage_error, average_precision_score

from plants_sm.models.ec_number_prediction.deepec import DeepEC, DeepECCNN

logger = Logger("train_model")


def precision_score_macro(y_true, y_pred):
    return precision_score(y_true, y_pred, average="macro", zero_division=0)


def f1_score_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


class PipelineRunner:
    """
    Class to run the pipeline for training and evaluating the models.

    Parameters
    ----------
    features_folder: str
        Folder where the features are saved.
    features_base_directory: str
        Base directory where all the features are saved. (Absolute path)
    datasets_directory: str
        Directory where the datasets are saved. (Absolute path)
    project_base_dir: str
        Base directory of the project. (Absolute path)
    losses_directory: str
        Directory where the losses are saved. (Absolute path)
    metrics_directory: str
        Directory where the results are saved. (Absolute path)
    debug: bool
        If True, the model will be trained with a small dataset.
    last_sigmoid: bool
        If True, the model will have a sigmoid layer at the end.
    batch_size: int or None (default=None)
        Batch size used to load the datasets.
    """

    def __init__(self, features_folder: str,
                 features_base_directory: str = "/scratch/jribeiro/results/",
                 datasets_directory: str = "/scratch/jribeiro/ec_number_prediction/final_data",
                 project_base_dir: str = "/scratch/jribeiro/ec_number_prediction/",
                 losses_directory: str = "/scratch/jribeiro/results/losses",
                 metrics_directory: str = "/scratch/jribeiro/results/metrics",
                 debug: bool = False, last_sigmoid: bool = False, batch_size: int = None):

        self.features_folder = features_folder
        self.features_base_directory = features_base_directory
        self.datasets_directory = datasets_directory
        self.project_base_dir = project_base_dir
        self.debug = debug
        self.last_sigmoid = last_sigmoid
        self.batch_size = batch_size

        if not os.path.exists(losses_directory):
            os.makedirs(losses_directory)

        if not os.path.exists(metrics_directory):
            os.makedirs(metrics_directory)

        self.losses_directory = losses_directory
        self.metrics_directory = metrics_directory

        self.instantiate_paths()

    def generate_metrics(self, dataset: SingleInputDataset, model: PyTorchModel) -> dict:
        """
        Generate the metrics for the dataset and model.

        Parameters
        ----------
        dataset: SingleInputDataset
            Dataset to be used to generate the metrics.
        model: PyTorchModel
            Model to be used to generate the metrics.

        Returns
        -------
        metric_values: dict
            Dictionary with the metric values.
        """

        print("Generating metrics")

        if self.batch_size is not None:
            targets = np.empty((0, 2771))
            i = 0
            predictions = model.predict(dataset)
            probas = model.predict_proba(dataset)
            while dataset.next_batch():
                targets = np.concatenate((targets, dataset.y))
                i += 1
        else:
            predictions = model.predict(dataset)
            probas = model.predict_proba(dataset)
            targets = dataset.y

        precision_score_macro = precision_score(targets, predictions, average="macro")
        recall_score_macro = recall_score(targets, predictions, average="macro")
        f1_score_macro = f1_score(targets, predictions, average="macro")

        coverage_error_ = coverage_error(targets, probas)
        average_precision_score_macro = average_precision_score(targets, probas, average="macro")

        print("Metrics generated")

        metric_values = {
            "coverage_error": coverage_error_,
            "precision_score_macro": precision_score_macro, "recall_score_macro": recall_score_macro,
            "f1_score_macro": f1_score_macro,
            "average_precision_score_macro": average_precision_score_macro}

        labels_first_level_score = probas[:, :7]
        labels_second_level_score = probas[:, 7:84]
        labels_third_level_score = probas[:, 84:314]
        labels_fourth_level_score = probas[:, 314:]

        scores = [labels_first_level_score, labels_second_level_score, labels_third_level_score,
                  labels_fourth_level_score]

        prediction_first_level = predictions[:, :7]
        prediction_second_level = predictions[:, 7:84]
        prediction_third_level = predictions[:, 84:314]
        prediction_fourth_level = predictions[:, 314:]

        all_predictions = [prediction_first_level, prediction_second_level, prediction_third_level,
                           prediction_fourth_level]

        labels_first_level = targets[:, :7]
        labels_second_level = targets[:, 7:84]
        labels_third_level = targets[:, 84:314]
        labels_fourth_level = targets[:, 314:]

        labels = [labels_first_level, labels_second_level, labels_third_level, labels_fourth_level]

        for label_level in range(4):
            print(f"Metrics for level {label_level + 1}")

            precision_score_macro = precision_score(labels[label_level], all_predictions[label_level], average="macro")
            recall_score_macro = recall_score(labels[label_level], all_predictions[label_level], average="macro")
            f1_score_macro = f1_score(labels[label_level], all_predictions[label_level], average="macro")
            coverage_error_ = coverage_error(labels[label_level], scores[label_level])

            metric_values[f"precision_score_macro_level_{label_level + 1}"] = precision_score_macro
            metric_values[f"recall_score_macro_level_{label_level + 1}"] = recall_score_macro
            metric_values[f"f1_score_macro_level_{label_level + 1}"] = f1_score_macro

            metric_values[f"coverage_error_level_{label_level + 1}"] = coverage_error_

        return metric_values

    def prepare_dataset(self) -> (SingleInputDataset, SingleInputDataset, SingleInputDataset):
        """
        Prepare the datasets to be used in the training and evaluation of the models.

        Returns
        -------
        train_dataset: SingleInputDataset
            Train dataset.
        validation_dataset: SingleInputDataset
            Validation dataset.
        test_dataset: SingleInputDataset
            Test dataset.
        """
        if self.debug:
            slice_ = slice(7, 2778)
        else:
            slice_ = slice(8, 2779)

        logger.info("Loading the training and validation datasets")
        train_dataset = SingleInputDataset.from_csv(self.train_dataset_path,
                                                    instances_ids_field="accession", representation_field="sequence",
                                                    labels_field=slice_, batch_size=self.batch_size)

        validation_dataset = SingleInputDataset.from_csv(self.validation_dataset_path,
                                                         instances_ids_field="accession",
                                                         representation_field="sequence",
                                                         labels_field=slice_, batch_size=self.batch_size)

        test_dataset = SingleInputDataset.from_csv(self.test_dataset_path,
                                                   instances_ids_field="accession",
                                                   representation_field="sequence",
                                                   labels_field=slice_, batch_size=self.batch_size)

        train_dataset.load_features(self.train_dataset_features_path)
        validation_dataset.load_features(self.validation_dataset_features_path)
        test_dataset.load_features(self.test_dataset_features_path)
        test_dataset = None

        logger.info("datasets loaded")
        return train_dataset, validation_dataset, test_dataset

    def instantiate_paths(self):
        """
        Instantiate the paths to the datasets and features.

        Returns
        -------

        """
        self.train_dataset_features_path = os.path.join(self.features_base_directory, self.features_folder, "train")
        self.validation_dataset_features_path = os.path.join(self.features_base_directory, self.features_folder,
                                                             "validation")
        self.test_dataset_features_path = os.path.join(self.features_base_directory, self.features_folder, "test")

        self.merged_dataset_path = os.path.join(self.datasets_directory, "merged_dataset.csv")

        self.train_dataset_path = os.path.join(self.datasets_directory, "train_shuffled.csv")
        self.validation_dataset_path = os.path.join(self.datasets_directory, "validation_shuffled.csv")
        self.test_dataset_path = os.path.join(self.datasets_directory, "test.csv")

    def instantiate_model(self, model: Union[callable, PyTorchModel],
                          num_tokens: int, input_size: int, num_labels: int, **kwargs):
        """
        Instantiate the model to be used in the training and evaluation.

        Parameters
        ----------
        model: Union[callable, PyTorchModel]
            Model to be used.
        num_tokens: int
            Number of tokens in the vocabulary. Only necessary if the model is a PyTorchModel (this is used only
            for DeepEC and DSPACE).
        input_size: int
            Input size of the model. Only necessary if the model is a PyTorchModel (this is used only for DeepEC and
            DSPACE).
        num_labels: int
            Number of labels. Only necessary if the model is a PyTorchModel (this is used only for DeepEC and DSPACE).
        kwargs

        Returns
        -------
        model: Union[callable, PyTorchModel]
            Model to be used.
        train_dataset: SingleInputDataset
            Train dataset.
        validation_dataset: SingleInputDataset
            Validation dataset.
        test_dataset: SingleInputDataset
            Test dataset.
        model_path: str
            Path where the model will be saved.
        kwargs: dict
            Dictionary with the parameters used in the training.
        """
        if isinstance(model, nn.Module):
            os.makedirs(os.path.join(self.features_base_directory, self.features_folder,
                                     f"{model.__class__.__name__}_{self.features_folder}"), exist_ok=True)
            if "model_name" in kwargs:
                model_path = os.path.join(self.features_base_directory, self.features_folder,
                                          f"{model.__class__.__name__}_{self.features_folder}", kwargs["model_name"])
            else:
                model_path = os.path.join(self.features_base_directory, self.features_folder,
                                          f"{model.__class__.__name__}_{self.features_folder}", "model")
                kwargs["model_name"] = f"{model.__class__.__name__}_{self.features_folder}"

        else:
            os.makedirs(os.path.join(self.features_base_directory, self.features_folder,
                                     f"{model.__name__}_{self.features_folder}"), exist_ok=True)
            if "model_name" in kwargs:
                model_path = os.path.join(self.features_base_directory, self.features_folder,
                                          f"{model.__class__.__name__}_{self.features_folder}", kwargs["model_name"])
            else:
                model_path = os.path.join(self.features_base_directory, self.features_folder,
                                          f"{model.__name__}_{self.features_folder}", "model")
                kwargs["model_name"] = f"{model.__name__}_{self.features_folder}"

        train_dataset, validation_dataset, test_dataset = self.prepare_dataset()

        loss_function = nn.BCELoss()

        kwargs["loss_function"] = loss_function

        kwargs["checkpoints_path"] = os.path.join(self.project_base_dir, "model_checkpoints")

        if isinstance(model, nn.Module):
            model = PyTorchModel(model=model, scheduler=ReduceLROnPlateau(kwargs["optimizer"], 'min'), **kwargs)
        else:
            model = model(num_tokens, input_size, num_labels, **kwargs)

        if self.debug or self.last_sigmoid:
            model.model.last_sigmoid = True

        if "device" in kwargs:
            model.device = kwargs["device"]

        return model, train_dataset, validation_dataset, test_dataset, model_path, kwargs

    def train_model_with_all_data(self, model: Union[callable, PyTorchModel],
                                  num_tokens: int, input_size: int, num_labels: int, **kwargs):
        """
        Train the model with all the data.

        Parameters
        ----------
        model: Union[callable, PyTorchModel]
            Model to be used.
        num_tokens: int
            Number of tokens in the vocabulary. Only necessary if the model is a PyTorchModel (this is used only
            for DeepEC and DSPACE).
        input_size: int
            Input size of the model. Only necessary if the model is a PyTorchModel (this is used only for DeepEC and
            DSPACE).
        num_labels: int
            Number of labels. Only necessary if the model is a PyTorchModel (this is used only for DeepEC and DSPACE).
        kwargs

        Returns
        -------

        """
        kwargs["model_name"] = f"{kwargs['model_name']}_all_data"
        kwargs["validation_loss_function"] = None
        kwargs["validation_metric"] = None

        if isinstance(model, nn.Module):
            os.makedirs(os.path.join(self.features_base_directory, self.features_folder,
                                     f"{model.__class__.__name__}_{self.features_folder}"), exist_ok=True)
            if "model_name" in kwargs:
                model_path = os.path.join(self.features_base_directory, self.features_folder,
                                          f"{model.__class__.__name__}_{self.features_folder}", kwargs["model_name"])
            else:
                model_path = os.path.join(self.features_base_directory, self.features_folder,
                                          f"{model.__class__.__name__}_{self.features_folder}", "model")
                kwargs["model_name"] = f"{model.__class__.__name__}_{self.features_folder}"

        else:
            os.makedirs(os.path.join(self.features_base_directory, self.features_folder,
                                     f"{model.__name__}_{self.features_folder}"), exist_ok=True)
            if "model_name" in kwargs:
                model_path = os.path.join(self.features_base_directory, self.features_folder,
                                          f"{model.__class__.__name__}_{self.features_folder}", kwargs["model_name"])
            else:
                model_path = os.path.join(self.features_base_directory, self.features_folder,
                                          f"{model.__name__}_{self.features_folder}", "model")
                kwargs["model_name"] = f"{model.__name__}_{self.features_folder}"

        slice_ = slice(8, 5751)

        logger.info("Loading the training and validation datasets")
        train_dataset = SingleInputDataset.from_csv(self.merged_dataset_path,
                                                    instances_ids_field="accession", representation_field="sequence",
                                                    labels_field=slice_, batch_size=self.batch_size)

        train_dataset.load_features(os.path.join(self.features_base_directory, self.features_folder, "merged"))

        logger.info("datasets loaded")

        loss_function = nn.BCELoss()

        kwargs["loss_function"] = loss_function

        kwargs["checkpoints_path"] = os.path.join(self.project_base_dir, "model_checkpoints")
        kwargs["logger_path"] = f"{kwargs['model_name']}.log"

        if isinstance(model, nn.Module):
            model = PyTorchModel(model=model, scheduler=ReduceLROnPlateau(kwargs["optimizer"], 'min'), **kwargs)
        else:
            model = model(num_tokens, input_size, num_labels, **kwargs)

        if self.debug or self.last_sigmoid:
            model.model.last_sigmoid = True

        if "device" in kwargs:
            model.device = kwargs["device"]

        model.fit(train_dataset=train_dataset)

        model.model.last_sigmoid = True
        model.device = "cpu"
        model.save(model_path)

        # Evaluate the model
        logger.info("Evaluating the model")

        model.history["loss"].to_csv(
            os.path.join(self.losses_directory, f"{kwargs['model_name']}_loss_all_dataset.csv"), index=False)

    def train_model_with_validation_train_merged(self, model: Union[callable, PyTorchModel],
                                                 num_tokens: int, input_size: int, num_labels: int, **kwargs):
        """
        Train the model with the validation and train datasets merged.

        Parameters
        ----------
        model: Union[callable, PyTorchModel]
            Model to be used.
        num_tokens: int
            Number of tokens in the vocabulary. Only necessary if the model is a PyTorchModel (this is used only
            for DeepEC and DSPACE).
        input_size: int
            Input size of the model. Only necessary if the model is a PyTorchModel (this is used only for DeepEC and
            DSPACE).
        num_labels: int
            Number of labels. Only necessary if the model is a PyTorchModel (this is used only for DeepEC and DSPACE).
        kwargs

        Returns
        -------

        """

        kwargs["model_name"] = f"{kwargs['model_name']}_merged"
        kwargs["validation_loss_function"] = None
        kwargs["validation_metric"] = None

        model, train_dataset, validation_dataset, test_dataset, model_path, kwargs = self.instantiate_model(model,
                                                                                                            num_tokens,
                                                                                                            input_size,
                                                                                                            num_labels,
                                                                                                            **kwargs)

        train_dataset.merge(validation_dataset)

        model.fit(train_dataset=train_dataset)

        model.model.last_sigmoid = True
        model.save(model_path)

        # Evaluate the model
        logger.info("Evaluating the model")

        train_metrics = self.generate_metrics(train_dataset, model)
        metrics_test = self.generate_metrics(test_dataset, model)

        logger.info(f"Train metrics: {train_metrics}")
        logger.info(f"Test metrics: {metrics_test}")

        os.makedirs(os.path.join(self.features_base_directory, "results", "metrics"), exist_ok=True)

        # create dataframe with the metrics for all datasets and models

        if os.path.exists(os.path.join(self.project_base_dir, "results", "metrics", "metrics_merged.csv")):
            metrics = pd.read_csv(os.path.join(self.project_base_dir, "results", "metrics", "metrics_merged.csv"))
        else:
            metrics = pd.DataFrame(columns=["model", "metric", "train", "test"])

        dataset_metrics = [train_metrics, metrics_test]
        dataset_names = ["train", "test"]
        for metric in train_metrics.keys():
            row = len(metrics)
            metrics.at[row, "model"] = model.model_name
            metrics.at[row, "metric"] = metric
            for i, dataset_metric in enumerate(dataset_metrics):
                metrics.at[row, dataset_names[i]] = dataset_metric[metric]

        os.makedirs(os.path.join(self.project_base_dir, "results", "metrics"), exist_ok=True)
        metrics.to_csv(os.path.join(self.project_base_dir, "results", "metrics", "metrics_merged.csv"), index=False)

        model.history["loss"].to_csv(os.path.join(self.losses_directory, f"{kwargs['model_name']}_loss_merged.csv"),
                                     index=False)
        model.history["metric_results"].to_csv(os.path.join(self.metrics_directory,
                                                            f"{kwargs['model_name']}_metric_result_merged.csv"),
                                               index=False)

    def train_model(self, model: Union[callable, PyTorchModel], num_tokens: int,
                    input_size: int, num_labels: int, **kwargs):
        """
        Train the model using only the training set and validation set for early stopping.

        Parameters
        ----------
        model: Union[callable, PyTorchModel]
            Model to be used.
        num_tokens: int
            Number of tokens in the vocabulary. Only necessary if the model is a PyTorchModel (this is used only
            for DeepEC and DSPACE).
        input_size: int
            Input size of the model. Only necessary if the model is a PyTorchModel (this is used only for DeepEC and
            DSPACE).
        num_labels: int
            Number of labels. Only necessary if the model is a PyTorchModel (this is used only for DeepEC and DSPACE).
        kwargs

        Returns
        -------

        """

        (model, train_dataset, validation_dataset,
         test_dataset, model_path, kwargs) = self.instantiate_model(model,
                                                                    num_tokens,
                                                                    input_size,
                                                                    num_labels,
                                                                    **kwargs)
        validation_loss_function = nn.BCELoss()
        kwargs["validation_loss_function"] = validation_loss_function
        kwargs["validation_metric"] = f1_score_macro

        model.fit(train_dataset=train_dataset, validation_dataset=validation_dataset)

        model.model.last_sigmoid = True
        model.save(model_path)

        self.test_model(train_dataset, validation_dataset, test_dataset, model)

        model.history["loss"].to_csv(os.path.join(self.losses_directory, f"{kwargs['model_name']}_loss.csv"),
                                     index=False)
        model.history["metric_results"].to_csv(os.path.join(self.metrics_directory,
                                                            f"{kwargs['model_name']}_metric_result.csv"), index=False)

    def test_model(self, train_dataset: SingleInputDataset, validation_dataset: SingleInputDataset,
                   test_dataset: SingleInputDataset, model: PyTorchModel):
        """
        Test the model.

        Parameters
        ----------
        train_dataset: SingleInputDataset
            Train dataset.
        validation_dataset: SingleInputDataset
            Validation dataset.
        test_dataset: SingleInputDataset
            Test dataset.
        model: PyTorchModel
            Model to be used.

        Returns
        -------

        """
        # Evaluate the model
        logger.info("Evaluating the model")

        train_metrics = self.generate_metrics(train_dataset, model)
        validation_metrics = self.generate_metrics(validation_dataset, model)

        logger.info(f"Train metrics: {train_metrics}")
        logger.info(f"Validation metrics: {validation_metrics}")
        if test_dataset is not None:
            metrics_test = self.generate_metrics(test_dataset, model)
            logger.info(f"Test metrics: {metrics_test}")

        os.makedirs(os.path.join(self.features_base_directory, "results", "metrics"), exist_ok=True)

        # create dataframe with the metrics for all datasets and models

        if os.path.exists(os.path.join(self.project_base_dir, "results", "metrics", "metrics.csv")):
            metrics = pd.read_csv(os.path.join(self.project_base_dir, "results", "metrics", "metrics.csv"))
        else:
            metrics = pd.DataFrame(columns=["model", "metric", "train", "validation", "test"])

        if test_dataset is None:
            dataset_metrics = [train_metrics, validation_metrics]
            dataset_names = ["train", "validation"]
        else:
            dataset_metrics = [train_metrics, validation_metrics, metrics_test]
            dataset_names = ["train", "validation", "test"]
        for metric in train_metrics.keys():
            row = len(metrics)
            metrics.at[row, "model"] = model.model_name
            metrics.at[row, "metric"] = metric
            for i, dataset_metric in enumerate(dataset_metrics):
                metrics.at[row, dataset_names[i]] = dataset_metric[metric]

        os.makedirs(os.path.join(self.project_base_dir, "results", "metrics"), exist_ok=True)
        metrics.to_csv(os.path.join(self.project_base_dir, "results", "metrics", "metrics.csv"), index=False)


##################################### Test DNN #####################################

def test_dnn():
    dnn_list = ["esm2_t6_8M_UR50D", "prot_bert_vectors"]
    for dnn in dnn_list:
        if dnn == "esm2_t33_650M_UR50D":
            fc_model = DNN(1280, [], 2771, batch_norm=True, last_sigmoid=True)
        elif dnn == "esm2_t30_150M_UR50D":
            fc_model = DNN(640, [], 2771, batch_norm=True, last_sigmoid=True)
        elif dnn == "esm2_t12_35M_UR50D":
            fc_model = DNN(480, [], 2771, batch_norm=True, last_sigmoid=True)
        elif dnn == "esm2_t6_8M_UR50D":
            fc_model = DNN(320, [], 2771, batch_norm=True, last_sigmoid=True)
        elif dnn == "esm2_t36_3B_UR50D":
            fc_model = DNN(2560, [], 2771, batch_norm=True, last_sigmoid=True)
        elif dnn == "prot_bert_vectors":
            fc_model = DNN(1024, [], 2771, batch_norm=True, last_sigmoid=True)

        loss_function = nn.BCELoss()
        fc_model.load_state_dict(
            torch.load(f"/scratch/jribeiro/results/{dnn}/DNN_{dnn}/model/pytorch_model_weights.pt"))
        model = PyTorchModel(fc_model, loss_function=loss_function,
                             optimizer=torch.optim.Adam(params=fc_model.parameters(), lr=0.0001),
                             batch_size=64, epochs=60,
                             device="cuda:0", logger_path="./logs.log",
                             progress=200,
                             patience=3,
                             early_stopping_method="metric",
                             objective="max")

        pipeline = PipelineRunner(dnn, last_sigmoid=True)
        train_dataset, validation_dataset, test_dataset = pipeline.prepare_dataset()
        print("datasets loaded")
        pipeline.test_model(train_dataset, validation_dataset, test_dataset, model)


def train_dnn_optimization(set_, model, epochs=30, dropout=None, batch_size=64, lr=0.0001, weight_decay=0.0):
    if model == "esm2_t33_650M_UR50D" or model == "esm1b_t33_650M_UR50S":
        input_dim = 1280
    elif model == "esm2_t30_150M_UR50D":
        input_dim = 640
    elif model == "esm2_t12_35M_UR50D":
        input_dim = 480
    elif model == "esm2_t6_8M_UR50D":
        input_dim = 320
    elif model == "esm2_t36_3B_UR50D":
        input_dim = 2560
    elif model == "prot_bert_vectors":
        input_dim = 1024
    elif model == "esm2_t48_15B_UR50D":
        input_dim = 5120

    if set_ == "1":
        fc_model = DNN(input_dim, [640], 2771, batch_norm=True, last_sigmoid=True, dropout=dropout)

    elif set_ == "2":

        fc_model = DNN(input_dim, [2560], 2771, batch_norm=True, last_sigmoid=True, dropout=dropout)


    elif set_ == "3":

        fc_model = DNN(input_dim, [640, 320], 2771, batch_norm=True, last_sigmoid=True, dropout=dropout)


    elif set_ == "4":

        fc_model = DNN(input_dim, [2560, 5120], 2771, batch_norm=True, last_sigmoid=True, dropout=dropout)

    elif set_ == "5":

        fc_model = DNN(input_dim, [2560, 5120, 10240], 2771, batch_norm=True, last_sigmoid=True, dropout=dropout)

    elif set_ == "6":

        fc_model = DNN(input_dim, [2560, 5120, 10240, 20480], 2771, batch_norm=True, last_sigmoid=True, dropout=dropout)

    pipeline = PipelineRunner(model, last_sigmoid=True)
    pipeline.train_model(fc_model, num_columns=None, input_size=None, num_labels=None,
                         optimizer=torch.optim.Adam(params=fc_model.parameters(), lr=lr, weight_decay=weight_decay),
                         batch_size=batch_size, epochs=epochs,
                         device="cuda:0", logger_path="./logs.log",
                         progress=200,
                         patience=3,
                         early_stopping_method="metric",
                         objective="max",
                         model_name=f"DNN_{model}_optimization_set_{set_}")


def train_dnn_optimization_all_data(set_, model, epochs=30, dropout=None, batch_size=64, lr=0.0001, weight_decay=0.0):
    if model == "esm2_t33_650M_UR50D" or model == "esm1b_t33_650M_UR50S":
        input_dim = 1280
    elif model == "esm2_t30_150M_UR50D":
        input_dim = 640
    elif model == "esm2_t12_35M_UR50D":
        input_dim = 480
    elif model == "esm2_t6_8M_UR50D":
        input_dim = 320
    elif model == "esm2_t36_3B_UR50D":
        input_dim = 2560
    elif model == "prot_bert_vectors":
        input_dim = 1024

    if set_ == "1":
        fc_model = DNN(input_dim, [640], 5743, batch_norm=True, last_sigmoid=True, dropout=dropout)

    elif set_ == "2":

        fc_model = DNN(input_dim, [2560], 5743, batch_norm=True, last_sigmoid=True, dropout=dropout)


    elif set_ == "3":

        fc_model = DNN(input_dim, [640, 320], 5743, batch_norm=True, last_sigmoid=True, dropout=dropout)


    elif set_ == "4":

        fc_model = DNN(input_dim, [2560, 5120], 5743, batch_norm=True, last_sigmoid=True, dropout=dropout)

    pipeline = PipelineRunner(model, last_sigmoid=True)
    pipeline.train_model_with_all_data(fc_model, num_columns=None, input_size=None, num_labels=None,
                                       optimizer=torch.optim.Adam(params=fc_model.parameters(), lr=lr,
                                                                  weight_decay=weight_decay),
                                       batch_size=batch_size, epochs=epochs,
                                       device="cuda:0", logger_path="./logs.log",
                                       progress=200,
                                       patience=3,
                                       early_stopping_method="metric",
                                       objective="max",
                                       model_name=f"DNN_{model}_optimization_set_{set_}_all_data")


def train_dnn_trials_merged(set_, model, epochs=30, dropout=None, batch_size=64, lr=0.0001, weight_decay=0.0):
    if model == "esm2_t33_650M_UR50D" or model == "esm1b_t33_650M_UR50S":
        input_dim = 1280
    elif model == "esm2_t30_150M_UR50D":
        input_dim = 640
    elif model == "esm2_t12_35M_UR50D":
        input_dim = 480
    elif model == "esm2_t6_8M_UR50D":
        input_dim = 320
    elif model == "esm2_t36_3B_UR50D":
        input_dim = 2560
    elif model == "prot_bert_vectors":
        input_dim = 1024

    if set_ == "1":
        fc_model = DNN(input_dim, [640], 2771, batch_norm=True, last_sigmoid=True, dropout=dropout)

    elif set_ == "2":

        fc_model = DNN(input_dim, [2560], 2771, batch_norm=True, last_sigmoid=True, dropout=dropout)


    elif set_ == "3":

        fc_model = DNN(input_dim, [640, 320], 2771, batch_norm=True, last_sigmoid=True, dropout=dropout)


    elif set_ == "4":

        fc_model = DNN(input_dim, [2560, 5120], 2771, batch_norm=True, last_sigmoid=True, dropout=dropout)

    elif set_ == "5":

        fc_model = DNN(input_dim, [2560, 5120, 10240], 2771, batch_norm=True, last_sigmoid=True, dropout=dropout)

    elif set_ == "6":

        fc_model = DNN(input_dim, [2560, 5120, 10240, 20480], 2771, batch_norm=True, last_sigmoid=True, dropout=dropout)

    pipeline = PipelineRunner(model, last_sigmoid=True)
    pipeline.train_model_with_validation_train_merged(fc_model, num_columns=None, input_size=None, num_labels=None,
                                                      optimizer=torch.optim.Adam(params=fc_model.parameters(), lr=lr,
                                                                                 weight_decay=weight_decay),
                                                      batch_size=batch_size, epochs=epochs,
                                                      device="cuda:0", logger_path="./logs.log",
                                                      progress=200,
                                                      model_name=f"DNN_{model}_trial_{set_}")


def train_dnn_esm():
    fc_model = DNN(1280, [2560, 2560 * 2, 2560 * 2 * 2, 2560 * 2 * 2 * 2], 2771, batch_norm=True, last_sigmoid=True)
    pipeline = PipelineRunner("esm2_t33_650M_UR50D_encoding", last_sigmoid=True, batch_size=20000)
    pipeline.train_model(fc_model, num_columns=1280, input_size=884, num_labels=2771,
                         optimizer=torch.optim.Adam(params=fc_model.parameters(), lr=0.0001),
                         batch_size=64, epochs=60,
                         device="cuda:0", logger_path="./logs.log",
                         progress=200,
                         patience=3,
                         early_stopping_method="metric",
                         objective="max")


def evaluate_model(epoch):
    model = DNN(1280, [2560, 2560 * 2, 2560 * 2 * 2, 2560 * 2 * 2 * 2], 2771, batch_norm=True, last_sigmoid=True)
    pipeline = PipelineRunner("esm2_t33_650M_UR50D", last_sigmoid=True)
    pipeline.instantiate_paths()
    train_dataset, validation_dataset, _ = pipeline.prepare_dataset()

    pipeline_checkpoints_path = os.path.join(pipeline.project_base_dir, "model_checkpoints")
    if isinstance(model, nn.Module):
        name = f"{model.__class__.__name__}_{pipeline.features_folder}"

    else:
        name = f"{model.__name__}_{pipeline.features_folder}"

    model_path = os.path.join(pipeline_checkpoints_path, name)
    model.load_state_dict(torch.load(os.path.join(model_path, f"epoch_{epoch}", f"model.pt")))
    model = PyTorchModel(model=model, loss_function=nn.BCELoss(), device="cuda:0")

    predictions = model.predict(train_dataset)
    precision_score_macro = precision_score(train_dataset.y, predictions, average="macro")

    print(f"Precision score macro train: {precision_score_macro}")

    predictions = model.predict(validation_dataset)
    precision_score_macro = precision_score(validation_dataset.y, predictions, average="macro")

    print(f"Precision score macro validation: {precision_score_macro}")


########################################## Train HDMLF ##########################################


if __name__ == "__main__":
    # arg1 = sys.argv[1]
    # train_dnn_baselines(arg1)

    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    train_dnn_optimization(arg1, arg2)

    # train_dnn_trials_merged(arg1, arg2)

    # train_dnn_optimization_all_data(arg1, arg2)

    # train_deep_ec_merged()
    # train_dspace_merged()

    # train_dnn_baselines_merged(arg1)

    # arg3 = sys.argv[3]
    # arg4 = sys.argv[4]
    # arg5 = sys.argv[5]
    # arg6 = sys.argv[6]
    # arg7 = sys.argv[7]

    # if float(arg4) == 0.0:
    #     dropout = None
    # else:
    #     dropout = float(arg4)

    # train_clean()

    # test_dnn()

    # train_deep_ec()
    # test_deep_ec_esm()
    # run_test()
    # train_dspace_esm()
    # train_dspace()
    # run_test_deep_ec()
