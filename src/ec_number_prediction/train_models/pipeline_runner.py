from logging import Logger
import os
from typing import Union

import numpy as np
from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset
from plants_sm.models.pytorch_model import PyTorchModel
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, coverage_error, average_precision_score

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

        if os.path.exists(os.path.join(self.project_base_dir, "results", "metrics", "metric_values.csv")):
            metrics = pd.read_csv(os.path.join(self.project_base_dir, "results", "metrics", "metric_values.csv"))
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
        metrics.to_csv(os.path.join(self.project_base_dir, "results", "metrics", "metric_values.csv"), index=False)

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