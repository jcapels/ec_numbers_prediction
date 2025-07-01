from abc import abstractmethod
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
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

class FineTuneExperiment(Experiment):
    def __init__(self, datasets, path_to_model, baseline = False, results_output_file="results.csv", base_layers=[2560], input_dim=1024, classification_neurons=643,
                folder_path="trials", metric = f1_macro, **kwargs):
        super().__init__(**kwargs)

        self.folder_path = folder_path
        self.datasets = datasets
        self.baseline = baseline
        self.results_output_file = results_output_file
        self.path_to_model = path_to_model
        self.base_layers = base_layers
        self.input_dim = input_dim
        self.classification_neurons = classification_neurons
        self.metric = metric

    def _steps(self, trial):

        additional_layers = trial.suggest_categorical("additional_layers", ["[2560]","[1280]", "[640]", "[2560, 1280]", "[2560, 640]", "[1280, 640]",
                                                                            "[2560, 1280]", "[2560, 1280, 640]",
                                                                            "[2560, 1280, 1280]",
                                                                            "[2560, 1280, 1280, 640]",
                                                                            "[2560, 1280, 1280, 1280]",
                                                                            "[2560, 1280, 1280, 1280, 640]",
                                                                            "[2560, 1280, 1280, 1280, 1280]",
                                                                            "[2560, 2560, 2560, 1280]",
                                                                            "[2560, 2560, 2560, 1280, 640]"],)
        
        # evaluate literal
        additional_layers = eval(additional_layers)

        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)

        return additional_layers, batch_size, learning_rate

    @abstractmethod
    def objective(self, trial: optuna.Trial) -> float:
        pass

class FineTuneExperimentTFvsNoTFWithoutValidation(FineTuneExperiment):

    def __init__(self, datasets, path_to_model, baseline=False, results_output_file="results.csv", base_layers=[2560], input_dim=1024, classification_neurons=643, folder_path="trials", **kwargs):
        super().__init__(datasets, path_to_model, baseline, results_output_file, base_layers, input_dim, classification_neurons, folder_path, **kwargs)

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Method to be implemented by all experiments to define the objective function.

        Parameters
        ----------
        trial: optuna.trial.Trial
            A trial object that contains the current suggested hyperparameters.

        Returns
        -------
        float
            The value of the objective function.
        """

        additional_layers, batch_size, learning_rate = self._steps(trial)
        epochs = trial.suggest_int("epochs", 50, 200)
        results = []
        i=0
        if os.path.exists(self.results_output_file):
            results = pd.read_csv(self.results_output_file)
        else:
            results = pd.DataFrame(columns=["Model type", "Trial", "Fold", "F1_macro"])
        for train_dataset, test_dataset in self.datasets:
            f1_macro = self.fine_tune_with_no_layers(train_dataset, test_dataset, additional_layers=additional_layers, batch_size=batch_size, learning_rate=learning_rate,
                                                     path_to_model=self.path_to_model, input_dim=self.input_dim, classification_neurons=self.classification_neurons,
                                                     base_layers=self.base_layers, epochs=epochs)

            results = pd.concat([results, pd.DataFrame({"Model type": ["TF"], "Trial": [f"{trial.number}"], "Fold": [i], "F1_macro": [f1_macro]})], ignore_index=True)

            f1_macro = self.train_baseline(train_dataset, test_dataset, batch_size=batch_size, learning_rate=learning_rate,
                                            layers=additional_layers, input_dim=self.input_dim, classification_neurons=self.classification_neurons,
                                            base_layers=self.base_layers, epochs=epochs)

            results = pd.concat([results, pd.DataFrame({"Model type": ["No TF"], "Trial": [f"{trial.number}"], "Fold": [i], "F1_macro": [f1_macro]})], ignore_index=True)
            i+=1
            results.to_csv(self.results_output_file, index=False)
        
        return 1


    @staticmethod
    def train_baseline(train_dataset: SingleInputDataset, test_dataset: SingleInputDataset,
                       batch_size: int = 32, learning_rate: float = 1e-4, 
                        layers: list = [2560], input_dim: int = 1024, classification_neurons: int = 643, base_layers= [2560],
                        epochs: int = 50):
            
        module = ModelECNumber(input_dim=input_dim, layers=base_layers + layers, classification_neurons=classification_neurons, 
            metric=f1_macro, learning_rate=learning_rate, scheduler=False)
        
        # callbacks = EarlyStopping("val_metric", patience=5, mode="max")
        
        model = InternalLightningModel(module=module, max_epochs=epochs,
                batch_size=batch_size,
                devices=[1],
                accelerator="gpu",
                # strategy="fsdp",
                )
        
        model.fit(train_dataset)
        predictions = model.predict(test_dataset)

        return f1_macro(test_dataset.y, predictions)

    @staticmethod
    def fine_tune_with_no_layers(train_dataset: SingleInputDataset, test_dataset: SingleInputDataset,
                                 base_layers: list = [2560],
                                 additional_layers: list = [2560, 1280], batch_size: int = 32, learning_rate: float = 1e-4,
                                 path_to_model: str = "/home/jcapela/ec_numbers_prediction/plants_pipeline/pretrained_models/protbert.pt",
                                 input_dim: int = 1024, classification_neurons: int = 643,
                                 epochs: int = 50):

        module = FineTuneModelECNumber(input_dim=input_dim, additional_layers=additional_layers, classification_neurons=classification_neurons, \
            path_to_model=path_to_model,
            metric=f1_macro, learning_rate=learning_rate, base_layers=base_layers, layers_to_freeze=len(base_layers), scheduler=False)
        
        # callbacks = EarlyStopping("val_metric", patience=5, mode="max")
        
        model = InternalLightningModel(module=module, max_epochs=epochs,
                batch_size=batch_size,
                devices=[1],
                accelerator="gpu",
                # strategy="fsdp",
                )
        
        model.fit(train_dataset)
        predictions = model.predict(test_dataset)
        return f1_macro(test_dataset.y, predictions)

class FineTuneExperimentTFvsNoTF_PlantsSMOrComplete(FineTuneExperiment):

    def __init__(self, datasets, path_to_model, baseline=False, results_output_file="results.csv", base_layers=[2560], input_dim=1024, classification_neurons=643, folder_path="trials", **kwargs):
        super().__init__(datasets, path_to_model, baseline, results_output_file, base_layers, input_dim, classification_neurons, folder_path, **kwargs)

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Method to be implemented by all experiments to define the objective function.

        Parameters
        ----------
        trial: optuna.trial.Trial
            A trial object that contains the current suggested hyperparameters.

        Returns
        -------
        float
            The value of the objective function.
        """

        additional_layers, batch_size, learning_rate = self._steps(trial)
        epochs = trial.suggest_int("epochs", 50, 200)
        results = []
        i=0
        if os.path.exists(self.results_output_file):
            results = pd.read_csv(self.results_output_file)
        else:
            results = pd.DataFrame(columns=["Model type", "Trial", "Fold", "F1_macro"])
        for train_dataset, test_dataset, train_dataset_full in self.datasets:
            f1_macro = self.fine_tune_with_no_layers(train_dataset, test_dataset, additional_layers=additional_layers, batch_size=batch_size, learning_rate=learning_rate,
                                                     path_to_model=self.path_to_model, input_dim=self.input_dim, classification_neurons=self.classification_neurons,
                                                     base_layers=self.base_layers, epochs=epochs)

            results = pd.concat([results, pd.DataFrame({"Model type": ["TF"], "Trial": [f"{trial.number}"], "Fold": [i], "F1_macro": [f1_macro]})], ignore_index=True)

            f1_macro = self.train_baseline(train_dataset, test_dataset, batch_size=batch_size, learning_rate=learning_rate,
                                            layers=additional_layers, input_dim=self.input_dim, classification_neurons=self.classification_neurons,
                                            base_layers=self.base_layers, epochs=epochs)

            results = pd.concat([results, pd.DataFrame({"Model type": ["No TF"], "Trial": [f"{trial.number}"], "Fold": [i], "F1_macro": [f1_macro]})], ignore_index=True)
            
            f1_macro = self.fine_tune_with_no_layers(train_dataset_full, test_dataset, additional_layers=additional_layers, batch_size=batch_size, learning_rate=learning_rate,
                                                     path_to_model=self.path_to_model, input_dim=self.input_dim, classification_neurons=self.classification_neurons,
                                                     base_layers=self.base_layers, epochs=epochs)

            results = pd.concat([results, pd.DataFrame({"Model type": ["TF Full Plants"], "Trial": [f"{trial.number}"], "Fold": [i], "F1_macro": [f1_macro]})], ignore_index=True)

            f1_macro = self.train_baseline(train_dataset_full, test_dataset, batch_size=batch_size, learning_rate=learning_rate,
                                            layers=additional_layers, input_dim=self.input_dim, classification_neurons=self.classification_neurons,
                                            base_layers=self.base_layers, epochs=epochs)
            
            results = pd.concat([results, pd.DataFrame({"Model type": ["No TF Full Plants"], "Trial": [f"{trial.number}"], "Fold": [i], "F1_macro": [f1_macro]})], ignore_index=True)
            
            i+=1
            results.to_csv(self.results_output_file, index=False)
        
        return 1


    @staticmethod
    def train_baseline(train_dataset: SingleInputDataset, test_dataset: SingleInputDataset,
                       batch_size: int = 32, learning_rate: float = 1e-4, 
                        layers: list = [2560], input_dim: int = 1024, classification_neurons: int = 643, base_layers= [2560],
                        epochs: int = 50):
            
        module = ModelECNumber(input_dim=input_dim, layers=base_layers + layers, classification_neurons=classification_neurons, 
            metric=f1_macro, learning_rate=learning_rate, scheduler=False)
        
        # callbacks = EarlyStopping("val_metric", patience=5, mode="max")
        
        model = InternalLightningModel(module=module, max_epochs=epochs,
                batch_size=batch_size,
                devices=[1],
                accelerator="gpu",
                # strategy="fsdp",
                )
        
        model.fit(train_dataset)
        predictions = model.predict(test_dataset)

        return f1_macro(test_dataset.y, predictions)

    @staticmethod
    def fine_tune_with_no_layers(train_dataset: SingleInputDataset, test_dataset: SingleInputDataset,
                                 base_layers: list = [2560],
                                 additional_layers: list = [2560, 1280], batch_size: int = 32, learning_rate: float = 1e-4,
                                 path_to_model: str = "/home/jcapela/ec_numbers_prediction/plants_pipeline/pretrained_models/protbert.pt",
                                 input_dim: int = 1024, classification_neurons: int = 643,
                                 epochs: int = 50):

        module = FineTuneModelECNumber(input_dim=input_dim, additional_layers=additional_layers, classification_neurons=classification_neurons, \
            path_to_model=path_to_model,
            metric=f1_macro, learning_rate=learning_rate, base_layers=base_layers, layers_to_freeze=len(base_layers), scheduler=False)
        
        # callbacks = EarlyStopping("val_metric", patience=5, mode="max")
        
        model = InternalLightningModel(module=module, max_epochs=epochs,
                batch_size=batch_size,
                devices=[1],
                accelerator="gpu",
                # strategy="fsdp",
                )
        
        model.fit(train_dataset)
        predictions = model.predict(test_dataset)
        return f1_macro(test_dataset.y, predictions)


class FineTuneExperimentTFvsNoTF(FineTuneExperiment):

    @staticmethod
    def train_baseline(train_dataset: SingleInputDataset, test_dataset: SingleInputDataset, validation_set: SingleInputDataset, 
                       batch_size: int = 32, learning_rate: float = 1e-4, 
                        layers: list = [2560], input_dim: int = 1024, classification_neurons: int = 643, base_layers= [2560],
                        metric=f1_macro):
            
        module = ModelECNumber(input_dim=input_dim, layers=base_layers + layers, classification_neurons=classification_neurons, 
            metric=metric, learning_rate=learning_rate)
        
        callbacks = EarlyStopping("val_metric", patience=5, mode="max")
        
        model = InternalLightningModel(module=module, max_epochs=200,
                batch_size=batch_size,
                devices=[1],
                accelerator="gpu",
                # strategy="fsdp",
                callbacks=[callbacks])
        
        model.fit(train_dataset, validation_set)
        predictions = model.predict(test_dataset)

        return metric(test_dataset.y, predictions)

    @staticmethod
    def fine_tune_with_no_layers(train_dataset: SingleInputDataset, test_dataset: SingleInputDataset, validation_set: SingleInputDataset,
                                 base_layers: list = [2560],
                                 additional_layers: list = [2560, 1280], batch_size: int = 32, learning_rate: float = 1e-4,
                                 path_to_model: str = "/home/jcapela/ec_numbers_prediction/plants_pipeline/pretrained_models/protbert.pt",
                                 input_dim: int = 1024, classification_neurons: int = 643,
                                 metric=f1_macro):

        module = FineTuneModelECNumber(input_dim=input_dim, additional_layers=additional_layers, classification_neurons=classification_neurons, \
            path_to_model=path_to_model,
            metric=metric, learning_rate=learning_rate, base_layers=base_layers, layers_to_freeze=len(base_layers))
        
        callbacks = EarlyStopping("val_metric", patience=5, mode="max")
        
        model = InternalLightningModel(module=module, max_epochs=200,
                batch_size=batch_size,
                devices=[0],
                accelerator="gpu",
                # strategy="fsdp",
                callbacks=[callbacks])
        
        model.fit(train_dataset, validation_set)
        predictions = model.predict(test_dataset)

        return metric(test_dataset.y, predictions)


    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Method to be implemented by all experiments to define the objective function.

        Parameters
        ----------
        trial: optuna.trial.Trial
            A trial object that contains the current suggested hyperparameters.

        Returns
        -------
        float
            The value of the objective function.
        """

        additional_layers, batch_size, learning_rate = self._steps(trial)
        results = []
        i=0
        if os.path.exists(self.results_output_file):
            results = pd.read_csv(self.results_output_file)
        else:
            results = pd.DataFrame(columns=["Model type", "Trial", "Fold", self.metric.__name__])
        for train_dataset, test_dataset, validation_set in self.datasets:
            metric_value = self.fine_tune_with_no_layers(train_dataset, test_dataset, validation_set, additional_layers=additional_layers, batch_size=batch_size, learning_rate=learning_rate,
                                                     path_to_model=self.path_to_model, input_dim=self.input_dim, classification_neurons=self.classification_neurons,
                                                     base_layers=self.base_layers, metric=self.metric)

            results = pd.concat([results, pd.DataFrame({"Model type": ["TF"], "Trial": [f"{trial.number}"], "Fold": [i], self.metric.__name__: [metric_value]})], ignore_index=True)

            metric_value = self.train_baseline(train_dataset, test_dataset, validation_set, batch_size=batch_size, learning_rate=learning_rate,
                                            layers=additional_layers, input_dim=self.input_dim, classification_neurons=self.classification_neurons,
                                            base_layers=self.base_layers, metric=self.metric)

            results = pd.concat([results, pd.DataFrame({"Model type": ["No TF"], "Trial": [f"{trial.number}"], "Fold": [i], self.metric.__name__: [metric_value]})], ignore_index=True)
            i+=1
            results.to_csv(self.results_output_file, index=False)
        
        return 1

class FineTuneWithOptimization(FineTuneExperiment):

    @staticmethod
    def fine_tune_with_no_layers(train_dataset: SingleInputDataset, test_dataset: SingleInputDataset, validation_set: SingleInputDataset,
                                 base_layers: list = [2560],
                                 additional_layers: list = [2560, 1280], batch_size: int = 32, learning_rate: float = 1e-4,
                                 path_to_model: str = "/home/jcapela/ec_numbers_prediction/plants_pipeline/pretrained_models/protbert.pt",
                                 input_dim: int = 1024, classification_neurons: int = 643):
        module = FineTuneModelECNumber(input_dim=input_dim, additional_layers=additional_layers, classification_neurons=classification_neurons, \
            path_to_model=path_to_model,
            metric=f1_macro, learning_rate=learning_rate, base_layers=base_layers, layers_to_freeze=len(base_layers))
        
        callbacks = EarlyStopping("val_metric", patience=5, mode="max")
        
        model = InternalLightningModel(module=module, max_epochs=200,
                batch_size=batch_size,
                devices=[1],
                accelerator="gpu",
                # strategy="fsdp",
                callbacks=[callbacks])
        
        model.fit(train_dataset, validation_set)
        predictions = model.predict(validation_set)
        from sklearn.metrics import f1_score, precision_score, recall_score
        mf1_val = f1_score(validation_set.y, predictions, average='macro')

        predictions = model.predict(test_dataset)

        mf1 = f1_score(test_dataset.y, predictions, average='macro')
        wf1 = f1_score(test_dataset.y, predictions, average='weighted')
        mrecall = recall_score(test_dataset.y, predictions, average='macro')
        wrecall = recall_score(test_dataset.y, predictions, average='weighted')
        mprecision = precision_score(test_dataset.y, predictions, average='macro')
        wprecision = precision_score(test_dataset.y, predictions, average='weighted')

        return mf1_val, mf1, wf1, mrecall, wrecall, mprecision, wprecision, model
    
    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Method to be implemented by all experiments to define the objective function.

        Parameters
        ----------
        trial: optuna.trial.Trial
            A trial object that contains the current suggested hyperparameters.

        Returns
        -------
        float
            The value of the objective function.
        """

        additional_layers, batch_size, learning_rate = self._steps(trial)
        results = []
        i=0
        if os.path.exists(self.results_output_file):
            results = pd.read_csv(self.results_output_file)
        else:
            results = pd.DataFrame(columns=["Model type", "Trial", "Fold", "F1_macro"])
        
        with open("Test", "a") as f:
            f.write(f"Results file: {self.results_output_file}\n")
            f.write(f"Additional layers: {additional_layers}\n")

        val_mf1_scores = []

        for train_dataset, test_dataset, validation_set in self.datasets:
            
            mf1_val, mf1, wf1, mrecall, wrecall, mprecision, wprecision, model = self.fine_tune_with_no_layers(train_dataset, test_dataset, validation_set, additional_layers=additional_layers, 
                                                                                                        batch_size=batch_size, learning_rate=learning_rate,
                                                     path_to_model=self.path_to_model, input_dim=self.input_dim, classification_neurons=self.classification_neurons,
                                                     base_layers=self.base_layers)
            val_mf1_scores.append(mf1_val)

            results = pd.concat([results, pd.DataFrame({"Trial": [f"{trial.number}"], "Fold": [i], "mf1": [mf1], "wf1": 
                                                        [wf1], "mrecall": mrecall, "wrecall": wrecall, "mprecision": mprecision,
                                                        "wprecision": wprecision})], ignore_index=True)

            i+=1
            results.to_csv(self.results_output_file, index=False)
        val_mf1_scores = np.array(val_mf1_scores)

        score = val_mf1_scores.mean() - val_mf1_scores.std()

        model.save(f"{self.folder_path}/{trial.number}/")

        return score

class FineTuneWithOptimizationForPlants(FineTuneExperiment):

    @staticmethod
    def fine_tune_with_no_layers(train_dataset: SingleInputDataset, test_dataset: SingleInputDataset,
                                 base_layers: list = [2560],
                                 additional_layers: list = [2560, 1280], batch_size: int = 32, learning_rate: float = 1e-4,
                                 path_to_model: str = "/home/jcapela/ec_numbers_prediction/plants_pipeline/pretrained_models/protbert.pt",
                                 input_dim: int = 1024, classification_neurons: int = 643, epochs=50):
        
        module = FineTuneModelECNumber(input_dim=input_dim, additional_layers=additional_layers, classification_neurons=classification_neurons, \
            path_to_model=path_to_model,
            metric=f1_macro, learning_rate=learning_rate, base_layers=base_layers, layers_to_freeze=len(base_layers), scheduler=False)
        
        
        model = InternalLightningModel(module=module, max_epochs=epochs,
                batch_size=batch_size,
                devices=[1],
                accelerator="gpu")
        
        model.fit(train_dataset)
        predictions = model.predict(test_dataset)
        from sklearn.metrics import f1_score, precision_score, recall_score
        mf1_val = f1_score(test_dataset.y, predictions, average='macro')

        predictions = model.predict(test_dataset)

        mf1 = f1_score(test_dataset.y, predictions, average='macro')
        wf1 = f1_score(test_dataset.y, predictions, average='weighted')
        mrecall = recall_score(test_dataset.y, predictions, average='macro')
        wrecall = recall_score(test_dataset.y, predictions, average='weighted')
        mprecision = precision_score(test_dataset.y, predictions, average='macro')
        wprecision = precision_score(test_dataset.y, predictions, average='weighted')

        return mf1_val, mf1, wf1, mrecall, wrecall, mprecision, wprecision, model
    
    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Method to be implemented by all experiments to define the objective function.

        Parameters
        ----------
        trial: optuna.trial.Trial
            A trial object that contains the current suggested hyperparameters.

        Returns
        -------
        float
            The value of the objective function.
        """

        additional_layers, batch_size, learning_rate = self._steps(trial)
        epochs = trial.suggest_int("epochs", 50, 200)
        results = []
        i=0
        if os.path.exists(self.results_output_file):
            results = pd.read_csv(self.results_output_file)
        else:
            results = pd.DataFrame(columns=["Model type", "Trial", "Fold", "F1_macro"])
        
        with open("Test", "a") as f:
            f.write(f"Results file: {self.results_output_file}\n")
            f.write(f"Additional layers: {additional_layers}\n")

        val_mf1_scores = []

        for train_dataset, test_dataset in self.datasets:
            
            mf1_val, mf1, wf1, mrecall, wrecall, mprecision, wprecision, model = self.fine_tune_with_no_layers(train_dataset, test_dataset, additional_layers=additional_layers, 
                                                                                                        batch_size=batch_size, learning_rate=learning_rate,
                                                     path_to_model=self.path_to_model, input_dim=self.input_dim, classification_neurons=self.classification_neurons,
                                                     base_layers=self.base_layers, epochs=epochs)
            val_mf1_scores.append(mf1_val)

            results = pd.concat([results, pd.DataFrame({"Trial": [f"{trial.number}"], "Fold": [i], "mf1": [mf1], "wf1": 
                                                        [wf1], "mrecall": mrecall, "wrecall": wrecall, "mprecision": mprecision,
                                                        "wprecision": wprecision})], ignore_index=True)

            i+=1
            results.to_csv(self.results_output_file, index=False)
        val_mf1_scores = np.array(val_mf1_scores)

        score = val_mf1_scores.mean() - val_mf1_scores.std()

        model.save(f"{self.folder_path}/{trial.number}/")

        return score