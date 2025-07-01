

import re
from ec_number_prediction.transfer_learning.fine_tuning_experiment import FineTuneExperimentTFvsNoTF, FineTuneWithOptimization, FineTuneExperimentTFvsNoTF_PlantsSMOrComplete, FineTuneWithOptimizationForPlants
import os
import numpy as np
import optuna
import pandas as pd
from plants_sm.hyperparameter_optimization.experiment import Experiment
from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset

from sklearn.metrics import f1_score
from ec_number_prediction.transfer_learning.models import FineTuneModelECNumber, ModelECNumber
from ec_number_prediction.transfer_learning.fine_tuning_experiment import FineTuneExperiment
from ec_number_prediction._utils import get_ec_levels
from plants_sm.models.lightning_model import InternalLightningModel
from lightning.pytorch.callbacks import EarlyStopping

base_path_data = f"/home/jcapela/plants_ec_number_prediction/ec_numbers_prediction/plants_pipeline/data/"
base_path = f"/home/jcapela/plants_ec_number_prediction/ec_numbers_prediction/plants_pipeline/"

def experiment_tf_no_tf():
    datasets = []
    # load datasets
    train_datasets_path = os.path.join(base_path_data, "plants/train_datasets/")
    test_datasets_path = os.path.join(base_path_data, "plants/test_datasets/")
    validation_datasets_path = os.path.join(base_path_data, "plants/validation_datasets/")
    for i in range(5):
        print(f"Loading dataset {i}")
        train_dataset = SingleInputDataset.from_csv(train_datasets_path + f"train_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))
        test_dataset = SingleInputDataset.from_csv(test_datasets_path + f"test_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))
        validation_set = SingleInputDataset.from_csv(validation_datasets_path + f"validation_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))

        train_dataset.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))
        test_dataset.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))
        validation_set.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))
        
        datasets.append((train_dataset, test_dataset, validation_set))
    
    experiment = FineTuneExperimentTFvsNoTF(datasets=datasets, study_name="esm2_3b_experiment", storage="sqlite:///transfer_learning_experiment.db", sampler= optuna.samplers.RandomSampler(),
                                    direction="maximize", load_if_exists=True, path_to_model=os.path.join(base_path, "pretrained_models/esm2_3b_no_plants.ckpt"),
                                    base_layers=[2560], input_dim=2560, classification_neurons=643, results_output_file="results_esm2_3b_no_plants.csv")
    
    experiment.run(n_trials=10, n_jobs=1)

def experiment_optimize():
    datasets = []
    # load datasets
    train_datasets_path = os.path.join(base_path_data, "plants/train_datasets/")
    test_datasets_path = os.path.join(base_path_data, "plants/test_datasets/")
    validation_datasets_path = os.path.join(base_path_data, "plants/validation_datasets/")
    for i in range(5):
        print(f"Loading dataset {i}")
        train_dataset = SingleInputDataset.from_csv(train_datasets_path + f"train_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))
        test_dataset = SingleInputDataset.from_csv(test_datasets_path + f"test_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))
        validation_set = SingleInputDataset.from_csv(validation_datasets_path + f"validation_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))

        train_dataset.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))
        test_dataset.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))
        validation_set.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))
        
        datasets.append((train_dataset, test_dataset, validation_set))
    
    experiment = FineTuneWithOptimization(datasets=datasets, study_name="esm2_3b_experiment_with_optimization", storage="sqlite:///transfer_learning_experiment.db", sampler= optuna.samplers.TPESampler(),
                                    direction="maximize", load_if_exists=True, path_to_model=os.path.join(base_path, "pretrained_models/esm2_3b.pt"),
                                    base_layers=[2560], input_dim=2560, classification_neurons=643, results_output_file="results_esm2_3b_optimization_results_no_plants.csv",
                                    folder_path="esm2_3b/trials")
    
    experiment.run(n_trials=50, n_jobs=1)

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def evaluate_model():

    from plants_sm.models.lightning_model import InternalLightningModel
    from ec_number_prediction.transfer_learning.models import ModelECNumber
    from plants_sm.io.pickle import read_pickle
    import os
    from plants_sm.models.constants import BINARY, FileConstants

    path = os.path.join(base_path, "train_models/esm2_3b/trials/24/")
    weights_path = os.path.join(path, "pytorch_model_weights.ckpt")
    model = read_pickle(os.path.join(path, FileConstants.PYTORCH_MODEL_PKL.value))
    model = model.load_from_checkpoint(weights_path,input_dim=2560, additional_layers=[640], classification_neurons=643,
                                       learning_rate=0.002686377622297537, 
                                       batch_size=16, path_to_model=os.path.join(base_path, "pretrained_models/esm2_3b_no_plants.ckpt"))
    model = InternalLightningModel(module=model)

    test_datasets_path = os.path.join(base_path_data, "plants/test_datasets/")
    results_all = pd.DataFrame()
    for i in range(5):
        print(f"Loading dataset {i}")
        test_dataset = SingleInputDataset.from_csv(test_datasets_path + f"test_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))

        test_dataset.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))

        from sklearn.metrics import f1_score, precision_score, recall_score

        predictions = model.predict(test_dataset)

        mf1 = f1_score(test_dataset.y, predictions, average='macro')
        wf1 = f1_score(test_dataset.y, predictions, average='weighted')
        mrecall = recall_score(test_dataset.y, predictions, average='macro')
        wrecall = recall_score(test_dataset.y, predictions, average='weighted')
        mprecision = precision_score(test_dataset.y, predictions, average='macro')
        wprecision = precision_score(test_dataset.y, predictions, average='weighted')
        level_1, level_2, level_3, level_4 = get_ec_levels(test_dataset._labels_names)
        level_1_f1 = f1_score(test_dataset.y[:, level_1], predictions[:, level_1], average='macro')
        level_2_f1 = f1_score(test_dataset.y[:, level_2], predictions[:, level_2], average='macro')
        level_3_f1 = f1_score(test_dataset.y[:, level_3], predictions[:, level_3], average='macro')
        level_4_f1 = f1_score(test_dataset.y[:, level_4], predictions[:, level_4], average='macro')

        level_1_wf1 = f1_score(test_dataset.y[:, level_1], predictions[:, level_1], average='weighted')
        level_2_wf1 = f1_score(test_dataset.y[:, level_2], predictions[:, level_2], average='weighted')
        level_3_wf1 = f1_score(test_dataset.y[:, level_3], predictions[:, level_3], average='weighted')
        level_4_wf1 = f1_score(test_dataset.y[:, level_4], predictions[:, level_4], average='weighted')

        results = pd.DataFrame({"fold": [i], "mf1": [mf1], "wf1": [wf1], "mrecall": [mrecall], "wrecall": [wrecall], "mprecision": [mprecision], "wprecision": [wprecision],
                                "level_1_f1": [level_1_f1], "level_2_f1": [level_2_f1], "level_3_f1": [level_3_f1], "level_4_f1": [level_4_f1],
                                "level_1_wf1": [level_1_wf1], "level_2_wf1": [level_2_wf1], "level_3_wf1": [level_3_wf1], "level_4_wf1": [level_4_wf1],
                                })

        results_all = pd.concat([results_all, results])

        results_all.to_csv(os.path.join(base_path, "train_models/results_esm2_3b_trial_24.csv"))


def train_model_and_evaluate():
    # load datasets

    from plants_sm.models.lightning_model import InternalLightningModel
    from ec_number_prediction.transfer_learning.models import ModelECNumber
    from plants_sm.io.pickle import read_pickle
    import os
    from plants_sm.models.constants import BINARY, FileConstants

    # path = os.path.join(base_path, "train_models/esm2_3b/trials/24/")
    # model = read_pickle(os.path.join(path, FileConstants.PYTORCH_MODEL_PKL.value))

    train_datasets_path = os.path.join(base_path_data, "plants/train_datasets/")
    test_datasets_path = os.path.join(base_path_data, "plants/test_datasets/")
    validation_datasets_path = os.path.join(base_path_data, "plants/validation_datasets/")

    results_all = pd.DataFrame()
    for i in range(5):
        print(f"Loading dataset {i}")
        train_dataset = SingleInputDataset.from_csv(train_datasets_path + f"train_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))
        test_dataset = SingleInputDataset.from_csv(test_datasets_path + f"test_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))
        validation_set = SingleInputDataset.from_csv(validation_datasets_path + f"validation_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))

        train_dataset.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))
        test_dataset.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))
        validation_set.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))

        callbacks = EarlyStopping("val_metric", patience=5, mode="max")

        module = FineTuneModelECNumber(input_dim=2560, additional_layers=[640], classification_neurons=643, \
            path_to_model=os.path.join(base_path, "pretrained_models/esm2_3b_no_plants.ckpt"),
            metric=f1_macro, learning_rate=0.002686377622297537, base_layers=[2560], layers_to_freeze=len([2560]),
            scheduler=False)
        
        model = InternalLightningModel(module=module, max_epochs=200,
                batch_size=16,
                devices=[2],
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
        level_1, level_2, level_3, level_4 = get_ec_levels(test_dataset._labels_names)
        level_1_f1 = f1_score(test_dataset.y[:, level_1], predictions[:, level_1], average='macro')
        level_2_f1 = f1_score(test_dataset.y[:, level_2], predictions[:, level_2], average='macro')
        level_3_f1 = f1_score(test_dataset.y[:, level_3], predictions[:, level_3], average='macro')
        level_4_f1 = f1_score(test_dataset.y[:, level_4], predictions[:, level_4], average='macro')

        level_1_wf1 = f1_score(test_dataset.y[:, level_1], predictions[:, level_1], average='weighted')
        level_2_wf1 = f1_score(test_dataset.y[:, level_2], predictions[:, level_2], average='weighted')
        level_3_wf1 = f1_score(test_dataset.y[:, level_3], predictions[:, level_3], average='weighted')
        level_4_wf1 = f1_score(test_dataset.y[:, level_4], predictions[:, level_4], average='weighted')

        results = pd.DataFrame({"fold": [i], "mf1": [mf1], "wf1": [wf1], "mrecall": [mrecall], "wrecall": [wrecall], "mprecision": [mprecision], "wprecision": [wprecision],
                                "level_1_f1": [level_1_f1], "level_2_f1": [level_2_f1], "level_3_f1": [level_3_f1], "level_4_f1": [level_4_f1],
                                "level_1_wf1": [level_1_wf1], "level_2_wf1": [level_2_wf1], "level_3_wf1": [level_3_wf1], "level_4_wf1": [level_4_wf1],
                                "mf1_val": [mf1_val]})

        results_all = pd.concat([results_all, results])

        results_all.to_csv(os.path.join(base_path, "train_models/results_esm2_3b_trial_24.csv"))


def train_model():
    # load datasets
    train_datasets_path = "/home/jcapela/ec_numbers_prediction/plants_pipeline/data/plants/train_datasets/"
    test_datasets_path = "/home/jcapela/ec_numbers_prediction/plants_pipeline/data/plants/test_datasets/"
    validation_datasets_path = "/home/jcapela/ec_numbers_prediction/plants_pipeline/data/plants/validation_datasets/"
    train_dataset = SingleInputDataset.from_csv(train_datasets_path + f"train_dataset_{0}.csv", representation_field="sequence", instances_ids_field="accession",
                                                labels_field=slice(11,-1))
    test_dataset = SingleInputDataset.from_csv(test_datasets_path + f"test_dataset_{0}.csv", representation_field="sequence", instances_ids_field="accession",
                                                labels_field=slice(11,-1))
    validation_set = SingleInputDataset.from_csv(validation_datasets_path + f"validation_dataset_{0}.csv", representation_field="sequence", instances_ids_field="accession",
                                                labels_field=slice(11,-1))

    train_dataset.load_features("/home/jcapela/ec_numbers_prediction/plants_pipeline/data/swiss_prot_ec_plants_esm/")
    test_dataset.load_features("/home/jcapela/ec_numbers_prediction/plants_pipeline/data/swiss_prot_ec_plants_esm/")
    validation_set.load_features("/home/jcapela/ec_numbers_prediction/plants_pipeline/data/swiss_prot_ec_plants_esm/")

    def f1_macro(y_true, y_pred):
        return f1_score(y_true, y_pred, average="macro")

    dataset = train_dataset.merge(test_dataset)
    dataset = dataset.merge(validation_set)
    module = FineTuneModelECNumber(input_dim=2560, additional_layers=[2560, 1280], classification_neurons=643, \
            path_to_model="/home/jcapela/ec_numbers_prediction/plants_pipeline/pretrained_models/esm2_3b.pt",
            metric=f1_macro, learning_rate=0.001272374979682636, base_layers=[2560], layers_to_freeze=len([2560]),
            scheduler=False)
        
    model = InternalLightningModel(module=module, max_epochs=100,
            batch_size=16,
            devices=[0],
            accelerator="gpu"
            )
    
    model.fit(train_dataset)
    model.save("/home/jcapela/ec_numbers_prediction/plants_pipeline/models/esm2_3b_ft_2560_1280_643")

def experiment_tf_no_tf_plants_sm():
    datasets = []
    # load datasets
    train_datasets_path = os.path.join(base_path_data, "plants_sm/train_datasets/")
    test_datasets_path = os.path.join(base_path_data, "plants_sm/test_datasets/")
    full_dataset_path = os.path.join(base_path_data, "plants_sm/train_dataset_all_data_plants/")
    for i in range(5):
        print(f"Loading dataset {i}")
        train_dataset = SingleInputDataset.from_csv(train_datasets_path + f"train_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))
        test_dataset = SingleInputDataset.from_csv(test_datasets_path + f"test_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))
        train_dataset_full = SingleInputDataset.from_csv(full_dataset_path + f"train_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))

        train_dataset.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))
        test_dataset.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))
        train_dataset_full.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))
        
        datasets.append((train_dataset, test_dataset, train_dataset_full))
    
    experiment = FineTuneExperimentTFvsNoTF_PlantsSMOrComplete(datasets=datasets, study_name="esm2_3b_experiment_plants_sm", storage="sqlite:///transfer_learning_experiment.db", sampler= optuna.samplers.RandomSampler(),
                                    direction="maximize", load_if_exists=True, path_to_model=os.path.join(base_path, "pretrained_models/esm2_3b_no_plants.ckpt"),
                                    base_layers=[2560], input_dim=2560, classification_neurons=182, results_output_file="results_esm2_3b_plants_sm.csv")
    
    experiment.run(n_trials=10, n_jobs=1)

def experiment_optimize_plants_sm():
    datasets = []
    # load datasets
    train_datasets_path = os.path.join(base_path_data, "plants_sm/train_datasets/")
    test_datasets_path = os.path.join(base_path_data, "plants_sm/test_datasets/")
    for i in range(5):
        print(f"Loading dataset {i}")
        train_dataset = SingleInputDataset.from_csv(train_datasets_path + f"train_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))
        test_dataset = SingleInputDataset.from_csv(test_datasets_path + f"test_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))

        train_dataset.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))
        test_dataset.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))
        
        datasets.append((train_dataset, test_dataset))
    
    experiment = FineTuneWithOptimizationForPlants(datasets=datasets, study_name="esm2_3b_experiment_with_optimization_plants_sm2", storage="sqlite:///transfer_learning_experiment.db", sampler= optuna.samplers.TPESampler(),
                                    direction="maximize", load_if_exists=True, path_to_model=os.path.join(base_path, "pretrained_models/esm2_3b_no_plants.ckpt"),
                                    base_layers=[2560], input_dim=2560, classification_neurons=182, results_output_file="results_esm2_3b_experiment_with_optimization_plants_sm.csv",
                                    folder_path="esm2_3b/trials")
    
    experiment.run(n_trials=50, n_jobs=1)

def train_model_and_evaluate_plants_sm():
    # load datasets

    from plants_sm.models.lightning_model import InternalLightningModel
    from ec_number_prediction.transfer_learning.models import ModelECNumber
    from plants_sm.io.pickle import read_pickle
    import os
    from plants_sm.models.constants import BINARY, FileConstants

    # path = os.path.join(base_path, "train_models/esm2_3b/trials/24/")
    # model = read_pickle(os.path.join(path, FileConstants.PYTORCH_MODEL_PKL.value))

    train_datasets_path = os.path.join(base_path_data, "plants_sm/train_datasets/")
    test_datasets_path = os.path.join(base_path_data, "plants_sm/test_datasets/")

    results_all = pd.DataFrame()
    for i in range(5):
        print(f"Loading dataset {i}")
        train_dataset = SingleInputDataset.from_csv(train_datasets_path + f"train_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))
        test_dataset = SingleInputDataset.from_csv(test_datasets_path + f"test_dataset_{i}.csv", representation_field="sequence", instances_ids_field="accession",
                                                    labels_field=slice(11,-1))

        train_dataset.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))
        test_dataset.load_features(os.path.join(base_path_data, "swiss_prot_ec_plants_esm/"))

        module = FineTuneModelECNumber(input_dim=2560, additional_layers=[1280], classification_neurons=182, \
            path_to_model=os.path.join(base_path, "pretrained_models/esm2_3b_no_plants.ckpt"),
            metric=f1_macro, learning_rate=0.0009776521450902962, base_layers=[2560], layers_to_freeze=len([2560]),
            scheduler=False)
        
        model = InternalLightningModel(module=module, max_epochs=93,
                batch_size=64,
                devices=[2],
                accelerator="gpu",
                # strategy="fsdp",)
                )
        
        model.fit(train_dataset)
        from sklearn.metrics import f1_score, precision_score, recall_score

        predictions = model.predict(test_dataset)

        mf1 = f1_score(test_dataset.y, predictions, average='macro')
        wf1 = f1_score(test_dataset.y, predictions, average='weighted')
        mrecall = recall_score(test_dataset.y, predictions, average='macro')
        wrecall = recall_score(test_dataset.y, predictions, average='weighted')
        mprecision = precision_score(test_dataset.y, predictions, average='macro')
        wprecision = precision_score(test_dataset.y, predictions, average='weighted')
        level_1, level_2, level_3, level_4 = get_ec_levels(test_dataset._labels_names)
        level_1_f1 = f1_score(test_dataset.y[:, level_1], predictions[:, level_1], average='macro')
        level_2_f1 = f1_score(test_dataset.y[:, level_2], predictions[:, level_2], average='macro')
        level_3_f1 = f1_score(test_dataset.y[:, level_3], predictions[:, level_3], average='macro')
        level_4_f1 = f1_score(test_dataset.y[:, level_4], predictions[:, level_4], average='macro')

        level_1_wf1 = f1_score(test_dataset.y[:, level_1], predictions[:, level_1], average='weighted')
        level_2_wf1 = f1_score(test_dataset.y[:, level_2], predictions[:, level_2], average='weighted')
        level_3_wf1 = f1_score(test_dataset.y[:, level_3], predictions[:, level_3], average='weighted')
        level_4_wf1 = f1_score(test_dataset.y[:, level_4], predictions[:, level_4], average='weighted')

        results = pd.DataFrame({"fold": [i], "mf1": [mf1], "wf1": [wf1], "mrecall": [mrecall], "wrecall": [wrecall], "mprecision": [mprecision], "wprecision": [wprecision],
                                "level_1_f1": [level_1_f1], "level_2_f1": [level_2_f1], "level_3_f1": [level_3_f1], "level_4_f1": [level_4_f1],
                                "level_1_wf1": [level_1_wf1], "level_2_wf1": [level_2_wf1], "level_3_wf1": [level_3_wf1], "level_4_wf1": [level_4_wf1],
                                "mf1_val": [np.NaN]})

        results_all = pd.concat([results_all, results])

        results_all.to_csv(os.path.join(base_path, "train_models/results_esm2_3b_trial_9_plants_sm.csv"))


# [additional_layers: [1280], batch_size: 64, epochs: 93, learning_rate: 0.0009776521450902962]

if __name__=="__main__":
    # experiment_tf_no_tf_plants_sm()
    # evaluate_model()
    # experiment_optimize_plants_sm()
    experiment_optimize()
    # train_model_and_evaluate_plants_sm()
