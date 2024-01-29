import torch
from plants_sm.models.fc.fc import DNN

from ec_number_prediction.train_models.pipeline_runner import PipelineRunner


def train_dnn_baselines(model: str, working_dir: str):
    """
    Train the DNN model with the training dataset and validate it with the validation dataset.

    Parameters
    ----------
    model: str
        Model to be used.
    working_dir: str
        Working directory.
    """
    if model == "esm2_t33_650M_UR50D":
        fc_model = DNN(1280, [], 2771, batch_norm=True, last_sigmoid=True)
    elif model == "esm1b_t33_650M_UR50S":
        fc_model = DNN(1280, [], 2771, batch_norm=True, last_sigmoid=True)
    elif model == "esm2_t30_150M_UR50D":
        fc_model = DNN(640, [], 2771, batch_norm=True, last_sigmoid=True)
    elif model == "esm2_t12_35M_UR50D":
        fc_model = DNN(480, [], 2771, batch_norm=True, last_sigmoid=True)
    elif model == "esm2_t6_8M_UR50D":
        fc_model = DNN(320, [], 2771, batch_norm=True, last_sigmoid=True)
    elif model == "esm2_t36_3B_UR50D":
        fc_model = DNN(2560, [], 2771, batch_norm=True, last_sigmoid=True)
    elif model == "prot_bert_vectors":
        fc_model = DNN(1024, [], 2771, batch_norm=True, last_sigmoid=True)
    else:
        raise ValueError("Model not found. Options are: esm2_t33_650M_UR50D, esm1b_t33_650M_UR50S, "
                         "esm2_t30_150M_UR50D, esm2_t12_35M_UR50D, esm2_t6_8M_UR50D, esm2_t36_3B_UR50D, "
                         "prot_bert_vectors.")

    pipeline = PipelineRunner(model, last_sigmoid=True,
                              features_base_directory=working_dir,
                              datasets_directory=f"{working_dir}/data",
                              project_base_dir=working_dir,
                              losses_directory=f"{working_dir}/losses",
                              metrics_directory=f"{working_dir}/metrics",
                              )
    pipeline.train_model(fc_model, num_tokens=None, input_size=None, num_labels=None,
                         optimizer=torch.optim.Adam(params=fc_model.parameters(), lr=0.0001),
                         batch_size=64, epochs=30,
                         device="cuda:0", logger_path="./logs.log",
                         progress=200,
                         patience=3,
                         early_stopping_method="metric",
                         objective="max",
                         model_name=f"DNN_{model}_baseline")


def train_dnn_baselines_merged(model: str, working_dir: str):
    """
    Train the DNN model with the training and validation dataset.

    Parameters
    ----------
    model: str
        Model to be used.
    working_dir: str
        Working directory.
    """
    if model == "esm2_t33_650M_UR50D":
        fc_model = DNN(1280, [], 2771, batch_norm=True, last_sigmoid=True)
    elif model == "esm1b_t33_650M_UR50S":
        fc_model = DNN(1280, [], 2771, batch_norm=True, last_sigmoid=True)
    elif model == "esm2_t30_150M_UR50D":
        fc_model = DNN(640, [], 2771, batch_norm=True, last_sigmoid=True)
    elif model == "esm2_t12_35M_UR50D":
        fc_model = DNN(480, [], 2771, batch_norm=True, last_sigmoid=True)
    elif model == "esm2_t6_8M_UR50D":
        fc_model = DNN(320, [], 2771, batch_norm=True, last_sigmoid=True)
    elif model == "esm2_t36_3B_UR50D":
        fc_model = DNN(2560, [], 2771, batch_norm=True, last_sigmoid=True)
    elif model == "prot_bert_vectors":
        fc_model = DNN(1024, [], 2771, batch_norm=True, last_sigmoid=True)
    else:
        raise ValueError("Model not found. Options are: esm2_t33_650M_UR50D, esm1b_t33_650M_UR50S, "
                         "esm2_t30_150M_UR50D, esm2_t12_35M_UR50D, esm2_t6_8M_UR50D, esm2_t36_3B_UR50D, "
                         "prot_bert_vectors.")

    pipeline = PipelineRunner(model, last_sigmoid=True, features_base_directory=working_dir,
                              datasets_directory=f"{working_dir}/data",
                              project_base_dir=working_dir,
                              losses_directory=f"{working_dir}/losses",
                              metrics_directory=f"{working_dir}/metrics",)
    pipeline.train_model_with_validation_train_merged(fc_model, num_tokens=None, input_size=None, num_labels=None,
                                                      optimizer=torch.optim.Adam(params=fc_model.parameters(),
                                                                                 lr=0.0001),
                                                      batch_size=64, epochs=30,
                                                      device="cuda:0", logger_path="./logs.log",
                                                      progress=200,
                                                      patience=3,
                                                      early_stopping_method="metric",
                                                      objective="max",
                                                      model_name=f"DNN_{model}_baseline")

