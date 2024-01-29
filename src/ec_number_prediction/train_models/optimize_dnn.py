import torch
from plants_sm.models.fc.fc import DNN

from ec_number_prediction.train_models.pipeline_runner import PipelineRunner


def train_dnn_optimization(set_: str, model: str, working_dir: str, epochs: int = 30,
                           dropout: float = None, batch_size: int = 64, lr: float = 0.0001,
                           weight_decay: float = 0.0):
    """
    Train the DNN model with the training dataset and validate it with the validation dataset.

    Parameters
    ----------
    set_: str
        Set to be used.
    model: str
        Model to be used.
    working_dir
        Working directory.
    epochs: int
        Number of epochs.
    dropout: float
        Dropout to be used.
    batch_size: int
        Batch size to be used.
    lr: float
        Learning rate to be used.
    weight_decay: float
        Weight decay to be used.

    Returns
    -------

    """
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
    else:
        raise ValueError("Model not found. Options are: esm2_t33_650M_UR50D, esm1b_t33_650M_UR50S, "
                         "esm2_t30_150M_UR50D, esm2_t12_35M_UR50D, esm2_t6_8M_UR50D, esm2_t36_3B_UR50D, "
                         "prot_bert_vectors, esm2_t48_15B_UR50D.")

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

    else:
        raise ValueError("Set not found. Options are: 1, 2, 3, 4, 5, 6.")

    pipeline = PipelineRunner(model, last_sigmoid=True,
                              features_base_directory=working_dir,
                              datasets_directory=f"{working_dir}/data",
                              project_base_dir=working_dir,
                              losses_directory=f"{working_dir}/losses",
                              metrics_directory=f"{working_dir}/metrics",
                              )
    pipeline.train_model(fc_model, num_tokens=None, input_size=None, num_labels=None,
                         optimizer=torch.optim.Adam(params=fc_model.parameters(), lr=lr, weight_decay=weight_decay),
                         batch_size=batch_size, epochs=epochs,
                         device="cuda:0", logger_path="./logs.log",
                         progress=200,
                         patience=3,
                         early_stopping_method="metric",
                         objective="max",
                         model_name=f"DNN_{model}_optimization_set_{set_}")


def train_dnn_optimization_all_data(set_: str, model: str, working_dir: str , epochs: int = 30,
                                    dropout: float = None, batch_size: int = 64, lr: float = 0.0001,
                                    weight_decay: float = 0.0):
    """
    Train the DNN model with all the data.

    Parameters
    ----------
    set_: str
        Set to be used.
    model: str
        Model to be used.
    working_dir: str
        Working directory.
    epochs: int
        Number of epochs.
    dropout: float
        Dropout to be used.
    batch_size: int
        Batch size to be used.
    lr: float
        Learning rate to be used.
    weight_decay: float
        Weight decay to be used.

    Returns
    -------

    """
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
    else:
        raise ValueError("Model not found. Options are: esm2_t33_650M_UR50D, esm1b_t33_650M_UR50S, "
                         "esm2_t30_150M_UR50D, esm2_t12_35M_UR50D, esm2_t6_8M_UR50D, esm2_t36_3B_UR50D, "
                         "prot_bert_vectors.")

    if set_ == "1":
        fc_model = DNN(input_dim, [640], 5743, batch_norm=True, last_sigmoid=True, dropout=dropout)

    elif set_ == "2":

        fc_model = DNN(input_dim, [2560], 5743, batch_norm=True, last_sigmoid=True, dropout=dropout)

    elif set_ == "3":

        fc_model = DNN(input_dim, [640, 320], 5743, batch_norm=True, last_sigmoid=True, dropout=dropout)

    elif set_ == "4":

        fc_model = DNN(input_dim, [2560, 5120], 5743, batch_norm=True, last_sigmoid=True, dropout=dropout)

    else:
        raise ValueError("Set not found. Options are: 1, 2, 3, 4.")

    pipeline = PipelineRunner(model, last_sigmoid=True,
                              features_base_directory=working_dir,
                              datasets_directory=f"{working_dir}/data",
                              project_base_dir=working_dir,
                              losses_directory=f"{working_dir}/losses",
                              metrics_directory=f"{working_dir}/metrics",
                              )
    pipeline.train_model_with_all_data(fc_model, num_tokens=None, input_size=None, num_labels=None,
                                       optimizer=torch.optim.Adam(params=fc_model.parameters(), lr=lr,
                                                                  weight_decay=weight_decay),
                                       batch_size=batch_size, epochs=epochs,
                                       device="cuda:0", logger_path="./logs.log",
                                       progress=200,
                                       patience=3,
                                       early_stopping_method="metric",
                                       objective="max",
                                       model_name=f"DNN_{model}_optimization_set_{set_}_all_data")


def train_dnn_trials_merged(set_: str, model: str, working_dir: str, epochs: int = 30,
                            dropout: float = None, batch_size: int = 64, lr: float = 0.0001,
                            weight_decay: float = 0.0):
    """
    Train the DNN model with the training and validation dataset merged.

    Parameters
    ----------
    set_: str
        Set to be used.
    model: str
        Model to be used.
    working_dir: str
        Working directory.
    epochs: int
        Number of epochs.
    dropout: float
        Dropout to be used.
    batch_size: int
        Batch size to be used.
    lr: float
        Learning rate to be used.
    weight_decay: float
        Weight decay to be used.

    Returns
    -------

    """
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
    else:
        raise ValueError("Model not found. Options are: esm2_t33_650M_UR50D, esm1b_t33_650M_UR50S, "
                         "esm2_t30_150M_UR50D, esm2_t12_35M_UR50D, esm2_t6_8M_UR50D, esm2_t36_3B_UR50D, "
                         "prot_bert_vectors.")

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

    else:
        raise ValueError("Set not found. Options are: 1, 2, 3, 4, 5, 6.")

    pipeline = PipelineRunner(model, last_sigmoid=True,
                              features_base_directory=working_dir,
                              datasets_directory=f"{working_dir}/data",
                              project_base_dir=working_dir,
                              losses_directory=f"{working_dir}/losses",
                              metrics_directory=f"{working_dir}/metrics",
                              )
    pipeline.train_model_with_validation_train_merged(fc_model, num_tokens=None, input_size=None, num_labels=None,
                                                      optimizer=torch.optim.Adam(params=fc_model.parameters(), lr=lr,
                                                                                 weight_decay=weight_decay),
                                                      batch_size=batch_size, epochs=epochs,
                                                      device="cuda:0", logger_path="./logs.log",
                                                      progress=200,
                                                      model_name=f"DNN_{model}_trial_{set_}")
