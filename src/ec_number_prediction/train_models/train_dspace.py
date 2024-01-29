import os

from plants_sm.models.ec_number_prediction.d_space import DSPACE

from .pipeline_runner import PipelineRunner


def train_dspace(working_dir: str):
    """
    Train the DSPACE model with the training dataset and validate it with the validation dataset.

    Parameters
    ----------
    working_dir: str
        Working directory.
    """
    model = DSPACE
    pipeline = PipelineRunner("one_hot_encoding", last_sigmoid=True,
                              features_base_directory=working_dir,
                              datasets_directory=f"{working_dir}/data",
                              project_base_dir=working_dir,
                              losses_directory=f"{working_dir}/losses",
                              metrics_directory=f"{working_dir}/metrics",
                              )
    pipeline.train_model(model, num_tokens=20, input_size=884,
                         num_labels=2771,
                         batch_size=64, epochs=30,
                         device="cuda:0", logger_path="./logs.log",
                         progress=200,
                         patience=3,
                         tensorboard_file_path=os.path.join("results", "runs"),
                         early_stopping_method="metric",
                         objective="max",
                         model_name=f"DSPACE_epochs_{30}"
                         )


def train_dspace_merged(working_dir: str):
    """
    Train the DSPACE model with the training and validation dataset.

    Parameters
    ----------
    working_dir: str
        Working directory.
    """
    model = DSPACE
    pipeline = PipelineRunner("one_hot_encoding", last_sigmoid=True,
                              features_base_directory=working_dir,
                              datasets_directory=f"{working_dir}/data",
                              project_base_dir=working_dir,
                              losses_directory=f"{working_dir}/losses",
                              metrics_directory=f"{working_dir}/metrics",
                              )
    pipeline.train_model_with_validation_train_merged(model, num_tokens=20, input_size=884,
                                                      num_labels=2771,
                                                      batch_size=64, epochs=30,
                                                      device="cuda:0", logger_path="./logs.log",
                                                      progress=200,
                                                      patience=3,
                                                      tensorboard_file_path=os.path.join("results", "runs"),
                                                      early_stopping_method="metric",
                                                      objective="max",
                                                      model_name=f"DSPACE_merged"
                                                      )
