# EC number prediction pipeline

### Description


### Table of contents:

- [Requirements](#requirements)
- [Installation](#installation)
    - [Pip](#pip)
    - [From github](#From-github)
- [Getting Started](#getting-started)
    - [Run pipeline to obtain the data](#Run-pipeline-to-obtain-the-data)
    - [Extract features](#extract-features)
    - [Train models](#Train-models)
      - [Train baselines](#Train-baselines)

## Requirements
- Python >= 3.9
- BLAST >= 2.11.0

## Installation

### Pip

Under construction

### From github

```bash
pip install git+https://github.com/jcapels/PlantsSM.git
pip install git+https://github.com/jcapels/ec_numbers_prediction.git
```

## Getting Started

### Run pipeline to obtain the data

```python
from ec_number_prediction.run_data_processing_pipeline import EnzymesPipeline
import luigi

luigi.build([EnzymesPipeline()], workers=1, scheduler_host = '127.0.0.1',
        scheduler_port = 8083, local_scheduler = True)
```

### Extract features

```python
from ec_number_prediction.feature_extraction.generate_features import generate_esm_vectors, generate_prot_bert_vectors, \
generate_one_hot_encodings

generate_esm_vectors(esm_function="esm2_t33_650M_UR50D", 
                     save_folder="/home/working_dir", dataset_directory="/home/working_dir/data")

generate_prot_bert_vectors(save_folder="/home/working_dir", dataset_directory="/home/working_dir/data")

generate_one_hot_encodings(save_folder="/home/working_dir", dataset_directory="/home/working_dir/data")
```

### Train models

#### Train baselines

```python
from ec_number_prediction.train_models.train_baselines import train_dnn_baselines

train_dnn_baselines(model = "esm2_t33_650M_UR50D", working_dir="/home/working_dir/")
```

#### Train models

```python
from ec_number_prediction.train_models.optimize_dnn import train_dnn_optimization
from ec_number_prediction.train_models.train_deep_ec import train_deep_ec
from ec_number_prediction.train_models.train_dspace import train_dspace 

train_dnn_optimization(set_="1", model = "esm2_t33_650M_UR50D", working_dir="/home/working_dir/")
train_deep_ec(working_dir="/home/working_dir/")
train_dspace(working_dir="/home/working_dir/")
```






