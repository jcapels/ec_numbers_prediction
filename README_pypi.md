# EC number prediction pipeline

### Description

Enzyme Commission (EC) numbers serve as a hierarchical system that categorizes 
and organizes enzyme activities. 
Within the realm of genome annotation, EC numbers are attributed to protein sequences 
to concisely represent specific chemical reaction patterns. 
These patterns mirror the chemical transformations enabled by the enzymes associated 
with their respective EC numbers. The structure of EC numbers is divided 
into four levels: (1) the primary category (e.g., 1 for oxidoreductases, 
2 for transferases, 3 for hydrolases, 4 for lyases, 5 for isomerases, 6 for ligases, 
and 7 for translocases), (2) the subclass (for instance, 1.2: Targets the aldehyde or 
oxo group of donors), (3) the sub-subclass (such as With NAD(+) or NADP(+) as acceptor), 
and (4) the final level, which identifies the enzyme's substrate (for example, 1.2.1.3:
aldehyde dehydrogenase (NAD(+))).

We have developed a robust framework for evaluating deep learning (DL) models dedicated 
to EC number prediction. These models are trained using embeddings from ESM2, ESM1b, and 
ProtBERT. Through this detailed method, our objective is to thoroughly examine the efficacy
of both BLASTp and DL models. This will enable us to provide insightful observations on the
superiority of DL models, augmented with the latest large language model (LLM) embeddings, 
over traditional alignment-based methods for predicting protein functions.



## Requirements
- Python >= 3.9
- BLAST >= 2.12.0

## Install with conda
    
```bash
conda create -n ec_numbers_prediction python=3.9
conda activate ec_numbers_prediction
conda install bioconda::blast==2.12.0
```

## Installation

### Pip

```bash
pip install ec-numbers-prediction
```

### From github

```bash
pip install git+https://github.com/jcapels/ec_numbers_prediction.git
```

## Run pipeline to obtain the data

Here is how you can obtain the data from the pipeline using luigi. This pipeline will download the data from the
[UniProt](https://www.uniprot.org/) database, and will process it to obtain the data used in the project.
This will generate several files in the working directory, some are intermediate files, other are the final division into training,
validation and test sets. The final files are the ones that will be used in the next steps of the pipeline. 

The steps of this pipeline are the following:

1. Download of UniProt data.
2. Scraping of UniProt enzymes.
3. Filter UniProt enzymes with UniRef90 cluster representatives.
4. Enrich underrepresented EC classes with TrEMBL data.
5. Generate a multi-label binary matrix for the resultant dataset.
6. Underrepresented classes removal.
7. Temporary EC numbers removal (e.g. EC 3.5.1.n3)
8. Split the data with a division of 60/20/20.

As for the intermediate files, those are the following:

- **uniprot_sprot.xml.gz** - SwissProt raw data.
- **uniprot_trembl.xml.gz** - TrEMBL raw data.
- **uniref90.xml.gz** - UniRef90 raw data.
- **trembl_prot_ec.csv** - TrEMBL enzymes with the respective assigned EC number.
- **swiss_prot_ec.csv** - SwissProt enzymes with the respective assigned EC number.
- **trembl_prot_ec_filtered.csv** - TrEMBL enzymes filtered with UniRef90 cluster representatives.
- **swiss_prot_ec_filtered.csv** - SwissProt enzymes filtered with UniRef90 cluster representatives.
- **dataset_enriched.csv** - Dataset with SwissProt enzymes enriched with TrEMBL.
- **dataset_binarized.csv** - Dataset with a multi-label binary matrix.
- **dataset_binarized_filtered.csv** - Dataset with underrepresented classes removed.
- **dataset_binarized_filtered_without_n.csv** - Dataset with underrepresented classes and temporary EC numbers removed.
- **train.csv** - Training data.
- **test.csv** - Test data.
- **validation.csv** - Validation data.

```python
from ec_number_prediction.run_data_processing_pipeline import EnzymesPipeline
import luigi

luigi.build([EnzymesPipeline()], workers=1, scheduler_host = '127.0.0.1',
        scheduler_port = 8083, local_scheduler = True)
```

After this, we shuffled all the datasets for being ready for training and evaluation. Here's an example:

```python
import pandas as pd

training_data = pd.read_csv("train.csv")
training_data = training_data.sample(frac = 1)
training_data.to_csv("train_shuffled.csv", index=False)
```

## Extract features

The following functions will take the data you have in your working directory (please pass its path as 
input to the following functions) and generate features using ESM, ProtBERT and one-hot encoding.

```python
from ec_number_prediction.feature_extraction.generate_features import generate_esm_vectors, generate_prot_bert_vectors, \
    generate_one_hot_encodings

generate_esm_vectors(esm_function="esm2_t33_650M_UR50D", 
                     save_folder="/home/working_dir", dataset_directory="/home/working_dir/data")

generate_prot_bert_vectors(save_folder="/home/working_dir", dataset_directory="/home/working_dir/data")

generate_one_hot_encodings(save_folder="/home/working_dir", dataset_directory="/home/working_dir/data")
```

## Train models

Training the baselines is also easy:

### Train baselines

```python
from ec_number_prediction.train_models.train_baselines import train_dnn_baselines

train_dnn_baselines(model = "esm2_t33_650M_UR50D", working_dir="/home/working_dir/")
```

### Train models

Train models with the specific sets, DeepEC and DSPACE. Note that the set chosen in the 
following examples is set 1, but
you can choose any of the sets 1, 2, 3, 4. Also, note that the model chosen in the following examples is
esm2_t33_650M_UR50D, but you can choose any of the models:

- esm2_t33_650M_UR50D
- esm1b_t33_650M_UR50S 
- esm2_t30_150M_UR50D
- esm2_t12_35M_UR50D
- esm2_t6_8M_UR50D
- esm2_t36_3B_UR50D
- prot_bert_vectors
- esm2_t48_15B_UR50D

```python
from ec_number_prediction.train_models.optimize_dnn import train_dnn_optimization
from ec_number_prediction.train_models.train_deep_ec import train_deep_ec
from ec_number_prediction.train_models.train_dspace import train_dspace 

train_dnn_optimization(set_="1", model = "esm2_t33_650M_UR50D", working_dir="/home/working_dir/")
train_deep_ec(working_dir="/home/working_dir/")
train_dspace(working_dir="/home/working_dir/")
```

### Train models with both training and validation sets

Train models with the training and validation sets merged.

```python
from ec_number_prediction.train_models.optimize_dnn import train_dnn_trials_merged
from ec_number_prediction.train_models.train_deep_ec import train_deep_ec_merged
from ec_number_prediction.train_models.train_dspace import train_dspace_merged 

train_dnn_trials_merged(set_="1", model = "esm2_t33_650M_UR50D", working_dir="/home/working_dir/")
train_deep_ec_merged(working_dir="/home/working_dir/")
train_dspace_merged(working_dir="/home/working_dir/")
```

### Train models with the whole data

Train models with the whole data.

```python
from ec_number_prediction.train_models.optimize_dnn import train_dnn_optimization_all_data

train_dnn_optimization_all_data(set_ = "1", model = "esm2_t33_650M_UR50D", working_dir="/home/working_dir/")
```

## Predict EC numbers

### Predict with model

Here you can see how to predict EC numbers with a model. Note that the model chosen in the following examples is
"DNN ProtBERT all data", but you can choose any of the models:

- DNN ProtBERT all data
- DNN ESM1b all data
- DNN ESM2 3B all data - note that this model requires at least **12 GB** of RAM to be run. If you intend to use GPU to 
make the predictions, you need to have at least **20 GB** of GPU memory or 4 GPUs with **8 GB**.
- ProtBERT trial 2 train plus validation (for this model, you need to pass all_data=False)
- DNN ESM1b trial 4 train plus validation (for this model, you need to pass all_data=False)
- DNN ESM2 3B trial 2 train plus validation (for this model, you need to pass all_data=False)

Here you can see the time taken and memory usage for each model to predict for different number of data points:

| Model           | Data Points | Time Taken | Memory Usage |
|-----------------|-------------|---------|--------------|
| DNN ProtBERT    | 25          | 0:00:05 | 1G           |
| DNN ProtBERT    | 100         | 0:00:08 | 1G           |
| DNN ProtBERT    | 1000        | 0:00:56 | 1G           |
| DNN ProtBERT    | 10000       | 0:09:00 | 1G           |
| DNN ProtBERT    | 100000      | 1:55:08 | 7G           |
| DNN ESM1b       | 25          | 0:00:28 | 2G           |
| DNN ESM1b       | 100         | 0:00:40 | 2G           |
| DNN ESM1b       | 1000        | 0:02:22 | 2G           |
| DNN ESM1b       | 10000       | 0:19:22 | 2G           |
| DNN ESM1b       | 100000      | 3:35:04 | 7G           |
| DNN ESM2 3B     | 25          | 0:01:35 | 10G          |
| DNN ESM2 3B     | 100         | 0:03:40 | 10G          |
| DNN ESM2 3B     | 1000        | 0:28:27 | 10G          |
| DNN ESM2 3B     | 10000       | 4:33:50 | 10G          |

The parameters of the function are the following:
- **pipeline**: name of the model to use.
- **dataset_path**: path to the dataset to predict.
- **output_path**: path to the output file.
- **ids_field**: name of the column with the ids.
- **all_data**: whether to use all the data or not.
- **sequences_field**: name of the column with the sequences.
- **device**: device to use for the predictions.

```python
from ec_number_prediction.predictions import predict_with_model


predict_with_model(pipeline="DNN ProtBERT all data",
                    dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                    output_path="predictions_prot_bert.csv",
                    ids_field="id",
                    all_data=True,
                    sequences_field="sequence",
                    device="cuda:0")
```

If don't have large enough GPU memory, you can use the following models but have at least 4 GPU with 10 GB of memory:

```python
from ec_number_prediction.predictions import predict_with_model


predict_with_model(pipeline="DNN ESM2 3B all data",
                    dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                    output_path="predictions_prot_bert.csv",
                    ids_field="id",
                    all_data=True,
                    sequences_field="sequence",
                    device="cuda", num_gpus=4)
```

If you don't have a GPU, you can use the CPU:

```python
from ec_number_prediction.predictions import predict_with_model

predict_with_model(pipeline="DNN ESM2 3B all data",
                    dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                    output_path="predictions_prot_bert.csv",
                    ids_field="id",
                    all_data=True,
                    sequences_field="sequence",
                    device="cpu")
```

You can also make predictions using a FASTA file:

```python
from ec_number_prediction.predictions import predict_with_model_from_fasta

predict_with_model_from_fasta(pipeline="DNN ProtBERT all data",
                    fasta_path="/home/jcapela/ec_numbers_prediction/data/test_data.fasta",
                    output_path="predictions_prot_bert.csv",
                    all_data=True,
                    device="cuda:0")
```

### Predict with BLAST

Here you can see how to predict EC numbers with BLAST. Note that the database chosen in the following examples is
"BLAST all data", but you can choose any of the databases:

- BLAST all data
- BLAST train plus validation

The parameters of the function are the following:
- **database_name**: name of the database to use.
- **dataset_path**: path to the dataset to predict.
- **output_path**: path to the output file.
- **ids_field**: name of the column with the ids.
- **sequences_field**: name of the column with the sequences.

```python
from ec_number_prediction.predictions import predict_with_blast

predict_with_blast(database_name="BLAST all data",
                            dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                            output_path="test_blast_predictions.csv",
                            ids_field="id",
                            sequences_field="sequence")
```

You can also make predictions using a FASTA file:

```python

from ec_number_prediction.predictions import predict_with_blast_from_fasta

predict_with_blast_from_fasta(database_name="BLAST all data",
                            fasta_path="/home/jcapela/ec_numbers_prediction/data/test_data.fasta",
                            output_path="test_blast_predictions.csv")
```


### Predict with an ensemble of BLAST and DL models

Here you can see how to predict EC numbers with an ensemble between BLAST and models.

The parameters of the function are the following:

- **dataset_path**: path to the dataset to predict.
- **output_path**: path to the output file.
- **ids_field**: name of the column with the ids.
- **sequences_field**: name of the column with the sequences.
- **device**: device to use for the predictions.

```python
from ec_number_prediction.predictions import predict_with_ensemble

predict_with_ensemble(dataset_path="/home/jcapela/ec_numbers_prediction/data/test_data.csv",
                        output_path="predictions_ensemble.csv",
                        ids_field="id",
                        sequences_field="sequence",
                        device="cuda:0")
```

You can also make predictions using a FASTA file:

```python
from ec_number_prediction.predictions import predict_with_ensemble_from_fasta

predict_with_ensemble_from_fasta(fasta_path="/home/jcapela/ec_numbers_prediction/data/test_data.fasta",
                        output_path="predictions_ensemble.csv",
                        device="cuda:0")
```