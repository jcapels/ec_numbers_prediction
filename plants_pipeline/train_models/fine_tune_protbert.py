from sklearn.metrics import f1_score
from ec_number_prediction.transfer_learning.models import FineTuneModelECNumber
from plants_sm.models.lightning_model import InternalLightningModel
from lightning.pytorch.callbacks import EarlyStopping
from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

def fine_tune_with_no_layers():
    train_dataset = SingleInputDataset.from_csv("/home/jcapela/ec_numbers_prediction/plants_pipeline/data/plants/train_datasets/train_dataset_0.csv", 
                representation_field="sequence", instances_ids_field="accession", labels_field=slice(11, -1))
    
    test_dataset = SingleInputDataset.from_csv("/home/jcapela/ec_numbers_prediction/plants_pipeline/data/plants/test_datasets/test_dataset_0.csv", 
                representation_field="sequence", instances_ids_field="accession", labels_field=slice(11, -1))
    
    train_dataset.load_features("/home/jcapela/ec_numbers_prediction/plants_pipeline/data/swiss_prot_ec_plants_prot_bert")
    test_dataset.load_features("/home/jcapela/ec_numbers_prediction/plants_pipeline/data/swiss_prot_ec_plants_prot_bert")

    module = FineTuneModelECNumber(input_dim=1024, additional_layers=[2560, 1280], classification_neurons=867, \
        path_to_model="/home/jcapela/ec_numbers_prediction/plants_pipeline/pretrained_models/protbert.pt",
        metric=f1_macro)
    
    callbacks = EarlyStopping("", patience=5, mode="max")
    
    model = InternalLightningModel(module=module, max_epochs=2,
            batch_size=32,
            devices=[0],
            accelerator="gpu",
            # strategy="fsdp",
            callbacks=[callbacks])
    
    model.fit(train_dataset, test_dataset)
    model.predict(test_dataset)

fine_tune_with_no_layers()
    
