from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset
from plants_sm.data_standardization.truncation import Truncator
from plants_sm.data_standardization.x_padder import XPadder
from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder
from plants_sm.featurization.proteins.bio_embeddings.prot_bert import ProtBert


def generate_prot_bert_data(dataset, representation_field, ids_field, output_path):

    dataset = SingleInputDataset.from_csv(dataset, representation_field=representation_field, instances_ids_field=ids_field)
    transformers = [ProteinStandardizer(), Truncator(max_length=884), ProtBert(device="cuda")]
    for transformer in transformers:
        transformer.fit(dataset)
        transformer.transform(dataset)
    dataset.save_features(output_path)

def generate_esm_vectors(dataset, representation_field, ids_field, output_path):

    dataset = SingleInputDataset.from_csv(dataset, representation_field=representation_field, instances_ids_field=ids_field)
    transformers = [ProteinStandardizer(), Truncator(max_length=884), ESMEncoder(esm_function="esm2_t36_3B_UR50D", batch_size=1, num_gpus=4, 
                                                                                 device="cuda")]
    for transformer in transformers:
        transformer.fit(dataset)
        transformer.transform(dataset)
    dataset.save_features(output_path)

def generate_esm1b_vectors(dataset, representation_field, ids_field, output_path):
    
    dataset = SingleInputDataset.from_csv(dataset, representation_field=representation_field, instances_ids_field=ids_field)
    transformers = [ProteinStandardizer(), Truncator(max_length=884), ESMEncoder(esm_function="esm1b_t33_650M_UR50S", batch_size=1, num_gpus=4, 
                                                                                device="cuda")]
    for transformer in transformers:
        transformer.fit(dataset)
        transformer.transform(dataset)
    dataset.save_features(output_path)

if __name__ == "__main__":
    generate_prot_bert_data("/home/jcapela/ec_numbers_prediction/plants_pipeline/data/swiss_prot_ec_plants.csv", 
                            "sequence", "accession", "/home/jcapela/ec_numbers_prediction/plants_pipeline/data/swiss_prot_ec_plants_prot_bert")

    generate_esm_vectors("/home/jcapela/ec_numbers_prediction/plants_pipeline/data/swiss_prot_ec_plants.csv",
                            "sequence", "accession", "/home/jcapela/ec_numbers_prediction/plants_pipeline/data/swiss_prot_ec_plants_esm")
    
    generate_esm1b_vectors("/home/jcapela/ec_numbers_prediction/plants_pipeline/data/swiss_prot_ec_plants.csv",
                            "sequence", "accession", "/home/jcapela/ec_numbers_prediction/plants_pipeline/data/swiss_prot_ec_plants_esm1b")
    