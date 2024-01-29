
import os

def generate_features_for_halogenases():
    from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset
    halogenases_dataset = SingleInputDataset.from_csv('/scratch/jribeiro/ec_number_prediction/final_data/halogenase.csv',
                                                instances_ids_field="Entry", representation_field="Sequence", sep="\t")

    from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder
    from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
    from plants_sm.data_standardization.truncation import Truncator


    transformers = [ProteinStandardizer(), Truncator(max_length=884), ESMEncoder(esm_function="esm2_t6_8M_UR50D", batch_size=1, num_gpus=1,
                                                                                model_path="/scratch/jribeiro/esm2_t6_8M_UR50D.pt")]
    for transformer in transformers:
        halogenases_dataset = transformer.fit_transform(halogenases_dataset)

    from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset

    halogenases_dataset = SingleInputDataset.from_csv('/scratch/jribeiro/ec_number_prediction/final_data/halogenase.csv',
                                            instances_ids_field="Entry", representation_field="Sequence", sep="\t")
    
    import os
    from plants_sm.featurization.proteins.bio_embeddings.prot_bert import ProtBert
    from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
    from plants_sm.data_standardization.truncation import Truncator
    from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder

    encoding = "esm2_t33_650M_UR50D"

    if not os.path.exists("esm2_t33_650M_UR50D_features_halogenases"):
        transformers = [ProteinStandardizer(), Truncator(max_length=884), ESMEncoder(esm_function=encoding, batch_size=1)]
        for transformer in transformers:
            halogenases_dataset = transformer.fit_transform(halogenases_dataset)
        halogenases_dataset.save_features("esm2_t33_650M_UR50D_features_halogenases")
    else:
        halogenases_dataset.load_features("esm2_t33_650M_UR50D_features_halogenases")


    encoding = "esm2_t36_3B_UR50D"

    if not os.path.exists("esm2_t36_3B_UR50D"):
        transformers = [ProteinStandardizer(), Truncator(max_length=884), ESMEncoder(esm_function=encoding, batch_size=1)]
        for transformer in transformers:
            halogenases_dataset = transformer.fit_transform(halogenases_dataset)
        halogenases_dataset.save_features("esm2_t36_3B_UR50D_features_halogenases")
    else:
        halogenases_dataset.load_features("esm2_t36_3B_UR50D_features_halogenases")

def generate_features_for_price():
    import os
    from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset
    price_dataset = SingleInputDataset.from_csv('/scratch/jribeiro/ec_number_prediction/final_data/price.csv',
                                                instances_ids_field="Entry", representation_field="Sequence", sep="\t")

    from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder
    from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
    from plants_sm.data_standardization.truncation import Truncator


    encoding = "esm1b_t33_650M_UR50S"

    if not os.path.exists("ESM1b_features_price"):
        transformers = [ProteinStandardizer(), Truncator(max_length=884), ESMEncoder(esm_function=encoding, batch_size=1, num_gpus=4)]
        for transformer in transformers:
            price_dataset = transformer.fit_transform(price_dataset)
        price_dataset.save_features("ESM1b_features_price")
    else:
        price_dataset.load_features("ESM1b_features_price")

    from plants_sm.data_structures.dataset.single_input_dataset import SingleInputDataset

    price_dataset = SingleInputDataset.from_csv('/scratch/jribeiro/ec_number_prediction/final_data/price.csv',
                                            instances_ids_field="Entry", representation_field="Sequence", sep="\t")
    
    import os
    from plants_sm.featurization.proteins.bio_embeddings.prot_bert import ProtBert
    from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
    from plants_sm.data_standardization.truncation import Truncator
    from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder

    encoding = "esm2_t33_650M_UR50D"

    if not os.path.exists("esm2_t33_650M_UR50D_features_price"):
        transformers = [ProteinStandardizer(), Truncator(max_length=884), ESMEncoder(esm_function=encoding, batch_size=1)]
        for transformer in transformers:
            price_dataset = transformer.fit_transform(price_dataset)
        price_dataset.save_features("esm2_t33_650M_UR50D_features_price")
    else:
        price_dataset.load_features("esm2_t33_650M_UR50D_features_price")


    encoding = "esm2_t36_3B_UR50D"

    if not os.path.exists("esm2_t36_3B_UR50D_features_price"):
        transformers = [ProteinStandardizer(), Truncator(max_length=884), ESMEncoder(esm_function=encoding, batch_size=1)]
        for transformer in transformers:
            price_dataset = transformer.fit_transform(price_dataset)
        price_dataset.save_features("esm2_t36_3B_UR50D_features_price")
    
    else:
        price_dataset.load_features("esm2_t36_3B_UR50D_features_price")

    if not os.path.exists("protbert_features_price"):
        transformers = [ProteinStandardizer(), Truncator(max_length=884), ProtBert(device="cuda:0")]
        for transformer in transformers:
            price_dataset = transformer.fit_transform(price_dataset)
        price_dataset.save_features("protbert_features_price")
    
    else:
        price_dataset.load_features("protbert_features_price")

if __name__ == "__main__":
    generate_features_for_price()