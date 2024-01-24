import re
import luigi
import numpy as np
import pandas as pd

from ec_number_prediction.data_processing_pipeline.enrichment_with_trembl import EnrichmentWithTrembl


class MultiLabelBinarizer(luigi.Task):

    def requires(self):
        return EnrichmentWithTrembl()
    
    def output(self):
        return luigi.LocalTarget('dataset_binarized.csv')
    
    def get_unique_labels_by_level(self, dataset, level):
        final_dataset_test = dataset.copy()
        final_dataset_test = final_dataset_test.loc[:,level]
        final_dataset_test.fillna("0", inplace=True)
        values = pd.Series(final_dataset_test.values.reshape(-1)).str.split(";")
        list_of_unique_labels = np.unique(values.explode()).tolist()
        if "0" in list_of_unique_labels:
            list_of_unique_labels.remove("0")
        list_of_unique_labels_dict = dict(zip(list_of_unique_labels, range(len(list_of_unique_labels))))
        return list_of_unique_labels_dict

    def get_final_labels(self, dataset):

        unique_EC1 = self.get_unique_labels_by_level(dataset, "EC1")
        unique_EC2 = self.get_unique_labels_by_level(dataset, "EC2")
        unique_EC3 = self.get_unique_labels_by_level(dataset, "EC3")
        unique_EC4 = self.get_unique_labels_by_level(dataset, "EC4")
        array_EC1 = np.zeros((len(dataset), len(unique_EC1)))
        array_EC2 = np.zeros((len(dataset), len(unique_EC2)))
        array_EC3 = np.zeros((len(dataset), len(unique_EC3)))
        array_EC4 = np.zeros((len(dataset), len(unique_EC4)))

        dataset.fillna("0", inplace=True)

        for i, row in dataset.iterrows():
            for ec in ["EC1", "EC2", "EC3", "EC4"]:
                for EC in row[ec].split(";"):
                    if EC != "0":
                        if ec == "EC1":
                            array_EC1[i, unique_EC1[EC]] = 1
                        elif ec == "EC2":
                            array_EC2[i, unique_EC2[EC]] = 1
                        elif ec == "EC3":
                            array_EC3[i, unique_EC3[EC]] = 1
                        elif ec == "EC4":
                            array_EC4[i, unique_EC4[EC]] = 1
        
        array_EC1 = pd.DataFrame(array_EC1, columns=unique_EC1.keys())
        array_EC2 = pd.DataFrame(array_EC2, columns=unique_EC2.keys())
        array_EC3 = pd.DataFrame(array_EC3, columns=unique_EC3.keys())
        array_EC4 = pd.DataFrame(array_EC4, columns=unique_EC4.keys())

        dataset = pd.concat((dataset, array_EC1, array_EC2, array_EC3, array_EC4), axis=1)
        return dataset
    

    def get_ec_from_regex_match(self, match):
        if match is not None:
            EC = match.group()
            if EC is not None:
                return EC
        return None
    
    def divide_labels_by_EC_level(self, final_dataset_path):
        final_dataset = pd.read_csv(final_dataset_path)

        EC1_lst = []
        EC2_lst = []
        EC3_lst = []
        EC4_lst = []


        for _, row in final_dataset.iterrows():
            ECs = row["EC"]
            ECs = ECs.split(";")
            # get the first 3 ECs with regular expression
            EC3 = []
            EC2 = []
            EC1 = []
            EC4 = []
            for EC in ECs:
                new_EC = re.search(r"^\d+.\d+.\d+.n*\d+", EC)
                new_EC = self.get_ec_from_regex_match(new_EC)
                if isinstance(new_EC, str):
                    if new_EC not in EC4:
                        EC4.append(new_EC)

                new_EC = re.search(r"^\d+.\d+.\d+", EC)
                new_EC = self.get_ec_from_regex_match(new_EC)
                if isinstance(new_EC, str):
                    if new_EC not in EC3:
                        EC3.append(new_EC)

                new_EC = re.search(r"^\d+.\d+", EC)
                new_EC = self.get_ec_from_regex_match(new_EC)
                if isinstance(new_EC, str):
                    if new_EC not in EC2:
                        EC2.append(new_EC)

                new_EC = re.search(r"^\d+", EC)
                new_EC = self.get_ec_from_regex_match(new_EC)
                if isinstance(new_EC, str):
                    if new_EC not in EC1:
                        EC1.append(new_EC)

            if len(EC4) == 0:
                EC4_lst.append(np.NaN)
            else:
                EC4_lst.append(";".join(EC4))
            if len(EC3) == 0:
                EC3_lst.append(np.NaN)
            else:
                EC3_lst.append(";".join(EC3))
            if len(EC2) == 0:
                EC2_lst.append(np.NaN)
            else:
                EC2_lst.append(";".join(EC2))
            if len(EC1) == 0:
                EC1_lst.append(np.NaN)
            else:
                EC1_lst.append(";".join(EC1))

        assert None not in EC1_lst
        assert None not in EC2_lst
        assert None not in EC3_lst
        assert None not in EC4_lst

        assert len(EC1_lst) == len(final_dataset)
        assert len(EC2_lst) == len(final_dataset)
        assert len(EC3_lst) == len(final_dataset)
        assert len(EC4_lst) == len(final_dataset)

        final_dataset["EC1"] = EC1_lst
        final_dataset["EC2"] = EC2_lst
        final_dataset["EC3"] = EC3_lst
        final_dataset["EC4"] = EC4_lst

        assert final_dataset["EC1"].isnull().sum() == 0
        print("EC1 is not null")

        return final_dataset


    def run(self):
        final_dataset = self.divide_labels_by_EC_level(self.input().path)
        final_dataset = self.get_final_labels(final_dataset)
        final_dataset.to_csv(self.output().path, index=False)