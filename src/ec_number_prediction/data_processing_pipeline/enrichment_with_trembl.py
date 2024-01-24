import luigi
from tqdm import tqdm

from ec_number_prediction.data_processing_pipeline.filter_uniref import FilterByUniRef90

import pandas as pd

import re


class EnrichmentWithTrembl(luigi.Task):

    def requires(self):
        return FilterByUniRef90()

    def input(self):
        return [luigi.LocalTarget('trembl_prot_ec_filtered.csv'), luigi.LocalTarget('swiss_prot_ec_filtered.csv')]

    def output(self):
        return luigi.LocalTarget('dataset_enriched.csv')

    def get_ec_numbers_with_less_than_n_sequences(self, df, n):
        return df[df["num_sequences"] < n]

    def get_sequences_per_ec(self, swiss_prot_dataframe, take_out_incomplete_ecs=True, level=4):
        df = swiss_prot_dataframe
        print("Number of proteins: {}".format(len(df)))
        df = df.dropna()
        print("Number of proteins: {}".format(len(df)))
        df = df.drop_duplicates(subset="accession")
        print("Number of proteins: {}".format(len(df)))
        df["EC"] = df["EC"].str.split(";")

        # cut ECs by level (e.g. level 4 is 1.1.1.1 and level 3 is 1.1.1) with regex
        if level == 1:
            df["EC"] = df["EC"].apply(lambda x: [re.search(r"^\d+", i).group() for i in x if re.search(r"^\d+", i)])
        elif level == 2:
            df["EC"] = df["EC"].apply(
                lambda x: [re.search(r"^\d+.\d+", i).group() for i in x if re.search(r"^\d+.\d+", i)])
        elif level == 3:
            df["EC"] = df["EC"].apply(
                lambda x: [re.search(r"^\d+.\d+.\d+", i).group() for i in x if re.search(r"^\d+.\d+.\d+", i)])
        elif level == 4:
            df["EC"] = df["EC"].apply(lambda x: [re.search(r"^\d+.\d+.\d+.n*\d+", i).group() for i in x if
                                                 re.search(r"^\d+.\d+.\d+.n*\d+", i)])

        df = df.explode("EC")

        df = df.dropna()
        df = df.groupby("EC").agg({"sequence": "unique"})
        df = df.reset_index()
        # drop incomplete ECs
        if take_out_incomplete_ecs:
            df = df[~df["EC"].str.contains("-")]

        df["sequence"] = df["sequence"].apply(lambda x: list(x))
        df = df.reset_index(drop=True)
        # get the number of sequences per EC
        df["num_sequences"] = df["sequence"].apply(lambda x: len(x))

        # generate a graph of the number of ECs with a minimum number of sequences with seaborn
        print("Number of ECs: {}".format(len(df)))

        return df

    def create_enriched_dataset(self, sequences_per_ec_less_than_100):
        trembl = pd.read_csv(self.input()[0].path)

        # put first column as keys of dictionary and the rsecond as values
        ecs_for_enrichment_dict = dict(
            zip(sequences_per_ec_less_than_100.loc[:, "EC"], sequences_per_ec_less_than_100.loc[:, "num_sequences"]))

        bar = tqdm(total=trembl.shape[0])
        ecs_for_enrichment_dict_keys = set(ecs_for_enrichment_dict.keys())
        sequences_added = set()
        sequences = []
        for _, row in trembl.iterrows():

            ECs = row['EC']
            ECs = ECs.split(';')
            if len(ECs) < 2:
                for EC in ECs:
                    # level 4 enrichment
                    if EC in ecs_for_enrichment_dict_keys and ecs_for_enrichment_dict[EC] < 100 and row[
                        "accession"] not in sequences_added:
                        sequences.append(row.to_list())
                        sequences_added.add(row["accession"])
                        ecs_for_enrichment_dict[EC] += 1

            bar.update(1)

        trembl_for_enrichment = pd.DataFrame(sequences, columns=['accession', 'name', 'sequence', 'EC'])
        return trembl_for_enrichment

    def run(self):

        swiss_prot = pd.read_csv(self.input()[1].path)

        sequences_per_ec = self.get_sequences_per_ec(swiss_prot, level=4)
        sequences_per_ec_less_than_100 = self.get_ec_numbers_with_less_than_n_sequences(sequences_per_ec, 100)

        trembl_for_enrichment = self.create_enriched_dataset(sequences_per_ec_less_than_100)

        swiss_prot = pd.read_csv(self.input()[1].path)

        dataset = pd.concat([swiss_prot, trembl_for_enrichment], ignore_index=True)

        dataset.to_csv(self.output().path, index=False)
