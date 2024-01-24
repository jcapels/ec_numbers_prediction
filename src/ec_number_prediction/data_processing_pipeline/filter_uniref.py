import pickle
import luigi
import pandas as pd

from ec_number_prediction.data_processing_pipeline.scrape_uniprot import UniprotScraper


class FilterByUniRef90(luigi.Task):

    def requires(self):
        return UniprotScraper()

    def output(self):
        return [luigi.LocalTarget('trembl_prot_ec_filtered.csv'), luigi.LocalTarget('swiss_prot_ec_filtered.csv')]
    
    def input(self):
        return [luigi.LocalTarget('trembl_prot_ec.csv'), luigi.LocalTarget('swiss_prot_ec.csv'), luigi.LocalTarget('cluster_representatives.pkl')]

    def run(self):
        df_trembl = pd.read_csv(self.input()[0].path)
        df_swiss_prot = pd.read_csv(self.input()[1].path)

        with open(self.input()[2].path, 'rb') as fp:
            cluster_representatives = pickle.load(fp)

        df_sp_filtered = df_swiss_prot[df_swiss_prot["accession"].isin(cluster_representatives)]
        df_trembl_filtered = df_trembl[df_trembl["accession"].isin(cluster_representatives)]

        df_trembl_filtered.to_csv(self.output()[0].path, index=False)
        df_sp_filtered.to_csv(self.output()[1].path, index=False)