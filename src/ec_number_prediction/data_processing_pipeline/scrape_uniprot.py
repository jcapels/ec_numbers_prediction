import luigi

from ec_number_prediction.data_processing_pipeline.uniprot_xml_parser import UniprotXmlECNumbersParser, UniprotEnzymesNonEnzymesXmlParser
from ec_number_prediction.data_processing_pipeline.download_uniprot import DownloadSwissProt, DownloadTrembl, DownloadUniref
from ec_number_prediction.data_processing_pipeline.uniref_xml_parser import UniRefXmlParser


class UniprotScraper(luigi.Task):

    def requires(self):
        return DownloadSwissProt(), DownloadTrembl(), DownloadUniref()

    def output(self):
        return luigi.LocalTarget('swiss_prot_ec.csv'), luigi.LocalTarget('trembl_prot_ec.csv'), luigi.LocalTarget('cluster_representatives.pkl')

    def run(self):
        UniprotXmlECNumbersParser(self.input()[0].path).parse(self.output()[0].path)
        UniprotXmlECNumbersParser(self.input()[1].path).parse(self.output()[1].path)
        UniRefXmlParser(self.input()[2].path).parse(self.output()[2].path)

class UniprotScraperEnzymes(luigi.Task):

    def requires(self):
        return DownloadSwissProt(), DownloadUniref()

    def output(self):
        return luigi.LocalTarget('swiss_prot_enzymes.csv')

    def run(self):
        UniprotEnzymesNonEnzymesXmlParser(self.input()[0].path).parse(self.output().path)
        UniRefXmlParser(self.input()[1].path).parse(self.output().path)

class UniprotScraper(luigi.Task):

    def requires(self):
        return DownloadSwissProt()

    def output(self):
        return luigi.LocalTarget('swiss_prot_enzymes.csv')
    
    def run(self):
        UniprotEnzymesNonEnzymesXmlParser(self.input()[0].path).parse(self.output().path)
