import luigi

import sys

sys.path.append("../data_processing_pipeline")

from .data_processing_pipeline.split import StratifiedSplitECNumbers, RandomSplitEnzymes


class EnzymesPipeline(luigi.WrapperTask):


    def requires(self):
        return StratifiedSplitECNumbers()
    
class EnzymesNonEnzymePipeline(luigi.WrapperTask):


    def requires(self):
        return RandomSplitEnzymes()


if __name__ == "__main__":
    luigi.build([EnzymesNonEnzymePipeline()], workers=1, scheduler_host = '127.0.0.1',
        scheduler_port = 8083, local_scheduler = True)
