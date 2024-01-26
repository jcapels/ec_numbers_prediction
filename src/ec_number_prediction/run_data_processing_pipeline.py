import luigi

import sys

sys.path.append("../data_processing_pipeline")

from .data_processing_pipeline.split import StratifiedSplit


class EnzymesPipeline(luigi.WrapperTask):

    def requires(self):
        return StratifiedSplit()
