import luigi

from ec_number_prediction.data_processing_pipeline.monte_carlo_folds import MonteCarloFoldSplit

luigi.build([MonteCarloFoldSplit()], workers=1, scheduler_host='127.0.0.1',
            scheduler_port=8083, local_scheduler=True)