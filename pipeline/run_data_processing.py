import luigi

from ec_number_prediction.run_data_processing_pipeline import EnzymesPipeline

luigi.build([EnzymesPipeline()], workers=1, scheduler_host='127.0.0.1',
            scheduler_port=8083, local_scheduler=True)
