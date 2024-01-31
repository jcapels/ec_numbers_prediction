import datetime
import time
from ec_number_prediction.predictions import predict_with_blast_from_fasta, predict_with_ensemble_from_fasta, predict_with_model_from_fasta
import tracemalloc
import pandas as pd
from hurry.filesize import size


def benchmark_resources(cuda):
    pipelines = ["DNN ProtBERT all data", "DNN ESM1b all data", "DNN ESM2 3B all data", "BLAST all data", "Ensemble"]
    datasets = [(25, "test_25.fasta"), (100, "test_100.fasta"), (1000, "test_1000.fasta"), 
                (10000, "test_10000.fasta"), (100000, "test_100000.fasta")]
    results = pd.DataFrame(columns=["pipeline", "dataset", "time", "memory"])
    for pipeline in pipelines:
        for dataset in datasets:
            tracemalloc.start()
            start = time.time()
            if pipeline == "BLAST all data":
                predict_with_blast_from_fasta(fasta_path=dataset[1],
                                              output_path="predictions.csv",
                                              database_name="BLAST all data"
                                              )
            elif pipeline == "Ensemble":
                if cuda:
                    predict_with_ensemble_from_fasta(fasta_path=dataset[1],
                                                     output_path="predictions.csv",
                                                     device="cuda", num_gpus=4)
                else:
                    predict_with_ensemble_from_fasta(fasta_path=dataset[1],
                                                     output_path="predictions.csv",
                                                     device="cpu")
            else:
                if cuda:
                    predict_with_model_from_fasta(pipeline=pipeline,
                                                fasta_path=dataset[1],
                                                output_path="predictions.csv",
                                                device="cuda", num_gpus=4)
                else:
                    predict_with_model_from_fasta(pipeline=pipeline,
                                                fasta_path=dataset[1],
                                                output_path="predictions.csv",
                                                device="cpu")
            end = time.time()
            print("Time spent: ", end - start)
            print("Memory needed: ", tracemalloc.get_traced_memory())
            results = pd.concat((results, 
                                    pd.DataFrame({"pipeline": [pipeline], 
                                                  "dataset": [dataset[0]], 
                                                  "time": [str(datetime.timedelta(seconds=end - start))], 
                                                  "memory": [size(int(tracemalloc.get_traced_memory()[1]))]})), 
                                                  ignore_index=True, axis=0)
            tracemalloc.stop()

            results.to_csv("benchmark_results.csv", index=False)
            
if __name__ == "__main__":
    benchmark_resources(cuda=True)