from ec_number_prediction.train_models.optimize_dnn import train_dnn_optimization, train_dnn_trials_merged, train_dnn_optimization_all_data

if __name__ == "__main__":
    import sys

    set_ = sys.argv[1]
    model = sys.argv[2]
    data = sys.argv[3]

    if data == "all":
        train_dnn_optimization_all_data(set_, model)
    elif data == "merged":
        train_dnn_trials_merged(set_, model)
    else:
        train_dnn_optimization(set_, model)
