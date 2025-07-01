from ec_number_prediction.train_models.train_baselines import train_dnn_baselines

if __name__ == "__main__":
    import sys

    model = sys.argv[1]
    train_dnn_baselines(model)
