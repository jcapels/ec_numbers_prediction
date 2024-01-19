from ec_number_prediction.train_models.train_deep_ec import train_deep_ec_merged, train_deep_ec

if __name__ == "__main__":
    import sys

    data = sys.argv[1]
    if data == "merged":
        train_deep_ec_merged()
    else:
        train_deep_ec()
