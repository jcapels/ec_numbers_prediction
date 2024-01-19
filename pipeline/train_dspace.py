from ec_number_prediction.train_models.train_dspace import train_dspace, train_dspace_merged

if __name__ == "__main__":
    import sys

    data = sys.argv[1]
    if data == "merged":
        train_dspace_merged()
    else:
        train_dspace()
