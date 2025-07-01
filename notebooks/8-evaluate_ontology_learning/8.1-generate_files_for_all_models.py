import pandas as pd

def get_predictions_format_for_model(model):
    test = pd.read_csv("../required_data_ec_number_paper/data/test.csv")
    accessions = test["accession"]
    terms = test.iloc[:, 8:].columns

    from plants_sm.io.pickle import read_pickle

    test_esm1b = read_pickle(f"../pr_auc_validation/{model}_predictions/test_{model}_predictions.pkl")

    import numpy as np

    indexes = [np.where(row >= 0.02)[0] for row in test_esm1b]
    terms = np.array(terms)

    from tqdm import tqdm 
    results = pd.DataFrame()

    accessions_ = []
    ec_list = []
    prob = []
    for i, index_ in tqdm(enumerate(indexes), total=75566):
        ecs = terms[index_]
        for j, ec in enumerate(ecs):
            accessions_.append(accessions[i])
            ec_list.append(f"EC:{ec}")
            prob.append(test_esm1b[i, index_[j]])

    results = pd.DataFrame({
        "accession": accessions_,
        "EC": ec_list,
        "Prob": prob
    })
    results.to_csv(f"{model}_results.tsv", index=False, sep="\t", header=False)

get_predictions_format_for_model("esm1b")
get_predictions_format_for_model("prot_bert")
get_predictions_format_for_model("esm2_3B")
get_predictions_format_for_model("dspace")
get_predictions_format_for_model("deep_ec")