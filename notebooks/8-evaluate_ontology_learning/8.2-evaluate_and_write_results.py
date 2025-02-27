# Run the evaluation
from cafaeval.evaluation import cafa_eval, write_results

res = cafa_eval("data/ontology_output.obo", "pred_dir", "gt.tsv")
write_results(*res)
