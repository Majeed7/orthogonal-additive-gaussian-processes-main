import dill 
import numpy as np 

dataset_name = "keggdirected"
  
with open(f"trained_models/shogp_{dataset_name}_sgpr.pkl", "rb") as f:
        shogp = dill.load(f)

sv, sv_res = shogp.global_shapley_value()

print("done!")