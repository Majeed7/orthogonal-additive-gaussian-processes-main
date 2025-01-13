import dill 
import numpy as np 

dataset_name = "pumadyn32nm"
  
with open(f"trained_models/shogp_{dataset_name}.pkl", "rb") as f:
        loaded_model = dill.load(f)

sv, sv_res = loaded_model.global_shapley_value()

print("done!")