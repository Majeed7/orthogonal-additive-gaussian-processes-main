
import dill

with open(f"trained_models/classification/shogp_nomao.pkl", "rb") as f:
   shogp = dill.load(f)

sv = shogp.global_shapley_value()
