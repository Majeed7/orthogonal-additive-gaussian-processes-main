
import dill

with open(f"trained_models/classification/shogp_sonar.pkl", "rb") as f:
   shogp = dill.load(f)

sv = shogp.global_shapley_value()
