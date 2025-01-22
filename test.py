import numpy as np
from real_datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import dill

from util import * 

if __name__ == "__main__":

   X, y = load_dataset("breast_cancer")
   label_encoder = LabelEncoder()
   label_encoder.fit_transform(y)
   y = label_encoder.fit_transform(y).reshape(-1, 1)

   imputer = SimpleImputer(strategy='mean')
   imputer.fit(X)
   X = imputer.transform(X) 

   with open(f"trained_models/regression/shogp_breast_cancer.pkl", "rb") as f:
      shogp = dill.load(f)

   sv = local_Shapley_values(shogp, X[:2,])
   
   sv_local = shogp.local_Shapley_values(X[:2,])

   
   sv = shogp.global_shapley_value()
