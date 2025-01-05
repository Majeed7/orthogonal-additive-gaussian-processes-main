import numpy as np
import pandas as pd

# Helper function to convert SHAP values to DataFrame
def shap_values_to_df(shap_values):
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    feature_names = [f"Feature_{i}" for i in range(shap_values.shape[1])]
    return pd.DataFrame(shap_values, columns=feature_names)