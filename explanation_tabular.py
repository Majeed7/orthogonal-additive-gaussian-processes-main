import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from scipy.stats import kendalltau
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
import random

from lime.lime_tabular import LimeTabularExplainer
from shapreg import removal, games, shapley
from explainers.MAPLE import MAPLE
import shap
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr
import warnings
warnings.filterwarnings("ignore")

from explainers.bishapley_kernel import Bivariate_KernelExplainer
from real_datasets import load_dataset 

import time 
import openpyxl
from openpyxl import load_workbook
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

from pathlib import Path

from SHOGP import SHOGP

import sys

def feature_removing_effect(feature_importance, x, num_remove_feature, shogp):
    if x.ndim == 1:
        x = np.array(x).reshape(1,-1)

    rank = np.argsort(np.abs(feature_importance))
    y_x = shogp.OGP.predict(x)
    predic_diff = [y_x.item()]
        
    predic_diff += shogp.remove_features_one_by_one(x, rank, num_remove_feature)

    return np.array(predic_diff).squeeze()


if __name__ == "__main__":
        # Check if an argument is passed
    ds_index = 1
    if len(sys.argv) < 2:
        print("No argument passed. Using default value: 1")
        ds_index = 1
    else:
        # Retrieve the parameter passed from Bash
        parameter = sys.argv[1]

        # Try converting the argument to a number
        try:
            # Try converting to an integer
            ds_index = int(parameter)
        except ValueError:
            # If it fails, try converting to a float
            ds_index = 1
            print("Cannot process the value. Using default value: 0.1")

    dataset_names = ["breast_cancer", "nomao",  "sonar"]
    dataset_names2 = ["breast_cancer_wisconsin", "pumadyn32nm"]
    dataset_names3 = ["crime", "gas", "pol", 'parkinson', 'keggdirected']
    dataset_names4 =  ['keggdirected']
    dataset_names5 = ["skillcraft", "sml"]
    dataset_names6 = ["steel", "ionosphere"]
    dataset_names7 = ["pol", 'parkinson']


    all_ds = [dataset_names,dataset_names2,dataset_names3,dataset_names4,dataset_names5,dataset_names6,dataset_names7]
    datasets = all_ds[ds_index-1]
    # if ds_index == 1:
    #     datasets = dataset_names
    # elif ds_index == 2:
    #     datasets = dataset_names2
    # elif ds_index == 3:
    #     datasets = dataset_names3

    sampleNo_tbx = 30
    num_samples = 300

    for ds in datasets:
        print(f"\nProcessing dataset: {ds}")
        try:
            X, y = load_dataset(ds)
        except Exception as e:
            print(f"Failed to load dataset {ds}: {e}")
            continue

        # Determine if the dataset is for classification or regression
        mode = "classification" if type_of_target(y) in ["binary", "multiclass"] else "regression"

        if mode == "classification":
            label_encoder = LabelEncoder()
            label_encoder.fit_transform(y)
            y = label_encoder.fit_transform(y).reshape(-1, 1)
            continue
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        n, d = X_train.shape

        # Split the data into training and test sets
        n, d = X.shape
        num_remove_feature = int(np.ceil(d * 0.6))

        ## Training AGP model
        if X.shape[1] >=60:
            int_order = 3
        else:  
            int_order = 5

        shogp = SHOGP(X ,y, inte_order=int_order, inducing_points=200)
        exp_func = shogp.OGP.predict

        # Get baseline importance
        X_bg = X if X.shape[0] < 100 else shap.sample(X, 100)
        indices = np.random.choice(X.shape[0], size=min(sampleNo_tbx, X.shape[0]), replace=False)
        X_tbx = X[indices,:]


        # SHAP, Bivairate SHAP, and Unbiased SHAP setup
        shap_explainer = shap.KernelExplainer(exp_func, X_bg)
        shap_sampling_explainer = shap.SamplingExplainer(exp_func, X_bg)
        bshap = Bivariate_KernelExplainer(exp_func, X_bg)
        imputer = removal.MarginalExtension(X, exp_func)

        shogp_diff = []
        shogp_time = []

        kshap_diff = []
        kshap_time = []

        sshap_diff = []
        sshap_time = []

        bishap_diff = []
        bishap_time = []

        ushap_diff = []
        ushap_time = []

        lime_diff = []
        lime_time = []
        
        maple_diff = []
        maple_time = []

        
        
        # Perform experiment
        for i, x in enumerate(X_tbx):
            print(f"explaining sample number {i} for {ds}")
            # SHOGP shapley value
            st = time.time()
            shogp_sv = shogp.local_Shapley_values(x[np.newaxis,:])
            endt = time.time() - st
            shogp_removal_effect = feature_removing_effect(shogp_sv, x, num_remove_feature, shogp)

            shogp_diff.append(shogp_removal_effect)    
            shogp_time.append(endt)   

            # Kernel SHAP
            try:
                st = time.time()
                shap_values = shap_explainer.shap_values(x, nsamples=num_samples)        
                endt = time.time() - st 
                shap_removal_effect = feature_removing_effect(shap_values, x, num_remove_feature, shogp)

                kshap_diff.append(shap_removal_effect)    
                kshap_time.append(endt)             
            except:
                print(f"KSHAP did not work")
                pass

            # Sampling SHAP
            try:
                st = time.time()
                shap_sampling_values = shap_sampling_explainer.shap_values(x, nsamples=num_samples)
                endt = time.time() - st
                sshap_removal_effect = feature_removing_effect(shap_sampling_values, x, num_remove_feature, shogp)


                sshap_diff.append(sshap_removal_effect)    
                sshap_time.append(endt)            
            except:
                print(f"SSHAP did not work")
                pass

            # Bivariate SHAP
            try:
                st = time.time()
                bshap_values = bshap.shap_values(x, nsamples=num_samples)
                endt = time.time() - st
                bishap_removal_effect = feature_removing_effect(bshap_values, x, num_remove_feature, shogp)

                bishap_diff.append(bishap_removal_effect)    
                bishap_time.append(endt)  
            except:
                print(f"BSHAP did not work")
                pass

            ## Unbiased kernel shap
            try: 
                st = time.time()
                game = games.PredictionGame(imputer, x)
                values = shapley.ShapleyRegression(game, n_samples=num_samples, batch_size=int(num_samples / 10), paired_sampling=True)
                ushap_values = values.values.squeeze()   
                endt = time.time() - st

                ushap_removal_effect = feature_removing_effect(ushap_values, x, num_remove_feature, shogp)

                ushap_diff.append(ushap_removal_effect)    
                ushap_time.append(endt)         
            except:
                print(f"USHAP did not work")
                pass

            # LIME 
            try:
                lime_explainer = LimeTabularExplainer(training_data = X_bg, mode = mode)
                lime_values = []
                st = time.time()
                explanation = lime_explainer.explain_instance(data_row = x, predict_fn = exp_func, num_features = d)
                endt = time.time() - st
                exp = [0] * d
                for feature_idx, contribution in explanation.local_exp[0]:
                    exp[feature_idx] = contribution
                lime_values.append(exp)

                lime_removal_effect = feature_removing_effect(np.array(lime_values[0]), x, num_remove_feature, shogp)

                lime_diff.append(lime_removal_effect)    
                lime_time.append(endt)      
            except:
                print(f"LIME did not work")
                pass

            ## MAPLE
            try:
                st = time.time()
                exp_maple = MAPLE(X, y, X, y)
                mpl_exp = exp_maple.explain(x)
                maple_values = (mpl_exp['coefs'][1:]).squeeze()
                endt = time.time() - st 

                maple_removal_effect = feature_removing_effect(maple_values, x, num_remove_feature, shogp)

                maple_diff.append(maple_removal_effect)    
                maple_time.append(endt)      
            except:
                print(f"MAPLE did not work")
                pass
            

        all_results = [shogp_diff, kshap_diff, sshap_diff, bishap_diff, ushap_diff, lime_diff, maple_diff]
        time_results = [shogp_time, kshap_time, sshap_time, bishap_time, ushap_time, lime_time, maple_time]

        ## Storing the difference and execution time of the methods
        excel_path_feature_removal = Path(f'tabular_feature_removal {ds}.xlsx')
        excel_path_feature_removal_time = Path(f'tabular_feature_removal time {ds}.xlsx')
        mode = 'a' if excel_path_feature_removal.exists() else 'w'
        names = ['SHOGP', 'Kernel SHAP','Sampling SHAP', 'Bivariate SHAP', 'Unbiased SHAP', 'LIME', 'MAPLE']
        with pd.ExcelWriter(excel_path_feature_removal, engine='openpyxl', mode=mode) as writer:
            for i, l in enumerate(all_results):
                df = pd.DataFrame(all_results[i])
                df.to_excel(writer, sheet_name = names[i])    


        mode = 'a' if excel_path_feature_removal_time.exists() else 'w'
        with pd.ExcelWriter(excel_path_feature_removal_time, engine='openpyxl', mode=mode) as writer:
            df_time = pd.DataFrame(time_results)
            df_time.index = names

            df_time.to_excel(writer, index=True)


    print("done!")