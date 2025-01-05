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

from tabular_datasets import * 

from explainers.bishapley_kernel import Bivariate_KernelExplainer

import time 
import openpyxl
from openpyxl import load_workbook

from pathlib import Path

from SHOGP import SHOGP


def feature_removing_effect(feature_importance, x, remove_feature, shogp):
    
    sorted_features = np.argsort(np.abs(feature_importance), axis=0)
    predic_diff = []
    y_x = shogp.OGP.predict(x[np.newaxis,:])
    
    active_dim = np.ones((len(x,)))
    for j in range(remove_feature):
        active_dim[sorted_features[j]] = 0

        y_hat = shogp.prediction_featuresubset(x, active_dim)
        
        predic_diff.append(np.abs(y_x - y_hat))

    return np.array(predic_diff).squeeze()


#miles_per_gallon(), stackloos()
datasets = [extramarital_affairs()]#wine_quality(),, extramarital_affairs(), ]  
# done: diabetes(), guerry(), wine_quality() mode_choice
sampleNo_tbx = 30
num_samples = 50
for data in datasets:

    # Loading data
    X, y, db_name, mode = data
    print(db_name)

    excel_path_feature_removal = Path(f'tabular_feature_removal {db_name}.xlsx')
    excel_path_feature_removal_time = Path(f'tabular_feature_removal time {db_name}.xlsx')

    # Split the data into training and test sets
    n, d = X.shape
    remove_feature = int(np.ceil(d * 0.6))

    inducing_points = 200 if n > 200 else n
    shogp = SHOGP(X ,y, inducing_points=200) if n > 200 else SHOGP(X ,y, inte_order=5)

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
        print(f"explaining sample number {i}")
        # SHOGP shapley value
        st = time.time()
        shogp_sv = shogp.local_Shapley_values(x[np.newaxis,:])
        endt = time.time() - st
        shogp_removal_effect = feature_removing_effect(shogp_sv, x, remove_feature, shogp)

        shogp_diff.append(shogp_removal_effect)    
        shogp_time.append(endt)   

        # Kernel SHAP
        try:
            st = time.time()
            shap_values = shap_explainer.shap_values(x, nsamples=num_samples)        
            endt = time.time() - st 
            shap_removal_effect = feature_removing_effect(shap_values, x, remove_feature, shogp)


            kshap_diff.append(shap_removal_effect)    
            kshap_time.append(endt)             
        except:
            pass

        # Sampling SHAP
        try:
            st = time.time()
            shap_sampling_values = shap_sampling_explainer.shap_values(x, nsamples=num_samples)
            endt = time.time() - st
            sshap_removal_effect = feature_removing_effect(shap_sampling_values, x, remove_feature, shogp)


            sshap_diff.append(sshap_removal_effect)    
            sshap_time.append(endt)            
        except:
            pass

        # Bivariate SHAP
        try:
            st = time.time()
            bshap_values = bshap.shap_values(x, nsamples=num_samples)
            endt = time.time() - st
            bishap_removal_effect = feature_removing_effect(bshap_values, x, remove_feature, shogp)

            bishap_diff.append(bishap_removal_effect)    
            bishap_time.append(endt)  
        except:
            pass

        ## Unbiased kernel shap
        try: 
            st = time.time()
            game = games.PredictionGame(imputer, x)
            values = shapley.ShapleyRegression(game, n_samples=num_samples, batch_size=int(num_samples / 10), paired_sampling=True)
            ushap_values = values.values.squeeze()   
            endt = time.time() - st

            ushap_removal_effect = feature_removing_effect(ushap_values, x, remove_feature, shogp)

            ushap_diff.append(ushap_removal_effect)    
            ushap_time.append(endt)         
        except:
            pass

        # LIME 
        try:
            lime_explainer = LimeTabularExplainer(training_data = X_bg, mode = 'regression')
            lime_values = []
            st = time.time()
            explanation = lime_explainer.explain_instance(data_row = x, predict_fn = exp_func, num_features = d)
            endt = time.time() - st
            exp = [0] * d
            for feature_idx, contribution in explanation.local_exp[0]:
                exp[feature_idx] = contribution
            lime_values.append(exp)

            lime_removal_effect = feature_removing_effect(np.array(lime_values[0]), x, remove_feature, shogp)

            lime_diff.append(lime_removal_effect)    
            lime_time.append(endt)      
        except:
            pass

        ## MAPLE
        try:
            st = time.time()
            exp_maple = MAPLE(X, y, X, y)
            mpl_exp = exp_maple.explain(x)
            maple_values = (mpl_exp['coefs'][1:]).squeeze()
            endt = time.time() - st 

            maple_removal_effect = feature_removing_effect(maple_values, x, remove_feature, shogp)

            maple_diff.append(maple_removal_effect)    
            maple_time.append(endt)      
        except:
            pass
        

    all_results = [shogp_diff, kshap_diff, sshap_diff, bishap_diff, ushap_diff, lime_diff, maple_diff]
    time_results = [shogp_time, kshap_time, sshap_time, bishap_time, ushap_time, lime_time, maple_time]
    ## Storing the difference and execution time of the methods
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