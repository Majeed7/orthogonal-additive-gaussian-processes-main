from sklearn.feature_selection import mutual_info_classif, RFECV, SelectKBest, chi2, mutual_info_regression
from synthesized_datasets import *
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC, SVR
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import numpy as np
from pyHSICLasso import HSICLasso
from pathlib import Path
from datetime import datetime

from SHOGP import SHOGP

results_xsl = Path('fs_synthesized.xlsx')


# Set general font size for readability
plt.rcParams.update({
    'font.size': 12,     # Set default font size
    'font.weight': 'bold',  # Set default font weight to bold
    'axes.labelweight': 'bold',  # Ensure the axis labels are bold
    'axes.titleweight': 'bold',  # Ensure the titles are bold
    'figure.titleweight': 'bold',  # Bold for suptitle if you use fig.suptitle()
    'xtick.labelsize': 12,  # Font size for X-tick labels
    'ytick.labelsize': 12,  # Font size for Y-tick labels
    'xtick.major.size': 5,  # Length of major ticks
    'ytick.major.size': 5,  # Length of major ticks
    'xtick.minor.size': 3,  # Length of minor ticks
    'ytick.minor.size': 3   # Length of minor ticks
})



exp_no = 20

feature_no = 15
importance_mi = np.zeros((exp_no,feature_no))
importance_lasso = np.zeros((exp_no,feature_no))
orders_rfecv = np.zeros((exp_no,feature_no))
importance_k_best = np.zeros((exp_no,feature_no))
importance_ensemble = np.zeros((exp_no,feature_no))
importance_shogp = np.zeros((exp_no,feature_no))
importance_hsiclasso = np.zeros((exp_no,feature_no))


# OK: "XOR" "nonlinear_additive"
# half ok: 'orange_skin'
#not ok "simple additive"  generate_interactions

avg_mi = []
avg_lasso = []
avg_k_best = []
avg_ensemble = []
avg_rfecv = []
avg_shogp = []
avg_hsic = []

datasets = ['nonlinear_additive', 'XOR']#, 'simple_interaction', 'poly_sin']
for ds in datasets:
    for i in range(exp_no):
        X, y, prob, tif, dtype = generate_data(n=300, d=feature_no, datatype=ds)
        #y = prob #> 0.5 # y > np.mean(y)
        mode = 'regression'

        ## SVSVL 
        shogp = SHOGP(X, y, inte_order=5)
        shapley_values_rescaled, shapley_values = shogp.global_shapley_value()
        importance_shogp[i,:] = shapley_values


        ## HSIC lasso 
        hsic_lasso = HSICLasso()
        hsic_lasso.input(X,y)
        hsic_lasso.regression(feature_no, covars=X)
        hsic_ind = hsic_lasso.get_index()
        init_ranks = (len(hsic_ind) + (feature_no - 1/2 - len(hsic_ind))/2) * np.ones((feature_no,))
        init_ranks[hsic_ind] = np.arange(1,len(hsic_ind)+1)
        importance_hsiclasso[i,:] = init_ranks 

        ## Mutual Informmation Importance
        importance_mi[i,:] = mutual_info_classif(X,y) if mode == 'classification' else mutual_info_regression(X,y)
        
        ## Lasso importance
        lasso = Lasso().fit(X, y)
        importance_lasso[i,:] = np.abs(lasso.coef_)

        #Recursive elimination
        estimator = SVC(kernel="linear") if mode == 'classification' else SVR(kernel='linear')
        rfecv = RFECV(estimator, step=1, cv=5)
        rfecv.fit(X, y)
        orders_rfecv[i,:] = rfecv.ranking_

        ## K best
        bestfeatures = SelectKBest(score_func=f_classif, k='all') if mode == 'classification' else SelectKBest(score_func=f_regression, k='all') #F-ANOVA feature selection
        fit = bestfeatures.fit(X,y)
        importance_k_best[i,:] = fit.scores_

        ## Tree ensemble 
        model = ExtraTreesClassifier(n_estimators=50) if mode =='classification' else ExtraTreesRegressor(n_estimators=50)
        model.fit(X,y)
        importance_ensemble[i,:] = model.feature_importances_


    ranking_mi = create_rank(importance_mi)
    ranking_lasso = create_rank(importance_lasso)
    ranking_k_best = create_rank(importance_k_best)
    ranking_ensemble = create_rank(importance_ensemble)
    ranking_rfecv = create_rank(orders_rfecv)
    ranking_shogp = create_rank(np.abs(importance_shogp))
    ranking_hsiclasso = importance_hsiclasso

    avg_mi = (np.mean(ranking_mi[:,tif],axis=1))
    avg_lasso = (np.mean(ranking_lasso[:,tif],axis=1))
    avg_k_best = (np.mean(ranking_k_best[:,tif],axis=1))
    avg_ensemble = (np.mean(ranking_ensemble[:,tif],axis=1))
    avg_rfecv = (np.mean(ranking_rfecv[:,tif],axis=1))
    avg_shogp = (np.mean(ranking_shogp[:,tif],axis=1))
    avg_hsic_lasso = (np.mean(ranking_hsiclasso[:,tif],axis=1))

 
    # Creating dataset
    data = [avg_shogp, avg_hsic_lasso, avg_mi, avg_k_best, avg_rfecv, avg_lasso]
    methods = ["SHOGP", "HSIC-Lasso", "MI", "F-ANOVA", "RFEC", "Lasso"]

    df = pd.DataFrame(data, index=methods)

    mode = 'a' if results_xsl.exists() else 'w'
    with pd.ExcelWriter(results_xsl, engine='openpyxl', mode=mode) as writer:
        df.to_excel(writer, sheet_name=ds + datetime.now().strftime("%Y%m%d_%H%M%S"), index_label='Method')


# print('done')

