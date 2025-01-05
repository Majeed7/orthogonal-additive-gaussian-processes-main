import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize, scale
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.svm import LinearSVC, LinearSVR
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, RFECV, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.base import clone
from tabular_datasets import *

from SHOGP import SHOGP
from pyHSICLasso import HSICLasso

#regression [ diabetes(), guerry() ]
#extramarital_affairs(), mode_choice(), diabetes(), wine_quality(), querry()

#classification [breast_cancer(), ]
# heart_mortality statlog_heart()

X, y, ds_name, mode = guerry()
n, d = X.shape
print(f'the size of the dataset is { X.shape[0] } times { X.shape[1] }')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
base_classifier = RandomForestRegressor(n_estimators=1000, random_state=42) if mode == 'regression' else RandomForestClassifier(n_estimators=500, random_state=42)

# Initialize feature selection methods for classification
selectors = {
    'ANOVA F-value': SelectKBest(score_func=f_classif, k='all'),
    'Mutual Information': SelectKBest(score_func=mutual_info_classif, k='all'), 
    'LassoCV': LassoCV(),
    'Random Forest': ExtraTreesClassifier(n_estimators=50, random_state=42),
    'RFECV': RFECV(estimator=clone(LinearSVC()), step=1, cv=5),
}

## For regression it is different
if mode == 'regression':
    selectors = {
        'SHOGP': SHOGP,
        'HSICLasso': HSICLasso(),
        'ANOVA F-value': SelectKBest(score_func=f_regression, k='all'),
        'Mutual Information': SelectKBest(score_func=mutual_info_regression, k='all'), 
        'LassoCV': LassoCV(),
        'Random Forest': ExtraTreesRegressor(n_estimators=50, random_state=42),
        'RFECV': RFECV(estimator=clone(LinearSVR()), step=1, cv=5),
    }


# Dictionary to store results
results = {key: {'Performance metric': [], 'Feature Indices': []} for key in selectors.keys()}

# Fit feature selectors and evaluate incrementally
for name, selector in selectors.items():
    print(f"Feature selection with {name}")
    if name in ['LassoCV', 'Random Forest', 'RFECV', 'SHOGP', 'HSICLasso']:
        if name == 'SHOGP':
            shogp = selector(X, y)#, inte_order=5, inducing_points=200)
            importances = np.abs(shogp.global_shapley_value()[0])

        elif name == "HSICLasso":
            hsic_lasso = HSICLasso()
            hsic_lasso.input(X,y)
            hsic_lasso.regression(d, covars=X)
            hsic_ind = hsic_lasso.get_index()
            init_ranks = (len(hsic_ind) + (d - 1/2 - len(hsic_ind))/2) * np.ones((d,))
            init_ranks[hsic_ind] = np.arange(1,len(hsic_ind)+1)
            importances = d - init_ranks 

        else:    
            selector.fit(X_train, y_train)
                
        if name == 'LassoCV':
            importances = np.abs(selector.coef_)
        elif name == 'Random Forest':
            importances = selector.feature_importances_
        elif name == 'RFECV':
            importances = selector.ranking_
            # Reverse the RFECV ranking so that lower means more important
            importances = max(importances) + 1 - importances

        # Get indices of features sorted by importance or ranking
        indices = np.argsort(importances)[::-1]
        for i in range(1, len(indices) + 1):
            selected_features = indices[:i]
            model = clone(base_classifier)
            model.fit(X_train[:, selected_features], y_train)
            y_pred = model.predict(X_test[:, selected_features])
            performance_met = mean_absolute_error(y_test, y_pred) if mode =='regression' else 1 - accuracy_score(y_test, y_pred)
            results[name]['Performance metric'].append(performance_met)
            results[name]['Feature Indices'].append(selected_features[i-1]) #append(', '.join(map(str, selected_features)))
    else:
        # For SelectKBest, incrementally select features
        selector.fit(X_train, y_train)
        scores = selector.scores_
        indices = np.argsort(scores)[::-1]
        for i in range(1, len(indices) + 1):
            selected_features = indices[:i]
            model = clone(base_classifier)
            model.fit(X_train[:, selected_features], y_train)
            y_pred = model.predict(X_test[:, selected_features])
            performance_met = mean_absolute_error(y_test, y_pred) if mode =='regression' else 1 - accuracy_score(y_test, y_pred)
            results[name]['Performance metric'].append(performance_met)
            results[name]['Feature Indices'].append(selected_features[i-1]) #append(', '.join(map(str, selected_features)))

# Plotting all results in one figure
plt.figure(figsize=(12, 8))
for name, data in results.items():
    plt.plot(range(1, len(data['Performance metric']) + 1), data['Performance metric'], marker='o', linestyle='-', label=f'{name}')
plt.title('Model Accuracy vs. Number of Features Selected')
plt.xlabel('Number of Features')
plt.ylabel(f'Error')
if mode == 'regression': plt.ylabel("Absolute Deviation")
plt.legend()
plt.grid(True)
plt.show()

# Convert results to DataFrame and save to Excel
results_df = pd.DataFrame.from_dict({(i,j): results[i][j] 
                           for i in results.keys() 
                           for j in results[i].keys()},
                       orient='index')
results_df.to_excel(f'results/{ds_name}.xlsx')


print("done!")
