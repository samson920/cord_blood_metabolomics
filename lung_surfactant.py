import pandas as pd
import numpy as np
import sklearn as skl
import xgboost as xgb
import re
from collections.abc import Iterable
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import os
from scipy.stats import ttest_ind

#the following is a file with rows for venous samples, columns for metabolite values including an ID column and a binary column indicating if the newborn received lung surfactant
ML_data = pd.read_csv('...')

#this is a list of babies born preterm
preterm_ids = pd.read_csv('...')

results = {}
for i in tqdm(['surfactant']):
    outcome_name = str(i)
    ML_data.fillna(0, inplace=True)
    results[i] = {'roc_auc':[], 'ap':[], 'pred_y':[], 'actual_y':[], 'test_ids': []}
    model = xgb.XGBClassifier(use_label_encoder=False)
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    kfold_indexer = ML_data[[outcome_name,'ID']].drop_duplicates()
    for train_index, test_index in tqdm(kfold.split(kfold_indexer['ID'], kfold_indexer[outcome_name])):
        test_ids = kfold_indexer.iloc[test_index,:]['ID']
        train_ids = kfold_indexer.iloc[train_index,:]['ID']
        train_data = ML_data[ML_data['ID'].isin(train_ids)]
        test_data = ML_data[ML_data['ID'].isin(test_ids)]
        train_X = np.matrix(train_data.loc[:,~train_data.columns.isin([outcome_name,'ID'])])
        test_X = np.matrix(test_data.loc[:,~test_data.columns.isin([outcome_name,'ID'])])
        train_y = np.array(train_data[outcome_name])
        test_y = np.array(test_data[outcome_name])

        model = xgb.XGBClassifier(use_label_encoder=False, nthread=8)
        param_grid = {"learning_rate": [0.01, 0.1], "n_estimators": [10, 25, 50, 75, 100, 150, 200]
                      , 'max_depth': range(2,5,1), 'scale_pos_weight': [1, 3, 5, 10, 15, 20]
                     }
        #instantiate the Grid Search:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
        grid_cv2 = GridSearchCV(model
                                , param_grid
                                , n_jobs=1
                                , cv=cv
                                , scoring="roc_auc"
                                , verbose=True)
        # Fit
        train_weights = [1 if (i in preterm_ids.values) else 1 for i in train_data['ID'].values]
        cv_obj = grid_cv2.fit(train_X, train_y, sample_weight=train_weights)
        
        best_params = grid_cv2.cv_results_['params'][np.argmax(grid_cv2.cv_results_['mean_test_score'])]
        print(best_params)
        print(np.max(grid_cv2.cv_results_['mean_test_score']))
        model = xgb.XGBClassifier(**best_params, use_label_encoder=False)
        model.fit(train_X, train_y, sample_weight=train_weights)
        
        
        preds = model.predict_proba(test_X)
        results[i]['roc_auc'].append(roc_auc_score(test_y, preds[:,1]))
        results[i]['ap'].append(average_precision_score(test_y, preds[:,1]))
        results[i]['test_ids'].extend(test_data['ID'])
        results[i]['pred_y'].extend(preds[:,1])
        results[i]['actual_y'].extend(test_y)

# these are the average of the best params across the three CV iterations        
final_model = xgb.XGBClassifier(use_label_encoder = False,
                                learning_rate=0.07,
                               max_depth=2,
                               n_estimators=133,
                               scale_pos_weight=22/3)
train_X = np.matrix(ML_data.loc[:,~ML_data.columns.isin([outcome_name,'ID'])])
train_y = np.array(ML_data[outcome_name])
final_model.fit(train_X, train_y, eval_metric='logloss')
feature_labels = ML_data.columns[:-2]
final_model.get_booster().feature_names = list(feature_labels)

top_features = pd.DataFrame([final_model.feature_importances_, feature_labels]).T
top_features.columns=['importance','name']
top_features = top_features.sort_values('importance', ascending=False)

preds_df = pd.DataFrame([results['surfactant']['pred_y'], results['surfactant']['actual_y'], results['surfactant']['test_ids']]).T
preds_df.columns = ['pred_y', 'surfactant', 'ids']
roc_auc_score(preds_df[preds_df['ids'].isin(preterm_ids)]['surfactant'], preds_df[preds_df['ids'].isin(preterm_ids)]['pred_y'])
