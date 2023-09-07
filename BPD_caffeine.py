import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn import preprocessing
import shap

def cross_val_opt_predict(X, y):
    """
    A nested-CV method to optimize hyperparameters and make predictions on all samples in X
    Inputs:
        X (nxd): feature df
        y (nx1): outcome df
    NOTE: Assumes X and y are properly paired (i.e. features in X[0,:] correspond to y[0])
    """
    try:
        X = X.values
    except:
        pass
    try:
        y = y.values
    except:
        pass
    estimator = xgb.XGBClassifier(
        objective= 'binary:logistic',
        nthread=2,
        seed=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    pred_list = []
    test_idx_list = []
    best_params = []
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=3)

    #use stratified cross validation for the outer CV loop
    for train_index, test_index in (skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        parameters = {
            'max_depth': range (2, 8, 1),
            'n_estimators': range(50, 210, 25),
            'learning_rate': [0.1, 0.05, 0.01],
            'scale_pos_weight': [1, 5, 10],
            'random_state': [3]
        }

        #use CV within the training data to identify optimal hyperparams
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=parameters,
            scoring = ['roc_auc','average_precision','neg_log_loss'],
            n_jobs = 64,
            cv = 3,
            verbose=False,
            refit='roc_auc'
        )

        grid_search.fit(X_train, y_train)
        
        #apply the best algorithm to the test set
        preds = grid_search.best_estimator_.predict_proba(X_test)[:,1]
        
        #extract best parameters and predictions for the test set
        best_param = grid_search.cv_results_['params'][np.argmax(grid_search.cv_results_['mean_test_roc_auc'])]
        best_params.append(best_param)
        pred_list.extend(preds)
        test_idx_list.extend(test_index)
       
    #use the average of the best parameters for the final model (used for feature importance)
    params_for_final_model = {'learning_rate': np.mean([i['learning_rate'] for i in best_params]),
                             'scale_pos_weight': np.mean([i['scale_pos_weight'] for i in best_params]),
                             'max_depth': round(np.mean([i['max_depth'] for i in best_params])),
                             'n_estimators': round(np.mean([i['n_estimators'] for i in best_params]))}
    final_model = xgb.XGBClassifier(**params_for_final_model, use_label_encoder = False)
    final_model.fit(X, y, eval_metric='logloss')

        
    
    
    return np.array(pred_list)[np.argsort(test_idx_list)], final_model


#this is a file where each row represents a sample from a preterm newborn, the columns contain at least caffeine metabolites and a "BPD" column indicating whether or not the newborn had BPD
df = pd.read_csv('...')

#caffeine metabolites in our dataset of high confidence metabolites
DRUG_METABS = ['Hypoxanthine.pHILIC','Hypoxanthine.nHILIC','Hypoxanthine.pRPLC','Xanthine.nHILIC','Caffeine.pRPLC','X7.Methylxanthine.nRPLC','X5.Acetylamino.6.amino.3.methyluracil.nHILIC','Theobromine.pRPLC','Theophylline.pRPLC','Theophylline.nRPLC']

#train the model and generate predictions for each patient via cross validation
preds, final_model = cross_val_opt_predict(df[DRUG_METABS], df['BPD'])

#compute performance metrics, feature importances, SHAP plots
final_model.get_booster().feature_names = list(df[DRUG_METABS].columns)
score = roc_auc_score(df['BPD'],preds)
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(df[DRUG_METABS])
shap.summary_plot(shap_values, df[DRUG_METABS], show=False)