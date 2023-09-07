import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, cross_val_predict, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, r2_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind
import seaborn as sns
import shap

def cross_val(X, y):
    """
    A nested-CV method to optimize hyperparameters and make predictions on all samples in X
    Inputs:
        X (nxd): feature df
        y (nx1): outcome df
    NOTE: Assumes X and y are properly paired (i.e. features in X[0,:] correspond to y[0])
    """
    X = X.values
    y = y.values
    estimator = xgb.XGBRegressor(
        objective= 'reg:squarederror',
        nthread=2,
        seed=42,
        use_label_encoder=False,
        eval_metric='rmse'
    )
    
    pred_list = []
    test_idx_list = []
    r2s = []
    mses = []
    param_list = []
    
    skf = KFold(n_splits=3, shuffle=True, random_state=17)
    
    for train_index, test_index in tqdm(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]        
    
        parameters = {
            'max_depth': range (2, 9, 1),
            'n_estimators': range(50, 210, 25),
            'learning_rate': [0.1, 0.05, 0.01],
            'colsample_bytree': [0.5, 0.75, 1],
            'random_state': [3]
        }
        
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=parameters,
            scoring = 'neg_mean_squared_error',
            n_jobs = 64,
            cv = 3,
            verbose=False,
            refit='neg_mean_squared_error'
        )

        grid_search.fit(X_train, y_train)
        
        preds = grid_search.best_estimator_.predict(X_test)
        pred_list.extend(preds)
        test_idx_list.extend(test_index)
        param_list.append(grid_search.best_params_)
        r2s.append(r2_score(y_test, preds))
        mses.append(np.mean((y_test-preds)**2))

    #predictions in the same order as input X, y
    ordered_preds = np.array(pred_list)[np.argsort(test_idx_list)]
    
    return np.mean(r2s), np.std(r2s), ordered_preds, mses, param_list

#this is a file where each row is a venous sample, each column is a metabolite measurement for each of the 874 high confidence metabolites, and an additional column for the GA of each newborn; some newborns are excluded as GA is not known
df = pd.read_csv('...')

r2, r2sd, preds, mses, param_list = cross_val(df.drop(['gest_age'], axis=1), df['gest_age'])

estimator = xgb.XGBRegressor(
    objective= 'reg:squarederror',
    nthread=2,
    seed=42,
    use_label_encoder=False,
    eval_metric='rmse',
    colsample_bytree = np.mean([i['colsample_bytree'] for i in param_list]),
    learning_rate = np.mean([i['learning_rate'] for i in param_list]),
    max_depth = np.round(np.mean([i['max_depth'] for i in param_list])).astype(int),
    n_estimators = np.round(np.mean([i['n_estimators'] for i in param_list])).astype(int),
    random_state = 3
)

estimator.fit(df.drop(['gest_age'], axis=1), df['gest_age'])

#feature importance
top_features = pd.DataFrame([estimator.feature_importances_, df.columns[:-1]]).T
top_features.columns=['importance','name']
top_features = top_features.sort_values('importance', ascending=False)

#investigate effect of drugs
#this is a file where the columns are binary indicators for whether or not a particular drug was used during the last 14 days of pregnancy, the rows represent samples and can be linked to the above dataframe with metabolomic and GA data
meds = pd.read_csv('...')

ddf = df.loc[:,['maternal_id','gest_age']]
ddf.loc[:,'pred'] = preds

on_off = []
errors = []
drugs = []
for i in meds.columns:
    if i != 'person_id':
        print(i)
        
        #merge the medications to the dataframe with predictions and actual GA
        med_ga_df = ddf.merge(meds[['person_id',i]], how='left', left_on='maternal_id', right_on='person_id')
        med_ga_df = med_ga_df.drop('person_id',axis=1).fillna(0)
        
        #split into patients who received the drug and those who did not
        on_drug = med_ga_df[med_ga_df[i] == 1]
        off_drug = med_ga_df[med_ga_df[i] == 0]
        
        #compute errors within each group
        on_drug_errors = (on_drug['pred']-on_drug['gest_age'])/on_drug['gest_age']
        off_drug_errors = (off_drug['pred']-off_drug['gest_age'])/off_drug['gest_age']
        print(100*np.mean(on_drug_errors), 100*np.mean(off_drug_errors))
        print(ttest_ind(on_drug_errors, off_drug_errors))
        print()
        
        #compile data for downstream processing
        drugs.extend([i]*len(on_drug_errors))
        drugs.extend([i]*len(off_drug_errors))
        errors.extend(on_drug_errors*100)
        errors.extend(off_drug_errors*100)
        on_off.extend(['+']*len(on_drug_errors))
        on_off.extend(['-']*len(off_drug_errors))
    
    
#make plots
for i in meds.columns:
    if i != 'person_id':
        graph_df = pd.DataFrame([drugs,errors,on_off]).T
        graph_df.columns = ['Drug','Error (%)','Taken?']
        graph_df = graph_df[graph_df['Drug'].isin([i])]
        sns.boxplot(x=graph_df['Drug'], y=graph_df['Error (%)'], hue=graph_df['Taken?'])
        fig = plt.gcf()
        fig.set_size_inches(5, 10)
        plt.xlabel('')
