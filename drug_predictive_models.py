import pandas as pd
import numpy as np
import sklearn as skl
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from tqdm.notebook import tqdm

#feature_data is a dataframe where each row represents a venous metabolomics sample and each column is one of the high confidence metabolites, there is an additional person_id column named "ID" to merge to the drug data
feature_data = pd.read_csv('...')

#drugs is a file with columns person_id, drug_concept_id that contains all drugs prescribed during pregnancy for each mother
drugs = pd.read_csv('...')

#convert to a binary dataframe indicating if the drug was received for a patient
drugs['indicator'] = 1
drug_outcomes = drugs.pivot(index='person_id', columns='drug_concept_id',values='indicator')
drug_outcomes.fillna(0, inplace=True)

#the following lists are drug_concept_ids (from the OMOP data model) used to identify each drug
drug_outcomes['insulin'] = (drug_outcomes[[40052768,  1361675, 19078552, 19078555, 19078559, 19058398, 46234047, 46234237, 19135276, 46233969,  1516980,  1550023, 19078558]].sum(axis=1) >= 1).astype(int)
drug_outcomes['bupivacaine'] = (drug_outcomes[[35603814, 35604011,   732893, 35603836, 40225949, 35603815]].sum(axis=1) >= 1).astype(int)
drug_outcomes['hydralazine'] = (drug_outcomes[[40174776, 40174765, 40174811, 40174825]].sum(axis=1) >= 1).astype(int)
drug_outcomes['heparin'] = (drug_outcomes[[43011884, 43011850, 43011892,  1718371,  1718370, 43011476,
       43011826, 43011962]].sum(axis=1) >= 1).astype(int)
drug_outcomes['betamethasone'] = (drug_outcomes[[19121383, 40018865]].sum(axis=1) >= 1).astype(int)
drug_outcomes['ampicillin'] = (drug_outcomes[[35605343, 35605342, 19073219, 19079893]].sum(axis=1) >= 1).astype(int)

drug_outcomes = drug_outcomes[['insulin', 'bupivacaine', 'hydralazine', 'heparin', 'betamethasone','ampicillin']]

results = {}

#create a predictive model for each drug using metabolites
for i in tqdm(drug_outcomes.columns):
    #set up data for ML
    y = drug_outcomes[[i]]
    outcome_name = 'drug'+str(i)
    y.columns = [outcome_name]
    y['ID'] = y.index
    ML_data = y.merge(feature_data, how='right', left_on='ID', right_on='ID')
    ML_data.fillna(0, inplace=True)
    results[i] = {'roc_auc':[], 'ap':[], 'pred_y':[], 'actual_y':[], 'params':[]}
    
    #use Stratified 5-fold CV for the outer CV loop
    model = xgb.XGBClassifier(use_label_encoder=False)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
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
        
        model = xgb.XGBClassifier(use_label_encoder=False)
        param_grid = {"learning_rate": [0.01, 0.1], "n_estimators": [25, 50, 75, 100]
                      , 'max_depth': range(2,6,1), 'scale_pos_weight': [1, 3, 5, 10]
                     }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
        
        #use stratified 3-fold CV on the training data for the inner loop
        grid_cv2 = GridSearchCV(model
                                , param_grid
                                , n_jobs=64
                                , cv=cv
                                , scoring="roc_auc"
                                , verbose=True)
        # Fit
        cv_obj = grid_cv2.fit(train_X, train_y)
        
        #use best params to train a model on all training data
        best_params = grid_cv2.cv_results_['params'][np.argmax(grid_cv2.cv_results_['mean_test_score'])]
        model = xgb.XGBClassifier(**best_params, use_label_encoder=False)
        model.fit(train_X, train_y, eval_metric='logloss')
        
        #generate predictions and performance on the test set
        preds = model.predict_proba(test_X)
        results[i]['roc_auc'].append(roc_auc_score(test_y, preds[:,1]))
        results[i]['ap'].append(average_precision_score(test_y, preds[:,1]))
        results[i]['pred_y'].extend(preds[:,1])
        results[i]['actual_y'].extend(test_y)
        results[i]['params'].append(best_params)
        
#compute the mean performance across the test sets
drug = []
auc_mean = []
auc_sd = []
pr_mean = []
pr_sd = []
prevalence = []
pr_curve = []
for i in results.keys():
    result = results[i]
    drug.append(i)
    auc_mean.append(np.mean(result['roc_auc']))
    auc_sd.append(np.std(result['roc_auc']))
    pr_mean.append(np.mean(result['ap']))
    pr_sd.append(np.std(result['ap']))
    prevalence.append(np.mean(result['actual_y']))
    pr_curve.append(precision_recall_curve(results[i]['actual_y'], results[i]['pred_y']))
    
df = pd.DataFrame([drug, auc_mean, auc_sd, pr_mean, pr_sd, prevalence]).T
df.columns=['Drug','AUC','AUC SD','AP','AP SD','Prevalence']
df.to_csv('./modeling_results.csv')