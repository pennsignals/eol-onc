#!/usr/bin/env python
# coding: utf-8

# # Conversation-Connect
# ### Identifying patients for Serious Illness Conversations
# 
# > Corey Chivers, PhD <corey.chivers@pennmedicine.upenn.edu> <br>
# > Copyright (c) 2019 University of Pennsylvania Health System, MIT License
# 
# Predict risk of 6 month mortality for a general population of Penn Medicine Oncology adult patients to improve access to advanced care planning for those most likely to benefit.
  
import argparse
import numpy as np
import pandas as pd
import pickle
from pprint import pprint
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

from load_data import load_train_test


def main(args):
    N_ITER = args.n_iter
    K_CV = args.k_cv

    print('loading data')
    label_name = 'label'
    train, test = load_train_test(f=args.infile, label_name=label_name)

    train=train.head(1000).copy()

    print('loaded', train.shape[0], test.shape[0])
    print('label rate', train[label_name].mean(), test[label_name].mean())

    num_cols = [c for c in train if ': ' in c]
    dx_cols = [c for c in train if c.startswith('n_')]
    cat_cols = [c for c in train if '>>>' in c]

    feature_cols = dx_cols + num_cols + cat_cols + ['PAT_AGE', 'SEX_C']

    in_vars = feature_cols
    print('Using', len(in_vars), 'vars')

    if args.model_type == 'rf':
        rf = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)

        # Look at parameters used by our current forest
        print('Starting parameters currently in use:\n')
        pprint(rf.get_params())

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=50, stop=1000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=10)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf}
        pprint(random_grid)

        # Use the random grid to search for best hyperparameters
        # Random search of parameters, using k fold cross validation,
        # search across n_iter different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, scoring='roc_auc',
                                       n_iter=N_ITER, cv=K_CV, verbose=2, random_state=42, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(train[in_vars], train['label'])

        # Save Model
        with open('models/rf_random_search.p', 'wb') as f:
            pickle.dump(rf_random, f, pickle.HIGHEST_PROTOCOL)
        with open('models/rf_args.p', 'wb') as f:
            pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

    if args.model_type == 'gb':
        gb = GradientBoostingClassifier(verbose=1, subsample=0.9, random_state=42, n_iter_no_change=5)

        print('Parameters currently in use:\n')
        pprint(gb.get_params())


        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num = 10)]
        max_depth.append(None)
        min_samples_leaf = [1, 2, 4]
        min_samples_split = [2, 5, 10]
        n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 10)]
        subsample = [0.5, 0.8, 1.0]
        loss = ['deviance', 'exponential']


        random_grid = {'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'n_estimators': n_estimators,
                       'subsample': subsample,
                       'loss': loss}
        pprint(random_grid)
        gb_random = RandomizedSearchCV(estimator = gb, param_distributions = random_grid, scoring='roc_auc',
                                       n_iter = N_ITER, cv = K_CV, verbose=2, random_state=42, n_jobs = -1)
        # Fit the random search model
        gb_random.fit(train[in_vars], train['label'])
        with open('models/gb_random_search.p', 'wb') as f:
            pickle.dump(gb_random, f, pickle.HIGHEST_PROTOCOL)
        with open('models/gb_args.p', 'wb') as f:
            pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--n-iter', help='number of random search iterations', required=False, default=3)
    parser.add_argument('-k','--k-cv', help='K folds for cross validation', required=False, default=5)
    parser.add_argument('-f', '--infile', help='Path to input file', required=True)
    parser.add_argument('-m','--model-type', help='Model to fit {"rf", "gb"}', required=False, default='rf')
    args = parser.parse_args()
    main(args)
