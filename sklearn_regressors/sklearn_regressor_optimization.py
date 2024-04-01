'''
Script for running Scikit-learn model optimization.
'''

import json
import numpy as np
import pickle

from argparse import ArgumentParser
from pathlib import Path
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from skopt.space import Real, Categorical, Integer

import sklearn_regressors.SklearnRegressorOptimization as SklOpt

# correction for compatibility between Skopt and NumPy.
np.int = int

# search spaces
# Random Forest
rf_space = {
    'n_estimators': Integer(50, 1000),
    'max_depth': Integer(1, 10),
    'max_features': Categorical(["sqrt", "log2", None]),
    'min_samples_split': Real(1e-4, 1 - 1e-4),
    'min_samples_leaf': Real(1e-4, 1 - 1e-4),
    'n_jobs': Categorical([-1]),
}

# Gradient Boosted Trees
gbt_space = {
    'n_estimators': Integer(50, 1000),
    'max_depth': Integer(1, 10),
    'max_features': Categorical(["sqrt", "log2", None]),
    'min_samples_split': Real(1e-4, 1 - 1e-4),
    'min_samples_leaf': Real(1e-4, 1 - 1e-4),
    'learning_rate': Real(1e-4, 5e-1),
    'subsample': Real(1e-4, 1 - 1e-4),
}

# Gaussian Process
gp_space = {
    'alpha': Real(1e-4, 5e-1)
}

# Support Vector Machine
svm_space = {
    'epsilon': Real(1e-4, 5e-1),
    'C': Real(1e-4, 100)
}

# model mappings
model_dict = {
    "rf": (RandomForestRegressor, rf_space, "Bayesian"),
    "gbt": (GradientBoostingRegressor, gbt_space, "Bayesian"),
    "gp": (GaussianProcessRegressor, gp_space, "Bayesian"),
    "svm": (SVR, svm_space, "Bayesian"),
}

# data processing
class MinMaxScaler:

    def __init__(self, y):
        self.min_y = np.min(y)
        self.max_y = np.max(y)
    
    def transform(self, y):
        y_scaled = (y - self.min_y)/(self.max_y - self.min_y)
        return y_scaled

    def invert(self, y):
        y_orig = y*(self.max_y - self.min_y) + self.min_y
        return y_orig

# argument parser
def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--X', type=Path, required=True)
    parser.add_argument('--y', type=Path, required=True)    
    parser.add_argument('--X_test', type=Path, required=False, default=None)    
    parser.add_argument('--y_test', type=Path, required=False, default=None)
    parser.add_argument('--model_name',  type=str, required=True)
    parser.add_argument('--enc_name', type=str, required=True)
    parser.add_argument('--save_dir', type=Path, required=True)
    parser.add_argument('--holdout_seed', type=int, required=False, default=0)
    parser.set_defaults(cv=True)
    return parser

# run optmisation
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    X_trn = np.load(args.X)
    y_trn_unscaled = np.load(args.y)
    scaler = MinMaxScaler(y_trn_unscaled)
    y_trn = scaler.transform(y_trn_unscaled)

    kf_inner = KFold(n_splits=3, random_state=8, shuffle=True)

    model, search_space, search_method = model_dict[args.model_name]

    # holdout for test scores
    cv_res, preds = SklOpt.holdout_CV(
        kf_inner=kf_inner,
        X=X_trn,
        y=y_trn,
        model=model(),
        search_space=search_space,
        search_method=search_method,
        return_predictions=True,
        random_state=args.holdout_seed
    )
    json.dump(cv_res, open(f"{args.save_dir}/CV_results.json", 'w'))
    tst_pred, y_tst = preds
    np.save(f"{args.save_dir}/test_predictions.npy", tst_pred)
    np.save(f"{args.save_dir}/test_actual.npy", y_tst)

    # cv for optimisation
    final_model, final_res = SklOpt.final_opt(
        kf=kf_inner, 
        X=X_trn, 
        y=y_trn,
        model=model(),
        search_space=search_space,
        search_method=search_method,
    )
    json.dump(final_res, open(f"{args.save_dir}/final_CV_results.json", 'w'))
    pickle.dump(final_model, open(f"{args.save_dir}/final_model.sav", 'wb'))

    if args.X_test:
        # run final model on true test dataset
        X_tst = np.load(args.X_test)
        y_tst_unscaled = np.load(args.y_test)
        y_tst = scaler.transform(y_tst_unscaled)

        tst_pred = final_model.predict(X_tst)
        tst_mae = - mean_absolute_error(y_tst, tst_pred)
        tst_r2 = r2_score(y_tst, tst_pred)
        res = {
            "model": args.model_name,
            "encoding": args.enc_name,
            "tst_r2": tst_r2,
            "tst_mae": tst_mae,
        }
        np.save(f"{args.save_dir}/gap_pred.npy", np.array(tst_pred))
        json.dump(res, open(f"{args.save_dir}/test_results.json", 'w'))