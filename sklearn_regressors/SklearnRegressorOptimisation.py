import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import (
    Dict,
    Literal
)
from numpy.typing import ArrayLike
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV


def inner_r2_scores(
    kf: KFold, 
    X: ArrayLike, 
    y: ArrayLike, 
    model,
):
    '''
    Replicate inner K-folds from search for extraction of R2 scores. 
    '''
    trn_r2 = 0
    val_r2 = 0
    kf.get_n_splits(X)
    for k, (trn_idx, val_idx) in enumerate(kf.split(X)):
        X_trn, X_val = X[trn_idx], X[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]
        model.fit(X_trn, y_trn)
        trn_pred = model.predict(X_trn)
        trn_r2 += r2_score(y_trn, trn_pred)
        val_pred = model.predict(X_val)
        val_r2 += r2_score(y_val, val_pred)
    return trn_r2/kf.n_splits, val_r2/kf.n_splits

def hyperparam_opt(
    kf: KFold, 
    X: ArrayLike,
    y: ArrayLike,
    model, 
    search_space: Dict,
    search_method: Literal["Grid", "Bayesian"],
):
    ''' 
    Perform exhaustive grid search on given model. 
    '''
    if search_method == "Grid":
        search = GridSearchCV(
            model, 
            search_space, 
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            cv=kf,
            verbose=1,
            return_train_score=True
        )
    elif search_method == "Bayesian":
        optimizer_kwargs = {
            "acq_func": "EI",
            "random_state": 8,
            "n_jobs":-1,
            "acq_func_kwargs": {
                "xi": 0.01
            }
        }
        search = BayesSearchCV(
            model, 
            search_space, 
            optimizer_kwargs=optimizer_kwargs,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            n_iter=75,
            n_points=5,
            cv=kf,
            verbose=0,
            return_train_score=True
        )
    _ = search.fit(X, y)
    search_res_dict = search.cv_results_
    # determine best hyperparam set
    best_idx = np.where(search_res_dict["rank_test_score"] == 1)[0][0]
    best_set = search_res_dict["params"][best_idx]
    # extract scores 
    trn_mae = search_res_dict["mean_train_score"][best_idx]
    val_mae = search_res_dict["mean_test_score"][best_idx]
    model.set_params(**best_set)
    trn_r2, val_r2 = inner_r2_scores(kf, X, y, model)
    return best_set, trn_r2, trn_mae, val_r2, val_mae 

def holdout_CV(    
    kf_inner: KFold,
    X: ArrayLike,
    y: ArrayLike,
    model: RegressorMixin,
    search_space: Dict,
    search_method: Literal["Grid", "Bayesian"],
    return_predictions=False,
    random_state=0
):
    '''
    Holdout cross-validation to estimate test scores/model generalisability after optimisation
    with standard cross-validation.
    '''
    result_dict = {
        "hyperparams": [],
        "trn_mae": [], "trn_r2": [],
        "val_mae": [], "val_r2": [],
        "tst_mae": [], "tst_r2": []
    }
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=random_state)
    # perform CV optimisation on training data
    best_set, trn_r2, trn_mae, val_r2, val_mae = hyperparam_opt(
        kf_inner,
        X_trn,
        y_trn,
        model,
        search_space,
        search_method,
    )
    result_dict["hyperparams"].append(best_set)
    result_dict["trn_mae"].append(trn_mae)
    result_dict["trn_r2"].append(trn_r2)
    result_dict["val_mae"].append(val_mae)
    result_dict["val_r2"].append(val_r2)
    # test with best parameter set
    model.set_params(**best_set)
    model.fit(X_trn, y_trn)
    tst_pred = model.predict(X_tst)
    tst_mae = - mean_absolute_error(y_tst, tst_pred)
    tst_r2 = r2_score(y_tst, tst_pred)
    result_dict["tst_mae"].append(tst_mae)
    result_dict["tst_r2"].append(tst_r2)
    if return_predictions:
        return result_dict, (tst_pred, y_tst)
    else:
        return result_dict


def final_opt(
    kf: KFold, 
    X: ArrayLike,
    y: ArrayLike,
    model, 
    search_space: Dict,
    search_method: Literal["Grid", "Bayesian"],
):
    ''' 
    Final optimisation using cross-validation to produce a final model for use.
    '''
    best_set, trn_r2, trn_mae, val_r2, val_mae = hyperparam_opt(
        kf,
        X,
        y,
        model,
        search_space,
        search_method,
        )
    model.set_params(**best_set)
    model.fit(X, y)
    result_dict = {
        "hyperparams": best_set,
        "trn_mae": trn_mae,
        "trn_r2": trn_r2,
        "val_mae": val_mae,
        "val_r2": val_r2
    }
    return model, result_dict