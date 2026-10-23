#!/usr/bin/env python3
# coding: utf-8

'''
Utility functions and model training for BO KFold CV modeling script
Pauline Bianchi and Samantha Sloane
May 22, 2026
'''

# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

import optuna
from xgboost import XGBRegressor

# Computes Root Mean Squared Error (RMSE) between true and predicted values
def safe_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Removes redundant features, loop through ranked features, keep feature if its not too correlated with already selected features, stop when max_features is reached
def decorrelate_features(ranked_features, corr_matrix, max_features,threshold):
    selected = []
    for feat in ranked_features:
        if feat not in corr_matrix.index:
            continue
        if all(corr_matrix.loc[feat, s] < threshold for s in selected):
            selected.append(feat)
        if len(selected) >= max_features:
            break
    if len(selected) == 0:
        selected = list(ranked_features[:max_features])
    return selected

# Builds a model based on model name and hyperparameters suggested by the Optuna trial
def build_model_from_trial(model_name, trial, random_state=42):
    if model_name == "MLR":
        return LinearRegression()
    if model_name == "Ridge":
        return Ridge(alpha=trial.suggest_float("alpha", 1e-3, 100, log=True))
    if model_name == "Lasso":
        return Lasso(alpha=trial.suggest_float("alpha", 1e-3, 100, log=True), max_iter=50000)
    if model_name == "BRR":
        return BayesianRidge(alpha_1=trial.suggest_float("alpha_1", 1e-8, 1e-2, log=True), alpha_2=trial.suggest_float("alpha_2", 1e-8, 1e-2, log=True), lambda_1=trial.suggest_float("lambda_1", 1e-8, 1e-2, log=True),lambda_2=trial.suggest_float("lambda_2", 1e-8, 1e-2, log=True))
    if model_name == "KRR":
        return KernelRidge(alpha=trial.suggest_float("alpha", 1e-3, 100, log=True), gamma=trial.suggest_float("gamma", 1e-3, 100, log=True), kernel=trial.suggest_categorical("kernel", ["rbf", "laplacian"]))
    if model_name == "SVR":
        return SVR(C=trial.suggest_float("C", 0.01, 10, log=True), epsilon=trial.suggest_float("epsilon", 0.05, 1, log=True), gamma=trial.suggest_categorical("gamma", ["scale", "auto"]), kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]))
    if model_name == "KNN":
        return KNeighborsRegressor(n_neighbors=trial.suggest_int("n_neighbors", 2, 8), weights=trial.suggest_categorical("weights", ["uniform", "distance"]))
    if model_name == "RF":
        return RandomForestRegressor(n_estimators=trial.suggest_int("n_estimators",20, 200), max_depth=trial.suggest_int("max_depth",1,5), min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 8), min_samples_split=trial.suggest_int("min_samples_split", 4, 12), max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),random_state=42, n_jobs=1)
    if model_name == "XGB":
        return XGBRegressor(objective="reg:squarederror",n_estimators=trial.suggest_int("n_estimators",20,200),max_depth=trial.suggest_int("max_depth",1,5),learning_rate=trial.suggest_float("learning_rate",0.01,0.2,log=True),subsample=trial.suggest_float("subsample",0.6,1.0),colsample_bytree=trial.suggest_float("colsample_bytree",0.6,1.0),reg_alpha=trial.suggest_float("reg_alpha",1e-4,10,log=True),reg_lambda=trial.suggest_float("reg_lambda",1e-2,100,log=True),min_child_weight=trial.suggest_float("min_child_weight",1,20),random_state=42,n_jobs=1)
    if model_name == "NN":
        return MLPRegressor(hidden_layer_sizes=trial.suggest_categorical("hidden_layer_sizes",[(5,),(10,),(5, 5),(10, 5)]),activation=trial.suggest_categorical("activation",["relu", "tanh"]),alpha=trial.suggest_float("alpha",1e-3,1,log=True),learning_rate_init=trial.suggest_float("learning_rate_init",1e-4,1e-2,log=True),early_stopping=True,validation_fraction=0.2,n_iter_no_change=30,max_iter=3000,random_state=42)
    raise ValueError(f"Unknown model: {model_name}")

# Builds a model based on the model name and hyperparameters from the best trial of the Optuna study
def build_final_model_from_params(model_name, params):
    params = remove_feature_selection_params(dict(params))
    if model_name == "MLR":
        return LinearRegression()
    if model_name == "Ridge":
        return Ridge(**params)
    if model_name == "Lasso":
        return Lasso(**params,max_iter=50000)
    if model_name == "BRR":
        return BayesianRidge(**params)
    if model_name == "KRR":
        return KernelRidge(**params)
    if model_name == "SVR":
        return SVR(**params)
    if model_name == "KNN":
        return KNeighborsRegressor(**params)
    if model_name == "RF":
        return RandomForestRegressor(**params,random_state=42,n_jobs=1)
    if model_name == "XGB":
        return XGBRegressor(**params,objective="reg:squarederror",random_state=42,n_jobs=1)
    if model_name == "NN":
        return MLPRegressor(**params,early_stopping=True,validation_fraction=0.2,n_iter_no_change=30,max_iter=3000,random_state=42)
    raise ValueError(f"Unknown model: {model_name}")

# Feature selection using either selectkbest, lasso, random forest, or RFE (recursive feature elimination)
def build_feature_selector_from_trial(trial, max_features_cap, random_state=42):
    selector_name = trial.suggest_categorical("feature_selector", ["selectkbest", "lasso_feature", "rf_feature", "rfe_feature"])
    
    # Selectkbest (ranks each feature independently using F-statistic, measuring the strength of the linear relationship between feature and ln(r), Optuna chooses how many top ranked feautres (k) to keep)
    if selector_name == "selectkbest":
        k = trial.suggest_int("selectkbest_k", 1, max_features_cap)
        selector = SelectKBest(score_func=f_regression, k=k)

    # If selectkbest is not chosen by Optuna, Lasso used (fits a lasso regression model and removes features whos coefficients ≤0 or chosen threshold, threshold and regulariation is chosen by Optuna)
    elif selector_name == "lasso_feature":
        alpha = trial.suggest_float("alpha_feature", 1e-3, 100, log=True)
        threshold = trial.suggest_categorical("lasso_threshold_feature", ["mean", "median"])
        selector = SelectFromModel(estimator=Lasso(alpha=alpha, max_iter=50000), threshold=threshold, max_features=max_features_cap)

    # If selectkbest is not chosen by Optuna, Random forest used (fits a random forest model and removes features according to feature importance values, threshold/forest size/tree depth are chosen by Optuna)
    elif selector_name == "rf_feature":
        threshold = trial.suggest_categorical("rf_threshold_feature",["mean", "median"])
        n_estimators = trial.suggest_int("rf_n_estimators_feature", 20, 200)
        max_depth = trial.suggest_int("rf_max_depth_feature", 1, 5)
        selector = SelectFromModel(estimator=RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=1), threshold=threshold, max_features=max_features_cap)

    # If selectkbest is not chosen by Optuna, Recursive feature elimination (RFE) used (fits a ridge model, ranks coefficients, removes least important features, refit model, rank coefficients, etc. until desired number of features/Ridge alpha reached, which were chosen by Optuna) 
    elif selector_name == "rfe_feature":
        rfe_n_features = trial.suggest_int("rfe_n_features", 1, max_features_cap)
        alpha = trial.suggest_float("rfe_alpha", 1e-3, 100, log=True)
        selector = RFE(estimator=Ridge(alpha=alpha), n_features_to_select=rfe_n_features)

    else:
        raise ValueError(f"Unknown feature selector: {selector_name}")

    return selector, selector_name

# Removes unwanted features from the above feature selection methods
def get_selected_feature_names(fitted_pipe, feature_names):
    selector = fitted_pipe.named_steps["selector"]

    if hasattr(selector, "get_support"):
        mask = selector.get_support()
        return list(pd.Index(feature_names)[mask])

    return list(feature_names)

# Cleans up the hyperparameters so only features are seen for modeling and new hyperparameters can be tuned for modeling
def remove_feature_selection_params(params):
    fs_params = {
        "feature_selector",
        "selectkbest_k",
        "alpha_feature",
        "lasso_threshold_feature",
        "rf_threshold_feature",
        "rf_n_estimators_feature",
        "rf_max_depth_feature",
        "rfe_n_features",
        "rfe_alpha"
    }
    return {k: v for k, v in params.items() if k not in fs_params}

#THIS IS WHERE THE MODEL TRAINING STARTS
def train_outer_fold(model_name, X_train_outer, y_train_outer, X_test_outer, y_test_outer, inner_cv_splits, inner_cv_repeats, max_features_cap, corr_threshold, n_trials, overfit_penalty_weight, random_state,):

    inner_cv = RepeatedKFold(n_splits=inner_cv_splits, n_repeats=inner_cv_repeats, random_state=random_state)

    def objective(trial):
        selector, selector_name = build_feature_selector_from_trial(trial, max_features_cap=max_features_cap, random_state=random_state)

        model = build_model_from_trial(model_name, trial)

        pipe = Pipeline([("scaler", StandardScaler()), ("selector", selector), ("model", model)])

        train_scores = []
        val_scores = []

        for train_i, val_i in inner_cv.split(X_train_outer):
            X_tr = X_train_outer.iloc[train_i]
            X_val = X_train_outer.iloc[val_i]

            y_tr = y_train_outer.iloc[train_i]
            y_val = y_train_outer.iloc[val_i]

            try:
                pipe.fit(X_tr, y_tr)

                train_pred = pipe.predict(X_tr)
                val_pred = pipe.predict(X_val)

                train_scores.append(r2_score(y_tr, train_pred))
                val_scores.append(r2_score(y_val, val_pred))

            except Exception:
                return -1e9

        mean_train = np.nanmean(train_scores)
        mean_val = np.nanmean(val_scores)

        if np.isnan(mean_val):
            return -1e9

        overfit_penalty = max(0, mean_train - mean_val)
        return mean_val - overfit_penalty_weight * overfit_penalty

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_selector_name = best_params["feature_selector"]

    fixed_trial = optuna.trial.FixedTrial(best_params)

    final_selector, _ = build_feature_selector_from_trial(fixed_trial, max_features_cap=max_features_cap, random_state=random_state)

    final_model = build_final_model_from_params(model_name, best_params)

    final_pipe = Pipeline([("scaler", StandardScaler()), ("selector", final_selector), ("model", final_model)])

    final_pipe.fit(X_train_outer, y_train_outer)

    selected_features = get_selected_feature_names(final_pipe, X_train_outer.columns)

    pred_train = final_pipe.predict(X_train_outer)
    pred_test = final_pipe.predict(X_test_outer)

    r2_train = r2_score(y_train_outer, pred_train)
    rmse_train = safe_rmse(y_train_outer, pred_train)
    mae_train = mean_absolute_error(y_train_outer, pred_train)

    r2_test = r2_score(y_test_outer, pred_test)
    rmse_test = safe_rmse(y_test_outer, pred_test)
    mae_test = mean_absolute_error(y_test_outer, pred_test)

    return {
        "model": final_pipe,
        "best_params": best_params,
        "feature_selection_method": best_selector_name,
        "selected_features": selected_features,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "metrics": {
            "r2_train": r2_train,
            "rmse_train": rmse_train,
            "mae_train": mae_train,
            "r2_test": r2_test,
            "rmse_test": rmse_test,
            "mae_test": mae_test,}}