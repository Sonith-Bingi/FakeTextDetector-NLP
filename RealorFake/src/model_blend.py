import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from src.config import LGB_PARAMS, SEED, LGB_ROUNDS, LGB_FOLDS

def add_ce_columns(Xf, logits):
    """Adds the cross-encoder predictions as features."""
    X = Xf.copy()
    X["ce_logit"]  = logits
    X["ce_prob"]   = 1 / (1 + np.exp(-logits))
    X["ce_margin"] = np.abs(X["ce_logit"])
    return X

def run_blender_training(Xf_train, Xf_test, oof_logits, ce_test_logits, y_values):
    """Trains the LightGBM blending model using K-Fold CV."""
    
    print("\n--- Starting Blender (LightGBM) Training ---")
    
    X_train_blend = add_ce_columns(Xf_train, oof_logits)
    X_test_blend = add_ce_columns(Xf_test, ce_test_logits)
    
    # Align features just in case
    train_cols = X_train_blend.columns
    X_test_blend = X_test_blend[train_cols]
    
    print(f"Blender training with {len(train_cols)} features.")

    skf = StratifiedKFold(n_splits=LGB_FOLDS, shuffle=True, random_state=SEED)
    oof_b = np.zeros(len(X_train_blend))
    test_b_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_blend, y_values), 1):
        print(f"[LGB] Fold {fold}/{LGB_FOLDS}")
        X_tr, X_va = X_train_blend.iloc[tr_idx], X_train_blend.iloc[va_idx]
        y_tr, y_va = y_values[tr_idx], y_values[va_idx]
        
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va)
        
        callbacks = [
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=500)
        ]
        
        mdl = lgb.train(
            LGB_PARAMS, 
            dtr, 
            num_boost_round=LGB_ROUNDS, 
            valid_sets=[dtr, dva],
            valid_names=["train", "val"],
            callbacks=callbacks
        )
        
        best_iter = mdl.best_iteration if mdl.best_iteration else LGB_ROUNDS
        
        oof_b[va_idx] = mdl.predict(X_va, num_iteration=best_iter)
        test_b = mdl.predict(X_test_blend, num_iteration=best_iter)
        test_b_folds.append(test_b)

    oof_acc = accuracy_score(y_values, (oof_b >= 0.5).astype(int))
    print(f"--- Blender Training Complete ---")
    print(f"Blender OOF accuracy: {oof_acc:.4f}")

    blend_test_prob = np.mean(np.stack(test_b_folds, axis=0), axis=0)
    return blend_test_prob