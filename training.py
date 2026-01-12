import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import r2_score, mean_absolute_error

import config
from utils import (filter_subgroup, get_threshold_paths, add_time_features, 
                  calculate_c_index, should_stop_training)
from pseudo_labels import (create_generalized_pseudo_labels_corrected, 
                           validate_initial_pseudolabels)
from models import build_model_ensemble
from calibration import (km_aware_calibration, time_aware_refinement, 
                        enforce_constraints_v2, update_expected_survival_time)
from sklearn.isotonic import IsotonicRegression
from evaluation import (plot_observed_vs_predicted_km_curves, 
                       analyze_final_results)
from training_helpers import _process_predictions, _save_final_results


def train_for_threshold(threshold, base_df, subgroup='Overall'):

    
    print("\n" + "="*80)
    print(f"{'='*80}")
    print(f"STARTING TRAINING - SUBGROUP: {subgroup} - THRESHOLD: {threshold} MONTHS")
    print(f"{'='*80}")
    print("="*80 + "\n")
    
    df = filter_subgroup(base_df, subgroup)
    
    if len(df) == 0:
        print(f" No data available for subgroup '{subgroup}'")
        print("Skipping this subgroup...")
        return None
    
    if len(df) < 30:
        print(f"Very small sample size ({len(df)}) for subgroup '{subgroup}'")
        print("Results may not be reliable.")
    
    paths = get_threshold_paths(threshold, subgroup)
    output_dir = paths['output_dir']
    weights_dir = paths['weights_dir']
    input_file = paths['input_file']
    preprocessor_file = paths['preprocessor_file']
    feature_file = paths['feature_file']
    target_col = paths['target_col']
    calibrated_col = paths['calibrated_col']
    expected_time_col = paths['expected_time_col']
    
    # ========================================================================
    # LOAD OR CREATE INITIAL PSEUDO-LABELS
    # ========================================================================
    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        print(f"Loaded existing data file for {subgroup} - {threshold}m threshold")
        
        required_cols = [target_col, calibrated_col, expected_time_col]
        for col in required_cols:
            if col not in df.columns:
                print(f"Creating missing column: {col}")
                df[col] = np.nan
        
        if df[target_col].isna().any():
            print("Re-creating initial pseudo labels (leave-one-out method)")
            df = create_generalized_pseudo_labels_corrected(
                df, config.TIME_COL, config.EVENT_COL, 
                target_col, expected_time_col, threshold
            )
    else:
        print(f"Creating initial pseudo labels for {subgroup} (leave-one-out method)")
        
        df = create_generalized_pseudo_labels_corrected(
            df, config.TIME_COL, config.EVENT_COL, 
            target_col, expected_time_col, threshold
        )
        df[calibrated_col] = df[target_col].copy()
        
        is_valid = validate_initial_pseudolabels(
            df, target_col, config.EVENT_COL, config.TIME_COL, threshold
        )
        
        if not is_valid:
            print(f"\n CRITICAL ERROR: Initial pseudo-labels failed validation")
            print(f"   Please review the pseudo-label generation process")
            print(f"   STOPPING training for {subgroup} - threshold {threshold}m")
            return None
        
        suffix = f"{threshold}m" if subgroup == 'Overall' else f"{threshold}m_{subgroup}"
        iter0_file = os.path.join(output_dir, f'iteration_0_predictions_{suffix}.csv')
        df.to_csv(iter0_file, index=False)
        print(f"\n Saved Iteration 0 predictions to: {iter0_file}")
        print(f"   This file contains initial pseudo-labels for ablation study")
    

    print("Adding time-based features...")
    df = add_time_features(df, config.TIME_COL, threshold)
    

    exclude_cols = config.EXCLUDE_COLS + [
        target_col, calibrated_col, expected_time_col, 
        config.TIME_COL, config.EVENT_COL
    ]
    

    for other_threshold in config.THRESHOLDS:
        for other_subgroup in config.SUBGROUPS.keys():
            if other_threshold != threshold or other_subgroup != subgroup:
                other_paths = get_threshold_paths(other_threshold, other_subgroup)
                exclude_cols.extend([
                    other_paths['target_col'],
                    other_paths['calibrated_col'],
                    other_paths['expected_time_col']
                ])
    
    feature_cols = [c for c in df.columns 
                   if c not in exclude_cols 
                   and not c.startswith('Pseudo_Label') 
                   and not c.startswith('Expected_Survival')]
    
    joblib.dump(feature_cols, feature_file)
    print(f"Using {len(feature_cols)} features for training")
    
    X = df[feature_cols]
    
    if X.shape[1] == 0:
        print("No valid features remaining!")
        return None
    
    # ========================================================================
    # SETUP PREPROCESSING PIPELINE
    # ========================================================================
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['number']).columns.tolist()
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # ========================================================================
    # ITERATIVE REFINEMENT LOOP
    # ========================================================================
    return _iterative_training_loop(
        df, X, feature_cols, preprocessor, threshold, subgroup,
        output_dir, weights_dir, input_file, 
        target_col, calibrated_col, expected_time_col
    )


def _iterative_training_loop(df, X, feature_cols, preprocessor, threshold, subgroup,
                             output_dir, weights_dir, input_file,
                             target_col, calibrated_col, expected_time_col):

    models = None
    best_r2 = -float('inf')
    best_mae = float('inf')
    best_km_loss = float('inf')
    best_c_index = -float('inf')
    best_c_index_iteration = 0
    stagnation_count = 0
    convergence_history = []
    

    best_df_state = None
    
    tf.keras.backend.clear_session()
    
    for iteration in range(1, config.MAX_ITER + 1):
        print(f"\n{'#'*60}")
        print(f"### {subgroup} - Threshold: {threshold}m - Iteration {iteration} ###")
        print(f"{'#'*60}")
        
        X = df[feature_cols]
        y = df[calibrated_col].copy()
        event = df[config.EVENT_COL]
        time = df[config.TIME_COL]
        
        indices = np.arange(len(X))

        try:
            train_idx, test_idx = train_test_split(
                indices, test_size=config.TEST_SIZE, 
                random_state=42, stratify=event
            )
        except ValueError:
            print("Warning: Cannot stratify. Using random split.")
            train_idx, test_idx = train_test_split(
                indices, test_size=config.TEST_SIZE, random_state=42
            )
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        y_train = y.iloc[train_idx].values
        y_test = y.iloc[test_idx].values
        event_train = event.iloc[train_idx].values
        event_test = event.iloc[test_idx].values
        time_train = time.iloc[train_idx].values
        time_test = time.iloc[test_idx].values
        
        if iteration > 1:
            train_min, train_max = y_train.min(), y_train.max()
            if train_max > train_min:
                y_train = (y_train - train_min) / (train_max - train_min + 1e-8)
                y_test = (y_test - train_min) / (train_max - train_min + 1e-8)
        
        y_train = np.clip(y_train, 0.01, 0.99)
        y_test = np.clip(y_test, 0.01, 0.99)
        
        df.iloc[train_idx, df.columns.get_loc(calibrated_col)] = y_train
        df.iloc[test_idx, df.columns.get_loc(calibrated_col)] = y_test
        
        sample_weights = compute_sample_weight('balanced', event_train)
        
        if models is None or stagnation_count > 5:
            print("Building new model ensemble")
            if models is not None:
                for model in models:
                    del model
                tf.keras.backend.clear_session()
            
            models = build_model_ensemble(X_train_processed.shape[1])
            stagnation_count = 0
        
        all_test_preds = []
        all_preds = []
        
        for i, model in enumerate(models):
            print(f"Training model {i+1}/{len(models)}")
            
            initial_lr = config.INITIAL_LEARNING_RATE * (
                config.LEARNING_RATE_DECAY ** (iteration // 5)
            )
            optimizer = Nadam(learning_rate=initial_lr, clipnorm=1.0)
            model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
            
            early_stopping = EarlyStopping(
                monitor='val_mae', patience=15, mode='min', 
                restore_best_weights=True, verbose=0
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_mae', factor=0.7, patience=8, 
                min_lr=1e-7, verbose=0
            )
            
            val_size = int(config.VALIDATION_SIZE * len(X_train_processed))
            if val_size < 5:
                val_size = min(5, len(X_train_processed) // 4)
            
            X_train_final = X_train_processed[:-val_size]
            X_val = X_train_processed[-val_size:]
            y_train_final = y_train[:-val_size]
            y_val = y_train[-val_size:]
            
            history = model.fit(
                X_train_final, y_train_final,
                validation_data=(X_val, y_val),
                epochs=config.EPOCHS,
                batch_size=min(config.BATCH_SIZE, len(X_train_final) // 4),
                callbacks=[early_stopping, reduce_lr, TerminateOnNaN()],
                verbose=0,
                sample_weight=(sample_weights[:-val_size] 
                             if len(sample_weights) > val_size 
                             else sample_weights)
            )
            
            test_pred = model.predict(X_test_processed, verbose=0).flatten()
            all_test_preds.append(test_pred)
            
            full_pred = model.predict(preprocessor.transform(X), verbose=0).flatten()
            all_preds.append(full_pred)
        
        
        result = _process_predictions(
            df, iteration, threshold, subgroup,
            all_test_preds, all_preds, y_test, event, time,
            train_idx, test_idx, event_test, time_test,
            target_col, calibrated_col, expected_time_col,
            models, preprocessor, weights_dir, input_file,
            convergence_history, best_r2, best_mae, best_c_index, 
            best_km_loss, best_c_index_iteration, stagnation_count,
            best_df_state
        )
        
        if result is None:

            break
        
        (df, convergence_history, best_r2, best_mae, best_c_index, 
         best_km_loss, best_c_index_iteration, stagnation_count, 
         best_df_state, should_break) = result
        
        if should_break:
            break
    
    _save_final_results(
        df, convergence_history, output_dir, weights_dir, input_file,
        threshold, subgroup, target_col, calibrated_col, 
        best_c_index, best_c_index_iteration
    )
    
    tf.keras.backend.clear_session()
    
    return df