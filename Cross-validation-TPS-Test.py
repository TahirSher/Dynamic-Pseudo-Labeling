import pandas as pd
import numpy as np
import torch
from lifelines import KaplanMeierFitter, WeibullFitter
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.class_weight import compute_sample_weight
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from scipy.stats import spearmanr
from scipy.interpolate import interp1d, PchipInterpolator
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.utils import resample
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("Using PyTorch device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Using CPU")

# ============================================================================
#  CONFIGURATION
# ============================================================================
USE_CROSS_VALIDATION = True 
N_CV_FOLDS = 5  
CV_RANDOM_STATE = 42  
THRESHOLDS = [60, 84, 120]

if USE_CROSS_VALIDATION:
    BASE_OUTPUT_DIR = "14SPARC_CV_multi_threshold_survival_output"
    BASE_WEIGHTS_DIR = "14SPARC_CV_multi_threshold_survival_weights"
else:
    BASE_OUTPUT_DIR = "2000hrs Final Lung CANCER 14 Jan TPS-multi_threshold_survival_output"
    BASE_WEIGHTS_DIR = "2000hrs Final Lung CANCER 14 Jan Final TPS-multi_threshold_survival_weights"


# TPS Subgroup definitions
SUBGROUPS = {
    'Overall': None,  # Overall population 
    #'C1': {'TPS_L1': 1, 'TPS_1_49': 0, 'TPS_50L': 0},  # TPS < 1%
    #'C2': {'TPS_L1': 0, 'TPS_1_49': 1, 'TPS_50L': 0},  # TPS 1-49%
    #'C3': {'TPS_L1': 0, 'TPS_1_49': 0, 'TPS_50L': 1}   # TPS >= 50%
}

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(BASE_WEIGHTS_DIR, exist_ok=True)

# Parameters
TIME_COL = 'Time'
EVENT_COL = 'Event'
MAX_ITER = 100  
TARGET_R2 = 0.95
MIN_DELTA = 0.001
PATIENCE = 15
PLOT_INTERVAL = 10
ENSEMBLE_SIZE = 3
TARGET_C_INDEX = 0.85
C_INDEX_TOLERANCE = 0.03
# ============================================================================
# CV CONFIGURATION
# ============================================================================
N_OUTER_FOLDS = 5     
N_INNER_FOLDS = 3      
N_REPEATS = 3          
N_BOOTSTRAP = 200      
# ============================================================================
# CROSS-VALIDATION FUNCTIONS
# ============================================================================

def get_cv_splits(df, n_folds=5, random_state=42):

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    splits = list(skf.split(df, df[EVENT_COL]))
    
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION SETUP")
    print(f"{'='*70}")
    print(f"  Number of folds: {n_folds}")
    print(f"  Random state: {random_state}")
    print(f"  Total samples: {len(df)}")
    print(f"  Events: {df[EVENT_COL].sum()} ({df[EVENT_COL].sum()/len(df)*100:.1f}%)")
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
        train_events = df[EVENT_COL].iloc[train_idx].sum()
        test_events = df[EVENT_COL].iloc[test_idx].sum()
        print(f"  Fold {fold_idx}: Train={len(train_idx)} ({train_events} events), "
              f"Test={len(test_idx)} ({test_events} events)")
    
    print(f"{'='*70}\n")
    return splits


def create_pseudo_labels_for_fold(df, train_idx, time_col, event_col, threshold):

    time_train = df[time_col].iloc[train_idx].values
    event_train = df[event_col].iloc[train_idx].values
    n_train = len(train_idx)

    kmf_train = KaplanMeierFitter()
    kmf_train.fit(time_train, event_train)
    
    try:
        S_all_train = kmf_train.predict(threshold)
        if np.isnan(S_all_train):
            S_all_train = kmf_train.survival_function_.iloc[-1].values[0]
    except:
        S_all_train = kmf_train.survival_function_.iloc[-1].values[0]

    pseudo_labels_train = np.zeros(n_train)
    
    progress_step = max(1, n_train // 10)
    for i in range(n_train):
        if i % progress_step == 0:
            print(f"    Computing pseudo-labels: {i}/{n_train} ({i/n_train*100:.0f}%)", end='\r')
        
        mask = np.ones(n_train, dtype=bool)
        mask[i] = False
        
        kmf_loo = KaplanMeierFitter()
        kmf_loo.fit(time_train[mask], event_train[mask])
        
        try:
            S_loo = kmf_loo.predict(threshold)
            if np.isnan(S_loo):
                S_loo = kmf_loo.survival_function_.iloc[-1].values[0]
        except:
            S_loo = kmf_loo.survival_function_.iloc[-1].values[0]
        
        pseudo_obs = n_train * S_all_train - (n_train - 1) * S_loo
        pseudo_labels_train[i] = np.clip(pseudo_obs, 0.001, 0.999)
    
    print(f"    Computing pseudo-labels: {n_train}/{n_train}")
    
    return pseudo_labels_train


def train_sparc_single_fold(df, train_idx, test_idx, threshold, fold_num, subgroup='Overall'):

    print(f"\n{'='*70}")
    print(f"FOLD {fold_num} - SPARC Training")
    print(f"Subgroup: {subgroup} | Threshold: {threshold}m")
    print(f"{'='*70}")

    print(f"  Creating pseudo-labels from training fold only...")
    pseudo_labels_train = create_pseudo_labels_for_fold(
        df, train_idx, TIME_COL, EVENT_COL, threshold
    )

    c_index_train = concordance_index_censored(
        df[EVENT_COL].iloc[train_idx].astype(bool).values,
        df[TIME_COL].iloc[train_idx].values,
        1 - pseudo_labels_train
    )[0]
    
    print(f"  Initial pseudo-label C-index (train only): {c_index_train:.4f}")
    
    if c_index_train > 0.90:
        print(f"   C-index suspiciously high ({c_index_train:.4f})")
        print(f"   Expected range: 0.55-0.75 for initial pseudo-observations")
    
    df_fold = df.copy()
    df_fold = add_time_features(df_fold, TIME_COL, threshold)
    
    exclude_cols = ['ID', 'year', 'Age at Diagnosis', TIME_COL, EVENT_COL,
                   'TPS_L1', 'TPS_1_49', 'TPS_50L']
    
    feature_cols = [c for c in df_fold.columns if c not in exclude_cols]
    
    X_train = df_fold[feature_cols].iloc[train_idx]
    X_test = df_fold[feature_cols].iloc[test_idx]
    y_train = pseudo_labels_train
    
    time_train = df[TIME_COL].iloc[train_idx].values
    time_test = df[TIME_COL].iloc[test_idx].values
    event_train = df[EVENT_COL].iloc[train_idx].values
    event_test = df[EVENT_COL].iloc[test_idx].values
    
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
    
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
    
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"  Training neural network...")
    model = build_time_aware_model(X_train_processed.shape[1])
    
    val_size = int(0.15 * len(X_train_processed))
    if val_size < 5:
        val_size = min(5, len(X_train_processed) // 4)
    
    X_train_final = X_train_processed[:-val_size]
    X_val = X_train_processed[-val_size:]
    y_train_final = y_train[:-val_size]
    y_val = y_train[-val_size:]
    
    early_stopping = EarlyStopping(
        monitor='val_mae', patience=15, mode='min', 
        restore_best_weights=True, verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_mae', factor=0.7, patience=8, 
        min_lr=1e-7, verbose=0
    )
    
    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=min(32, len(X_train_final) // 4),
        callbacks=[early_stopping, reduce_lr, TerminateOnNaN()],
        verbose=0
    )

    y_pred_test_raw = model.predict(X_test_processed, verbose=0).flatten()
    y_pred_test_raw = np.clip(y_pred_test_raw, 0.001, 0.999)
    
    c_index_raw = concordance_index_censored(
        event_test.astype(bool),
        time_test,
        1 - y_pred_test_raw  
    )[0]
    
    print(f"\n  {'='*50}")
    print(f"  CALIBRATION PIPELINE (C-index tracking):")
    print(f"  {'='*50}")
    print(f"    RAW predictions:      C-index = {c_index_raw:.4f}")
    
    y_pred_train_raw = model.predict(X_train_processed, verbose=0).flatten()
    y_pred_train_raw = np.clip(y_pred_train_raw, 0.001, 0.999)
    
    # ================================================================
    # STEP 1: Isotonic Regression
    # ================================================================
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
    iso.fit(y_pred_train_raw, y_train)  
    
    y_pred_test_iso = iso.predict(y_pred_test_raw)  # Apply to test fold
    y_pred_test_iso = np.clip(y_pred_test_iso, 0.001, 0.999)
    
    c_index_after_iso = concordance_index_censored(
        event_test.astype(bool),
        time_test,
        1 - y_pred_test_iso
    )[0]
    
    print(f"    After Isotonic:       C-index = {c_index_after_iso:.4f}")
    
    # ================================================================
    # STEP 2: KM-aware calibration
    # ================================================================
   
    df_test_temp = df.iloc[test_idx].copy()
    df_test_temp['temp_pred'] = y_pred_test_iso
    
    y_pred_test_km = km_aware_calibration(
        df_test_temp, 
        y_pred_test_iso, 
        iteration=1,
        time_col=TIME_COL, 
        event_col=EVENT_COL, 
        threshold=threshold
    )
    
    c_index_after_km = concordance_index_censored(
        event_test.astype(bool),
        time_test,
        1 - y_pred_test_km
    )[0]
    
    print(f"    After KM calibration: C-index = {c_index_after_km:.4f}")
    
    # ================================================================
    # STEP 3: Time-aware refinement
    # ================================================================
    y_pred_test_final = time_aware_refinement(
        df_test_temp,
        y_pred_test_km,
        event_test,
        time_test,
        threshold,
        iteration=1
    )
    
    c_index_after_time = concordance_index_censored(
        event_test.astype(bool),
        time_test,
        1 - y_pred_test_final
    )[0]
    
    print(f"    After Time-aware:     C-index = {c_index_after_time:.4f}")
    
    if c_index_after_time < c_index_raw - 0.05:
        print(f"\n   MAJOR C-INDEX DROP: {c_index_raw:.4f}, {c_index_after_time:.4f}")
        print(f"    Rolling back to lighter calibration...")
        y_pred_test_final = 0.8 * y_pred_test_raw + 0.2 * y_pred_test_final
        c_index_final = concordance_index_censored(
            event_test.astype(bool),
            time_test,
            1 - y_pred_test_final
        )[0]
        print(f"    After rollback:       C-index = {c_index_final:.4f}")
    else:
        c_index_final = c_index_after_time
    
    print(f"  {'='*50}\n")
    
    pseudo_labels_test = create_pseudo_labels_for_fold(df, test_idx, TIME_COL, EVENT_COL, threshold)
    
    test_mae = mean_absolute_error(pseudo_labels_test, y_pred_test_final)
    
    print(f"  FINAL TEST METRICS:")
    print(f"    C-index: {c_index_final:.4f}")
    print(f"    MAE:     {test_mae:.4f}")
    
    del model
    tf.keras.backend.clear_session()
    
    return {
        'test_predictions': y_pred_test_final,
        'test_risk_scores': 1 - y_pred_test_final,
        'test_indices': test_idx,
        'test_c_index': c_index_final,
        'test_c_index_raw': c_index_raw,  
        'test_c_index_iso': c_index_after_iso,  
        'test_c_index_km': c_index_after_km,  
        'test_c_index_time': c_index_after_time,  
        'test_mae': test_mae,
        'event_test': event_test,
        'time_test': time_test
    }


def train_sparc_with_cv(base_df, threshold, subgroup='Overall', n_folds=5):

    print(f"\n{'='*80}")
    print(f"SPARC WITH {n_folds}-FOLD CROSS-VALIDATION")
    print(f"Subgroup: {subgroup} | Threshold: {threshold}m")
    print(f"{'='*80}")
    
    df = filter_subgroup(base_df, subgroup)
    
    if len(df) == 0:
        print(f"ERROR: No data for subgroup '{subgroup}'")
        return None
    
    if len(df) < 30:
        print(f"WARNING: Small sample size ({len(df)}) for subgroup '{subgroup}'")
    
    paths = get_threshold_paths(threshold, subgroup)
    output_dir = paths['output_dir']
    weights_dir = paths['weights_dir']
    
    cv_splits = get_cv_splits(df, n_folds=n_folds, random_state=CV_RANDOM_STATE)

    all_test_predictions_raw = np.zeros(len(df))
    all_test_predictions_iso = np.zeros(len(df))
    all_test_predictions_km = np.zeros(len(df))
    all_test_predictions_time = np.zeros(len(df))
    all_test_predictions_final = np.zeros(len(df))
    all_test_risk_scores = np.zeros(len(df))
    all_test_indices = []
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits, 1):
        fold_result = train_sparc_single_fold(
            df=df,
            train_idx=train_idx,
            test_idx=test_idx,
            threshold=threshold,
            fold_num=fold_idx,
            subgroup=subgroup
        )
        
        all_test_predictions_final[test_idx] = fold_result['test_predictions']
        all_test_risk_scores[test_idx] = fold_result['test_risk_scores']

        all_test_indices.extend(test_idx.tolist())
        fold_results.append(fold_result)

    all_events = df[EVENT_COL].values
    all_times = df[TIME_COL].values
    
    overall_c_index_final = concordance_index_censored(
        all_events.astype(bool),
        all_times,
        all_test_risk_scores
    )[0]
    
    fold_c_indices_raw = [r['test_c_index_raw'] for r in fold_results]
    fold_c_indices_iso = [r['test_c_index_iso'] for r in fold_results]
    fold_c_indices_km = [r['test_c_index_km'] for r in fold_results]
    fold_c_indices_time = [r['test_c_index_time'] for r in fold_results]
    fold_c_indices_final = [r['test_c_index'] for r in fold_results]
    
    mean_c_index_raw = np.mean(fold_c_indices_raw)
    std_c_index_raw = np.std(fold_c_indices_raw)
    
    mean_c_index_iso = np.mean(fold_c_indices_iso)
    std_c_index_iso = np.std(fold_c_indices_iso)
    
    mean_c_index_km = np.mean(fold_c_indices_km)
    std_c_index_km = np.std(fold_c_indices_km)
    
    mean_c_index_time = np.mean(fold_c_indices_time)
    std_c_index_time = np.std(fold_c_indices_time)
    
    mean_c_index_final = np.mean(fold_c_indices_final)
    std_c_index_final = np.std(fold_c_indices_final)
    

    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION RESULTS - {subgroup} - {threshold}m")
    print(f"{'='*80}")
    
    print(f"\n{'RAW PREDICTIONS (before any calibration)':^80}")
    print(f"  Mean C-index across folds: {mean_c_index_raw:.4f} +/- {std_c_index_raw:.4f}")
    for fold_idx, c_idx in enumerate(fold_c_indices_raw, 1):
        print(f"  Fold {fold_idx}: {c_idx:.4f}")
    
    print(f"\n{'AFTER ISOTONIC REGRESSION':^80}")
    print(f"  Mean C-index across folds: {mean_c_index_iso:.4f} +/- {std_c_index_iso:.4f}")
    for fold_idx, c_idx in enumerate(fold_c_indices_iso, 1):
        print(f"  Fold {fold_idx}: {c_idx:.4f}")
    
    print(f"\n{'AFTER KM CALIBRATION':^80}")
    print(f"  Mean C-index across folds: {mean_c_index_km:.4f} +/- {std_c_index_km:.4f}")
    for fold_idx, c_idx in enumerate(fold_c_indices_km, 1):
        print(f"  Fold {fold_idx}: {c_idx:.4f}")
    
    print(f"\n{'AFTER TIME-AWARE REFINEMENT':^80}")
    print(f"  Mean C-index across folds: {mean_c_index_time:.4f} +/- {std_c_index_time:.4f}")
    for fold_idx, c_idx in enumerate(fold_c_indices_time, 1):
        print(f"  Fold {fold_idx}: {c_idx:.4f}")
    
    print(f"\n{'FINAL (all calibration steps combined)':^80}")
    print(f"  Overall C-index (all folds combined): {overall_c_index_final:.4f}")
    print(f"  Mean C-index across folds: {mean_c_index_final:.4f} +/- {std_c_index_final:.4f}")
    for fold_idx, c_idx in enumerate(fold_c_indices_final, 1):
        print(f"  Fold {fold_idx}: {c_idx:.4f}")
    
    print(f"{'='*80}\n")

    suffix = f"{threshold}m" if subgroup == 'Overall' else f"{threshold}m_{subgroup}"
    
    df_cv = df.copy()
    df_cv[f'SPARC_CV_Prediction_{suffix}'] = all_test_predictions_final
    df_cv[f'SPARC_CV_Risk_Score_{suffix}'] = all_test_risk_scores
    df_cv[f'Pseudo_Label_Calibrated_{suffix}'] = all_test_predictions_final
    
    cv_predictions_file = os.path.join(output_dir, f'sparc_cv_predictions_{suffix}.csv')
    df_cv.to_csv(cv_predictions_file, index=False)
    print(f"Saved CV predictions to: {cv_predictions_file}")
    
    compat_file = os.path.join(output_dir, f'optimized_survival_probabilities_{suffix}.csv')
    df_cv.to_csv(compat_file, index=False)
    print(f" Saved (compatibility format): {compat_file}")
    
    fold_results_df = pd.DataFrame([{
        'Fold': i+1,
        'C_Index_Raw': fold_c_indices_raw[i],
        'C_Index_Isotonic': fold_c_indices_iso[i],
        'C_Index_KM': fold_c_indices_km[i],
        'C_Index_TimeAware': fold_c_indices_time[i],
        'C_Index_Final': fold_c_indices_final[i],
        'MAE': fold_results[i]['test_mae'],
        'N_Test': len(fold_results[i]['test_indices'])
    } for i in range(len(fold_results))])
    
    fold_results_df['Overall_C_Index_Final'] = overall_c_index_final
    fold_results_df['Mean_C_Index_Raw'] = mean_c_index_raw
    fold_results_df['Std_C_Index_Raw'] = std_c_index_raw
    fold_results_df['Mean_C_Index_Isotonic'] = mean_c_index_iso
    fold_results_df['Std_C_Index_Isotonic'] = std_c_index_iso
    fold_results_df['Mean_C_Index_KM'] = mean_c_index_km
    fold_results_df['Std_C_Index_KM'] = std_c_index_km
    fold_results_df['Mean_C_Index_TimeAware'] = mean_c_index_time
    fold_results_df['Std_C_Index_TimeAware'] = std_c_index_time
    fold_results_df['Mean_C_Index_Final'] = mean_c_index_final
    fold_results_df['Std_C_Index_Final'] = std_c_index_final
    fold_results_df['Threshold'] = threshold
    fold_results_df['Subgroup'] = subgroup
    
    fold_results_file = os.path.join(output_dir, f'sparc_cv_fold_results_{suffix}.csv')
    fold_results_df.to_csv(fold_results_file, index=False)
    print(f" Saved comprehensive fold results to: {fold_results_file}")
    
    convergence_df = pd.DataFrame([{
        'iteration': 0,
        'Iteration': 0,
        'C_Index': mean_c_index_final,
        'c_index': mean_c_index_final,
        'C_Index_Raw': mean_c_index_raw,
        'C_Index_Isotonic': mean_c_index_iso,
        'C_Index_KM': mean_c_index_km,
        'C_Index_TimeAware': mean_c_index_time,
        'Best_C_Index': True
    }])
    
    history_file = os.path.join(output_dir, f'convergence_history_{suffix}.csv')
    convergence_df.to_csv(history_file, index=False)
    print(f" Saved convergence history (compat): {history_file}")
    
    return {
        'df_with_predictions': df_cv,
        'predictions': all_test_predictions_final,
        'risk_scores': all_test_risk_scores,
        'overall_c_index': overall_c_index_final,
        'mean_c_index': mean_c_index_final,
        'std_c_index': std_c_index_final,
        'mean_c_index_raw': mean_c_index_raw,
        'std_c_index_raw': std_c_index_raw,
        'mean_c_index_iso': mean_c_index_iso,
        'std_c_index_iso': std_c_index_iso,
        'mean_c_index_km': mean_c_index_km,
        'std_c_index_km': std_c_index_km,
        'mean_c_index_time': mean_c_index_time,
        'std_c_index_time': std_c_index_time,
        'fold_results': fold_results,
        'fold_c_indices_raw': fold_c_indices_raw,
        'fold_c_indices_iso': fold_c_indices_iso,
        'fold_c_indices_km': fold_c_indices_km,
        'fold_c_indices_time': fold_c_indices_time,
        'fold_c_indices_final': fold_c_indices_final
    }

def cross_validate_threshold_comprehensive(threshold, df, subgroup='Overall', feature_cols=None):    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE CROSS-VALIDATION - ALL 3 METHODS")
    print(f"Subgroup: {subgroup} | Threshold: {threshold}m")
    print(f"{'='*80}")
    
    paths = get_threshold_paths(threshold, subgroup)
    calibrated_col = paths['calibrated_col']
    
    df_filtered = filter_subgroup(df, subgroup)
    if len(df_filtered) < 50:
        print(f"Skipping CV for {subgroup} {threshold}m: too few samples")
        return None

    df_filtered = create_generalized_pseudo_labels_corrected(
        df_filtered.copy(), TIME_COL, EVENT_COL, calibrated_col,
        paths['expected_time_col'], threshold
    )
    df_filtered = add_time_features(df_filtered, TIME_COL, threshold)

    if feature_cols is None:
        exclude_cols = ['ID', 'year', 'Age at Diagnosis', TIME_COL, EVENT_COL, 
                       calibrated_col, paths['expected_time_col'],
                       'TPS_L1', 'TPS_1_49', 'TPS_50L']
        feature_cols = [c for c in df_filtered.columns if c not in exclude_cols]

    X = df_filtered[feature_cols]
    y = df_filtered[calibrated_col].values
    event = df_filtered[EVENT_COL].values.astype(bool)
    time = df_filtered[TIME_COL].values

    time_tertiles = np.digitize(time, np.percentile(time, [33.3, 66.6]))
    strata = event.astype(int) * 3 + time_tertiles

    cv_results = {
        'nested_results': {},
        'repeated_results': {},
        'bootstrap_results': {},
        'combined_summary': {}
    }

    # ========================================================================
    # METHOD 1: NESTED STRATIFIED K-FOLD
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"METHOD 1: NESTED STRATIFIED K-FOLD ({N_OUTER_FOLDS} folds)")
    print(f"{'='*70}")
    
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=42)
    
    nested_c_indices_raw = []
    nested_c_indices_iso = []
    nested_c_indices_km = []
    nested_c_indices_time = []
    nested_c_indices_final = []
    
    nested_predictions_raw = np.zeros(len(df_filtered))
    nested_predictions_iso = np.zeros(len(df_filtered))
    nested_predictions_km = np.zeros(len(df_filtered))
    nested_predictions_time = np.zeros(len(df_filtered))
    nested_predictions_final = np.zeros(len(df_filtered))
    nested_risk_scores = np.zeros(len(df_filtered))
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, strata), 1):
        fold_result = train_sparc_single_fold(
            df=df_filtered,
            train_idx=train_idx,
            test_idx=test_idx,
            threshold=threshold,
            fold_num=fold_idx,
            subgroup=subgroup
        )

        nested_predictions_final[test_idx] = fold_result['test_predictions']
        nested_risk_scores[test_idx] = fold_result['test_risk_scores']
        
        nested_c_indices_raw.append(fold_result['test_c_index_raw'])
        nested_c_indices_iso.append(fold_result['test_c_index_iso'])
        nested_c_indices_km.append(fold_result['test_c_index_km'])
        nested_c_indices_time.append(fold_result['test_c_index_time'])
        nested_c_indices_final.append(fold_result['test_c_index'])

    overall_nested_final = concordance_index_censored(event, time, nested_risk_scores)[0]
    
    cv_results['nested_results'] = {
        'overall_c_index_final': overall_nested_final,
        'mean_c_index_raw': np.mean(nested_c_indices_raw),
        'std_c_index_raw': np.std(nested_c_indices_raw),
        'mean_c_index_iso': np.mean(nested_c_indices_iso),
        'std_c_index_iso': np.std(nested_c_indices_iso),
        'mean_c_index_km': np.mean(nested_c_indices_km),
        'std_c_index_km': np.std(nested_c_indices_km),
        'mean_c_index_time': np.mean(nested_c_indices_time),
        'std_c_index_time': np.std(nested_c_indices_time),
        'mean_c_index_final': np.mean(nested_c_indices_final),
        'std_c_index_final': np.std(nested_c_indices_final),
        'fold_c_indices_raw': nested_c_indices_raw,
        'fold_c_indices_iso': nested_c_indices_iso,
        'fold_c_indices_km': nested_c_indices_km,
        'fold_c_indices_time': nested_c_indices_time,
        'fold_c_indices_final': nested_c_indices_final
    }

    # ========================================================================
    # METHOD 2: REPEATED STRATIFIED K-FOLD
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"METHOD 2: REPEATED STRATIFIED K-FOLD (5 folds {N_REPEATS} repeats)")
    print(f"{'='*70}")
    
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=42)
    
    repeated_c_indices_raw = []
    repeated_c_indices_iso = []
    repeated_c_indices_km = []
    repeated_c_indices_time = []
    repeated_c_indices_final = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, strata), 1):
        print(f"  Repeat fold {fold_idx}/{5*N_REPEATS}", end='\r')
        
        fold_result = train_sparc_single_fold(
            df=df_filtered,
            train_idx=train_idx,
            test_idx=test_idx,
            threshold=threshold,
            fold_num=fold_idx,
            subgroup=subgroup
        )
        
        repeated_c_indices_raw.append(fold_result['test_c_index_raw'])
        repeated_c_indices_iso.append(fold_result['test_c_index_iso'])
        repeated_c_indices_km.append(fold_result['test_c_index_km'])
        repeated_c_indices_time.append(fold_result['test_c_index_time'])
        repeated_c_indices_final.append(fold_result['test_c_index'])
    
    print()  
    
    cv_results['repeated_results'] = {
        'mean_c_index_raw': np.mean(repeated_c_indices_raw),
        'std_c_index_raw': np.std(repeated_c_indices_raw),
        'mean_c_index_iso': np.mean(repeated_c_indices_iso),
        'std_c_index_iso': np.std(repeated_c_indices_iso),
        'mean_c_index_km': np.mean(repeated_c_indices_km),
        'std_c_index_km': np.std(repeated_c_indices_km),
        'mean_c_index_time': np.mean(repeated_c_indices_time),
        'std_c_index_time': np.std(repeated_c_indices_time),
        'mean_c_index_final': np.mean(repeated_c_indices_final),
        'std_c_index_final': np.std(repeated_c_indices_final),
        'fold_c_indices_raw': repeated_c_indices_raw,
        'fold_c_indices_iso': repeated_c_indices_iso,
        'fold_c_indices_km': repeated_c_indices_km,
        'fold_c_indices_time': repeated_c_indices_time,
        'fold_c_indices_final': repeated_c_indices_final
    }

    # ========================================================================
    # METHOD 3: BOOTSTRAP OPTIMISM CORRECTION
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"METHOD 3: BOOTSTRAP OPTIMISM CORRECTION ({N_BOOTSTRAP} iterations)")
    print(f"{'='*70}")
    
    preprocessor_full = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]),
         X.select_dtypes(include=['number']).columns),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
         X.select_dtypes(include=['object', 'category']).columns)
    ])
    
    preprocessor_full.fit(X)
    X_full_p = preprocessor_full.transform(X)

    model_full = build_time_aware_model(X_full_p.shape[1])
    model_full.fit(X_full_p, y, epochs=150, batch_size=32, verbose=0,
                   validation_split=0.2, callbacks=[EarlyStopping(patience=20, verbose=0)])
    
    pred_full_raw = model_full.predict(X_full_p, verbose=0).flatten()
    pred_full_raw = np.clip(pred_full_raw, 0.001, 0.999)
    
    apparent_c_index_raw = concordance_index_censored(event, time, 1 - pred_full_raw)[0]
    
    iso_full = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
    iso_full.fit(pred_full_raw, y)
    pred_full_iso = iso_full.predict(pred_full_raw)
    
    apparent_c_index_iso = concordance_index_censored(event, time, 1 - pred_full_iso)[0]
    
    df_temp = df_filtered.copy()
    df_temp['temp_pred'] = pred_full_iso
    pred_full_km = km_aware_calibration(df_temp, pred_full_iso, iteration=1, 
                                        time_col=TIME_COL, event_col=EVENT_COL, threshold=threshold)
    
    apparent_c_index_km = concordance_index_censored(event, time, 1 - pred_full_km)[0]
    
    pred_full_time = time_aware_refinement(df_temp, pred_full_km, event, time, threshold, iteration=1)
    
    apparent_c_index_time = concordance_index_censored(event, time, 1 - pred_full_time)[0]
    
    print(f"\n  APPARENT PERFORMANCE (on full dataset):")
    print(f"    RAW:        {apparent_c_index_raw:.4f}")
    print(f"    Isotonic:   {apparent_c_index_iso:.4f}")
    print(f"    KM:         {apparent_c_index_km:.4f}")
    print(f"    Time-aware: {apparent_c_index_time:.4f}")
    
    print(f"\n  Running bootstrap iterations...")
    
    optimism_raw = []
    optimism_iso = []
    optimism_km = []
    optimism_time = []
    
    for boot_iter in range(N_BOOTSTRAP):
        if boot_iter % 50 == 0:
            print(f"    Bootstrap: {boot_iter}/{N_BOOTSTRAP}", end='\r')

        indices_boot = np.random.choice(len(X), size=len(X), replace=True)
        X_b = X.iloc[indices_boot]
        y_b = y[indices_boot]
        event_b = event[indices_boot]
        time_b = time[indices_boot]
        
        X_b_p = preprocessor_full.transform(X_b)
        
        model_b = build_time_aware_model(X_b_p.shape[1])
        model_b.fit(X_b_p, y_b, epochs=100, batch_size=32, verbose=0, 
                   callbacks=[EarlyStopping(patience=15, verbose=0)])

        # ================================================================
        # STAGE 1: RAW PREDICTIONS
        # ================================================================
        pred_b_raw = model_b.predict(X_b_p, verbose=0).flatten()
        pred_b_raw = np.clip(pred_b_raw, 0.001, 0.999)
        boot_c_raw = concordance_index_censored(event_b, time_b, 1 - pred_b_raw)[0]
        
        pred_orig_raw = model_b.predict(X_full_p, verbose=0).flatten()
        pred_orig_raw = np.clip(pred_orig_raw, 0.001, 0.999)
        test_c_raw = concordance_index_censored(event, time, 1 - pred_orig_raw)[0]
        
        optimism_raw.append(boot_c_raw - test_c_raw)
        
        # ================================================================
        # STAGE 2: ISOTONIC REGRESSION
        # ================================================================
        iso_b = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
        iso_b.fit(pred_b_raw, y_b)
        pred_b_iso = iso_b.predict(pred_b_raw)
        boot_c_iso = concordance_index_censored(event_b, time_b, 1 - pred_b_iso)[0]
        
        pred_orig_iso = iso_b.predict(pred_orig_raw)
        test_c_iso = concordance_index_censored(event, time, 1 - pred_orig_iso)[0]
        
        optimism_iso.append(boot_c_iso - test_c_iso)
        
        # ================================================================
        # STAGE 3: KM CALIBRATION
        # ================================================================
        df_b_temp = df_filtered.iloc[indices_boot].copy()
        df_b_temp['temp_pred'] = pred_b_iso
        
        try:
            pred_b_km = km_aware_calibration(df_b_temp, pred_b_iso, iteration=1,
                                            time_col=TIME_COL, event_col=EVENT_COL, threshold=threshold)
            boot_c_km = concordance_index_censored(event_b, time_b, 1 - pred_b_km)[0]

            pred_orig_km = pred_orig_iso  
            test_c_km = concordance_index_censored(event, time, 1 - pred_orig_km)[0]
            
            optimism_km.append(boot_c_km - test_c_km)
        except:
            optimism_km.append(optimism_iso[-1])  
        
        # ================================================================
        # STAGE 4: TIME-AWARE REFINEMENT
        # ================================================================
        try:
            pred_b_time = time_aware_refinement(df_b_temp, pred_b_km if 'pred_b_km' in locals() else pred_b_iso,
                                               event_b, time_b, threshold, iteration=1)
            boot_c_time = concordance_index_censored(event_b, time_b, 1 - pred_b_time)[0]
            
            pred_orig_time = pred_orig_km if 'pred_orig_km' in locals() else pred_orig_iso
            test_c_time = concordance_index_censored(event, time, 1 - pred_orig_time)[0]
            
            optimism_time.append(boot_c_time - test_c_time)
        except:
            optimism_time.append(optimism_km[-1] if optimism_km else optimism_iso[-1])

        del model_b
        tf.keras.backend.clear_session()

    print()  
    
    optimism_corrected_raw = apparent_c_index_raw - np.mean(optimism_raw)
    optimism_corrected_iso = apparent_c_index_iso - np.mean(optimism_iso)
    optimism_corrected_km = apparent_c_index_km - np.mean(optimism_km)
    optimism_corrected_time = apparent_c_index_time - np.mean(optimism_time)
    
    print(f"\n  OPTIMISM CORRECTION RESULTS:")
    print(f"    {'Stage':<15} {'Apparent':<12} {'Optimism':<12} {'Corrected':<12}")
    print(f"    {'-'*50}")
    print(f"    {'RAW':<15} {apparent_c_index_raw:<12.4f} {np.mean(optimism_raw):<12.4f} {optimism_corrected_raw:<12.4f}")
    print(f"    {'Isotonic':<15} {apparent_c_index_iso:<12.4f} {np.mean(optimism_iso):<12.4f} {optimism_corrected_iso:<12.4f}")
    print(f"    {'KM':<15} {apparent_c_index_km:<12.4f} {np.mean(optimism_km):<12.4f} {optimism_corrected_km:<12.4f}")
    print(f"    {'Time-aware':<15} {apparent_c_index_time:<12.4f} {np.mean(optimism_time):<12.4f} {optimism_corrected_time:<12.4f}")
    
    cv_results['bootstrap_results'] = {
        'apparent_c_index_raw': apparent_c_index_raw,
        'optimism_corrected_raw': optimism_corrected_raw,
        'mean_optimism_raw': np.mean(optimism_raw),
        'std_optimism_raw': np.std(optimism_raw),
        
        'apparent_c_index_iso': apparent_c_index_iso,
        'optimism_corrected_iso': optimism_corrected_iso,
        'mean_optimism_iso': np.mean(optimism_iso),
        'std_optimism_iso': np.std(optimism_iso),
        
        'apparent_c_index_km': apparent_c_index_km,
        'optimism_corrected_km': optimism_corrected_km,
        'mean_optimism_km': np.mean(optimism_km),
        'std_optimism_km': np.std(optimism_km),
        
        'apparent_c_index_time': apparent_c_index_time,
        'optimism_corrected_time': optimism_corrected_time,
        'mean_optimism_time': np.mean(optimism_time),
        'std_optimism_time': np.std(optimism_time)
    }


    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE CV RESULTS - {subgroup} - {threshold}m")
    print(f"{'='*80}")
    
    print(f"\n{'METHOD 1: NESTED STRATIFIED K-FOLD':^80}")
    print(f"  Overall C-index (all folds combined): {cv_results['nested_results']['overall_c_index_final']:.4f}")
    print(f"\n  {'Mean C-index across folds':<45}")
    print(f"    RAW:                 {cv_results['nested_results']['mean_c_index_raw']:.4f} +/- {cv_results['nested_results']['std_c_index_raw']:.4f}")
    print(f"    After Isotonic:      {cv_results['nested_results']['mean_c_index_iso']:.4f} +/- {cv_results['nested_results']['std_c_index_iso']:.4f}")
    print(f"    After KM:            {cv_results['nested_results']['mean_c_index_km']:.4f} +/- {cv_results['nested_results']['std_c_index_km']:.4f}")
    print(f"    After Time-Aware:    {cv_results['nested_results']['mean_c_index_time']:.4f} +/- {cv_results['nested_results']['std_c_index_time']:.4f}")
    print(f"    FINAL:               {cv_results['nested_results']['mean_c_index_final']:.4f} +/- {cv_results['nested_results']['std_c_index_final']:.4f}")
    
    print(f"\n{'METHOD 2: REPEATED STRATIFIED K-FOLD':^80}")
    print(f"  {'Mean C-index across folds':<45}")
    print(f"    RAW:                 {cv_results['repeated_results']['mean_c_index_raw']:.4f} +/- {cv_results['repeated_results']['std_c_index_raw']:.4f}")
    print(f"    After Isotonic:      {cv_results['repeated_results']['mean_c_index_iso']:.4f} +/- {cv_results['repeated_results']['std_c_index_iso']:.4f}")
    print(f"    After KM:            {cv_results['repeated_results']['mean_c_index_km']:.4f} +/- {cv_results['repeated_results']['std_c_index_km']:.4f}")
    print(f"    After Time-Aware:    {cv_results['repeated_results']['mean_c_index_time']:.4f} +/- {cv_results['repeated_results']['std_c_index_time']:.4f}")
    print(f"    FINAL:               {cv_results['repeated_results']['mean_c_index_final']:.4f} +/- {cv_results['repeated_results']['std_c_index_final']:.4f}")
    
    print(f"\n{'METHOD 3: BOOTSTRAP OPTIMISM CORRECTION':^80}")
    print(f"  {'RAW Predictions:':<45}")
    print(f"    Apparent:            {cv_results['bootstrap_results']['apparent_c_index_raw']:.4f}")
    print(f"    Optimism Corrected:  {cv_results['bootstrap_results']['optimism_corrected_raw']:.4f}")
    print(f"  {'After Isotonic:':<45}")
    print(f"    Apparent:            {cv_results['bootstrap_results']['apparent_c_index_iso']:.4f}")
    print(f"    Optimism Corrected:  {cv_results['bootstrap_results']['optimism_corrected_iso']:.4f}")
    print(f"  {'After KM:':<45}")
    print(f"    Apparent:            {cv_results['bootstrap_results']['apparent_c_index_km']:.4f}")
    print(f"    Optimism Corrected:  {cv_results['bootstrap_results']['optimism_corrected_km']:.4f}")
    print(f"  {'After Time-Aware:':<45}")
    print(f"    Apparent:            {cv_results['bootstrap_results']['apparent_c_index_time']:.4f}")
    print(f"    Optimism Corrected:  {cv_results['bootstrap_results']['optimism_corrected_time']:.4f}")
    
    print(f"{'='*80}\n")

    suffix = f"{threshold}m" if subgroup == 'Overall' else f"{threshold}m_{subgroup}"
    output_dir = paths['output_dir']
    
    results_df = pd.DataFrame({
        'Method': [
            'Nested_Overall_Final',
            'Nested_Mean_Raw', 'Nested_Mean_Iso', 'Nested_Mean_KM', 'Nested_Mean_Time', 'Nested_Mean_Final',

            'Repeated_Mean_Raw', 'Repeated_Mean_Iso', 'Repeated_Mean_KM', 'Repeated_Mean_Time', 'Repeated_Mean_Final',

            'Bootstrap_Apparent_Raw', 'Bootstrap_Corrected_Raw',
            'Bootstrap_Apparent_Iso', 'Bootstrap_Corrected_Iso',
            'Bootstrap_Apparent_KM', 'Bootstrap_Corrected_KM',
            'Bootstrap_Apparent_Time', 'Bootstrap_Corrected_Time'
        ],
        'C_Index': [

            cv_results['nested_results']['overall_c_index_final'],
            cv_results['nested_results']['mean_c_index_raw'],
            cv_results['nested_results']['mean_c_index_iso'],
            cv_results['nested_results']['mean_c_index_km'],
            cv_results['nested_results']['mean_c_index_time'],
            cv_results['nested_results']['mean_c_index_final'],

            cv_results['repeated_results']['mean_c_index_raw'],
            cv_results['repeated_results']['mean_c_index_iso'],
            cv_results['repeated_results']['mean_c_index_km'],
            cv_results['repeated_results']['mean_c_index_time'],
            cv_results['repeated_results']['mean_c_index_final'],

            cv_results['bootstrap_results']['apparent_c_index_raw'],
            cv_results['bootstrap_results']['optimism_corrected_raw'],
            cv_results['bootstrap_results']['apparent_c_index_iso'],
            cv_results['bootstrap_results']['optimism_corrected_iso'],
            cv_results['bootstrap_results']['apparent_c_index_km'],
            cv_results['bootstrap_results']['optimism_corrected_km'],
            cv_results['bootstrap_results']['apparent_c_index_time'],
            cv_results['bootstrap_results']['optimism_corrected_time']
        ],
        'Std': [

            np.nan,
            cv_results['nested_results']['std_c_index_raw'],
            cv_results['nested_results']['std_c_index_iso'],
            cv_results['nested_results']['std_c_index_km'],
            cv_results['nested_results']['std_c_index_time'],
            cv_results['nested_results']['std_c_index_final'],

            cv_results['repeated_results']['std_c_index_raw'],
            cv_results['repeated_results']['std_c_index_iso'],
            cv_results['repeated_results']['std_c_index_km'],
            cv_results['repeated_results']['std_c_index_time'],
            cv_results['repeated_results']['std_c_index_final'],

            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        ],
        'Threshold': threshold,
        'Subgroup': subgroup
    })
    
    results_file = os.path.join(output_dir, f'comprehensive_cv_results_{suffix}.csv')
    results_df.to_csv(results_file, index=False)
    print(f" Saved comprehensive CV results to: {results_file}")

    return cv_results
    
def filter_subgroup(df, subgroup_name):
    if subgroup_name == 'Overall' or SUBGROUPS[subgroup_name] is None:
        return df.copy().reset_index(drop=True)
    
    conditions = SUBGROUPS[subgroup_name]
    mask = pd.Series([True] * len(df), index=df.index)
    
    for col, val in conditions.items():
        if col in df.columns:
            mask &= (df[col] == val)
        else:
            print(f"Warning: Column '{col}' not found in dataframe for subgroup '{subgroup_name}'")
            return pd.DataFrame()  
    
    filtered_df = df[mask].copy()
    filtered_df = filtered_df.reset_index(drop=True)
    
    print(f"Subgroup {subgroup_name}: {len(filtered_df)} samples (from {len(df)} total)")
    
    return filtered_df

def get_threshold_paths(threshold, subgroup='Overall'):
    if subgroup == 'Overall':
        output_dir = os.path.join(BASE_OUTPUT_DIR, f"threshold_{threshold}m")
        weights_dir = os.path.join(BASE_WEIGHTS_DIR, f"threshold_{threshold}m")
    else:
        output_dir = os.path.join(BASE_OUTPUT_DIR, f"threshold_{threshold}m_{subgroup}")
        weights_dir = os.path.join(BASE_WEIGHTS_DIR, f"threshold_{threshold}m_{subgroup}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    
    suffix = f"{threshold}m" if subgroup == 'Overall' else f"{threshold}m_{subgroup}"
    
    paths = {
        'output_dir': output_dir,
        'weights_dir': weights_dir,
        'subgroup': subgroup,
        'threshold': threshold,
        'input_file': os.path.join(output_dir, f"optimized_survival_probabilities_{suffix}.csv"),
        'model_file': os.path.join(weights_dir, f'optimized_survival_predictor_{suffix}.keras'),
        'preprocessor_file': os.path.join(weights_dir, f'optimized_preprocessor_{suffix}.pkl'),
        'feature_file': os.path.join(weights_dir, f'optimized_feature_cols_{suffix}.pkl'),
        'target_col': f'Pseudo_Label_{suffix}',
        'calibrated_col': f'Pseudo_Label_Calibrated_{suffix}',
        'expected_time_col': f'Expected_Survival_Time_{suffix}'
    }
    
    return paths

def calculate_c_index(event, time, predictions, threshold):
    
    try:
        risk_scores = 1 - predictions
        y_structured = np.array([(bool(e), t) for e, t in zip(event, time)],
                              dtype=[('event', '?'), ('time', '<f8')])
        cindex = concordance_index_censored(y_structured['event'], y_structured['time'], risk_scores)
        return cindex[0]
    except Exception as e:
        print(f"Error calculating C-index: {e}")
        return 0.5


def create_generalized_pseudo_labels_corrected(df, time_col, event_col, target_col, expected_time_col, threshold):

    print(f"\n{'='*70}")
    print(f"CREATING INITIAL PSEUDO-LABELS (LEAVE-ONE-OUT METHOD)")
    print(f"  Threshold: {threshold} months")
    print(f"{'='*70}")
    
    n = len(df)

    kmf_full = KaplanMeierFitter()
    kmf_full.fit(df[time_col], df[event_col])
    
    try:
        S_all = kmf_full.predict(threshold)
        if np.isnan(S_all):
            S_all = kmf_full.survival_function_.iloc[-1].values[0]
    except:
        S_all = kmf_full.survival_function_.iloc[-1].values[0]
    
    max_observed_time = df[time_col].max()
    
    print(f"\n  Dataset Statistics:")
    print(f"    Total samples: {n}")
    print(f"    Events: {df[event_col].sum()} ({df[event_col].sum()/n*100:.1f}%)")
    print(f"    Censored: {n - df[event_col].sum()} ({(n - df[event_col].sum())/n*100:.1f}%)")
    print(f"    Max observed time: {max_observed_time:.2f} months")
    print(f"    Threshold: {threshold} months")
    print(f"    Overall KM survival at threshold: {S_all:.4f}")
    
    print(f"\n  Computing leave-one-out pseudo-observations...")
    print(f"    (This may take a moment for n={n} samples)")
    
    pseudo_labels = np.zeros(n)
    
    progress_step = max(1, n // 10)
    
    for i in range(n):
        if i % progress_step == 0:
            print(f"    Progress: {i}/{n} ({i/n*100:.0f}%)", end='\r')
        
        
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        
        kmf_loo = KaplanMeierFitter()
        kmf_loo.fit(df[time_col].iloc[mask], df[event_col].iloc[mask])
        
        try:
            S_loo = kmf_loo.predict(threshold)
            if np.isnan(S_loo):
                S_loo = kmf_loo.survival_function_.iloc[-1].values[0]
        except:
            S_loo = kmf_loo.survival_function_.iloc[-1].values[0]
        
        
        pseudo_obs = n * S_all - (n - 1) * S_loo
        
        
        pseudo_labels[i] = np.clip(pseudo_obs, 0.001, 0.999)
    
    print(f"    Progress: {n}/{n} (100%) - COMPLETE")
    
    df[target_col] = pseudo_labels
    
    df[expected_time_col] = df[time_col].copy()
    
    censored_mask = df[event_col] == 0
    if censored_mask.sum() > 0:

        remaining = np.maximum(threshold - df.loc[censored_mask, time_col], 0)
        df.loc[censored_mask, expected_time_col] = (
            df.loc[censored_mask, time_col] + 
            remaining * df.loc[censored_mask, target_col]
        )
    
    df[expected_time_col] = np.maximum(df[expected_time_col], df[time_col])
    
    deceased_mask = df[event_col] == 1
    
    print(f"\n  Pseudo-Label Statistics:")
    print(f"    Mean:   {pseudo_labels.mean():.4f}")
    print(f"    Std:    {pseudo_labels.std():.4f}")
    print(f"    Min:    {pseudo_labels.min():.4f}")
    print(f"    Max:    {pseudo_labels.max():.4f}")
    print(f"    Median: {np.median(pseudo_labels):.4f}")
    
    if deceased_mask.sum() > 0:
        deceased_mean = df.loc[deceased_mask, target_col].mean()
        print(f"\n    Deceased patients (n={deceased_mask.sum()}):")
        print(f"      Mean: {deceased_mean:.4f}")
        print(f"      Std:  {df.loc[deceased_mask, target_col].std():.4f}")
    
    if censored_mask.sum() > 0:
        censored_mean = df.loc[censored_mask, target_col].mean()
        print(f"\n    Censored patients (n={censored_mask.sum()}):")
        print(f"      Mean: {censored_mean:.4f}")
        print(f"      Std:  {df.loc[censored_mask, target_col].std():.4f}")
    
    if deceased_mask.sum() > 0 and censored_mask.sum() > 0:
        separation = abs(censored_mean - deceased_mean)
        print(f"\n    Separation (Censored - Deceased): {separation:.4f}")
        
        if separation < 0.05:
            print(f"Very low separation")
        elif separation > 0.7:
            print(f"Very high separation")
        else:
            print(f"Separation in reasonable range")
    
    try:
        c_index_init = concordance_index_censored(
            df[event_col].astype(bool).values,
            df[time_col].values,
            1 - pseudo_labels  
        )[0]
        
        print(f"\n  {'='*60}")
        print(f"  INITIAL C-INDEX (BEFORE ANY REFINEMENT): {c_index_init:.4f}")
        print(f"  {'='*60}")
        
        if c_index_init > 0.90:
            print(f"     CRITICAL ERROR: C-index = {c_index_init:.4f} is TOO HIGH!")
            print(f"     This indicates DATA LEAKAGE in pseudo-label generation.")
            print(f"     Expected range for initial pseudo-observations: 0.55-0.75")
            print(f"     STOPPING to prevent invalid results.")
            raise ValueError("Data leakage detected in initial pseudo-labels")
            
        elif c_index_init > 0.85:
            print(f"     WARNING: C-index = {c_index_init:.4f} is suspiciously high")
            print(f"     Expected range: 0.55-0.75")
            print(f"     Proceeding with caution...")
            
        elif c_index_init < 0.50:
            print(f"     WARNING: C-index = {c_index_init:.4f} is below random (0.5)")
            print(f"     This may indicate a problem with pseudo-label generation")
            
        elif 0.55 <= c_index_init <= 0.80:
            print(f"     VALIDATION PASSED: C-index in expected/excellent range")
            print(f"     Initial pseudo-observations are reasonable")
            
        else:
            print(f"     C-index = {c_index_init:.4f}")
            print(f"     Slightly outside typical range (0.55-0.75) but may be acceptable")
            
    except Exception as e:
        print(f"\n     Could not calculate initial C-index: {e}")
        c_index_init = None
    
    print(f"\n  {'='*60}")
    print(f"  PSEUDO-LABEL CREATION COMPLETE")
    print(f"  {'='*60}\n")
    
    return df


def validate_initial_pseudolabels(df, target_col, event_col, time_col, threshold):

    print(f"\n{'='*70}")
    print(f"VALIDATING INITIAL PSEUDO-LABELS")
    print(f"{'='*70}")
    
    pseudo_labels = df[target_col].values
    deceased_mask = df[event_col] == 1
    censored_mask = df[event_col] == 0
    
    validation_passed = True
    
    mean_prob = pseudo_labels.mean()
    std_prob = pseudo_labels.std()
    
    print(f"\n  1. DISTRIBUTION CHECK:")
    print(f"     Mean: {mean_prob:.4f}")
    print(f"     Std:  {std_prob:.4f}")
    
    if mean_prob < 0.05:
        print(f"       Mean too low (< 0.05)")
        validation_passed = False
    elif mean_prob > 0.95:
        print(f"       Mean too high (> 0.95)")
        validation_passed = False
    else:
        print(f"      Mean in reasonable range")
    
    if std_prob < 0.01:
        print(f"       Very low variance - pseudo-labels may be uninformative")
    

    if deceased_mask.sum() > 0 and censored_mask.sum() > 0:
        deceased_mean = df.loc[deceased_mask, target_col].mean()
        censored_mean = df.loc[censored_mask, target_col].mean()
        separation = abs(censored_mean - deceased_mean)
        
        print(f"\n  2. GROUP SEPARATION CHECK:")
        print(f"     Deceased mean: {deceased_mean:.4f}")
        print(f"     Censored mean: {censored_mean:.4f}")
        print(f"     Separation:    {separation:.4f}")
        
        if separation > 0.75:
            print(f"      CRITICAL: Separation too large ")
            validation_passed = False
        elif separation > 0.60:
            print(f"       Separation high ")
        elif separation < 0.05:
            print(f"       Very low separation ")
        else:
            print(f"      Separation in reasonable range (0.05-0.60)")
    

    try:
        c_index = concordance_index_censored(
            df[event_col].astype(bool).values,
            df[time_col].values,
            1 - pseudo_labels
        )[0]
        
        print(f"\n  3. C-INDEX CHECK (CRITICAL):")
        print(f"     C-index: {c_index:.4f}")
        
        if c_index > 0.90:
            print(f"       CRITICAL ERROR: C-index > 0.90 - DATA LEAKAGE CONFIRMED")
            print(f"        Expected range: 0.55-0.75 for initial pseudo-observations")
            validation_passed = False
        elif c_index > 0.85:
            print(f"       WARNING: C-index > 0.85 - suspiciously high")
            print(f"        Expected range: 0.55-0.75")
        elif c_index < 0.50:
            print(f"       WARNING: C-index < 0.50 - below random performance")
            validation_passed = False
        elif 0.55 <= c_index <= 0.80:
            print(f"      EXCELLENT: C-index in expected/good range for initial pseudo-obs")
        else:
            print(f"      C-index outside typical range but may be acceptable")
            
    except Exception as e:
        print(f"\n  3. C-INDEX CHECK: Failed to compute ({e})")
        validation_passed = False
    
    print(f"\n  {'='*60}")
    if validation_passed:
        print(f"   VALIDATION PASSED - Pseudo-labels appear valid")
    else:
        print(f"   VALIDATION FAILED - Critical issues detected")
        print(f"     DO NOT PROCEED - Fix pseudo-label generation first")
    print(f"  {'='*60}\n")
    
    return validation_passed


def create_predicted_survival_curve_with_overlap(df, time_col, calibrated_col, event_col, threshold):

    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    
    max_observed_time = df[time_col].max()
    curve_endpoint = max(threshold, max_observed_time)
    
    n_points_observed = 200
    n_points_extrapolate = 100
    
    time_points_observed = np.linspace(0, max_observed_time, n_points_observed)
    
    if curve_endpoint > max_observed_time:
        time_points_extrapolate = np.linspace(max_observed_time, curve_endpoint, n_points_extrapolate)[1:]
    else:
        time_points_extrapolate = np.array([])
    
    time_points = np.concatenate([time_points_observed, time_points_extrapolate])
    predicted_survival = np.zeros(len(time_points))
    
    for idx, t in enumerate(time_points_observed):
        if t == 0:
            predicted_survival[idx] = 1.0
        else:
            predicted_survival[idx] = kmf.predict(t)
    
    if len(time_points_extrapolate) > 0:
        surv_at_max_obs = kmf.predict(max_observed_time)
        
        censored_mask = df[event_col] == 0
        if censored_mask.sum() > 0:
            censored_probs = df.loc[censored_mask, calibrated_col].values
            censored_times = df.loc[censored_mask, time_col].values
            
            time_weights = np.exp(-(max_observed_time - censored_times) / (max_observed_time / 3))
            time_weights = time_weights / time_weights.sum()
            
            weighted_prob = np.average(censored_probs, weights=time_weights)
        else:
            weighted_prob = surv_at_max_obs * 0.8
        
        time_beyond = time_points_extrapolate - max_observed_time
        if threshold > max_observed_time:
            decay_rate = -np.log(max(weighted_prob, 0.01)) / (threshold - max_observed_time)
        else:
            decay_rate = 0.01
        
        for idx, t_extra in enumerate(time_beyond):
            surv_extrapolate = surv_at_max_obs * np.exp(-decay_rate * t_extra)
            predicted_survival[n_points_observed + idx] = surv_extrapolate
    
    for i in range(1, len(predicted_survival)):
        if predicted_survival[i] > predicted_survival[i-1]:
            predicted_survival[i] = predicted_survival[i-1]
    
    if len(time_points_extrapolate) > 0:
        from scipy.ndimage import gaussian_filter1d
        extrapolate_portion = predicted_survival[n_points_observed:]
        smoothed_extrapolate = gaussian_filter1d(extrapolate_portion, sigma=1.0)
        predicted_survival[n_points_observed:] = smoothed_extrapolate
    
    predicted_survival[0] = 1.0
    predicted_survival = np.clip(predicted_survival, 0, 1)
    
    km_at_max = kmf.predict(max_observed_time)
    pred_at_max = predicted_survival[n_points_observed-1]
    
    return time_points, predicted_survival

def calculate_km_calibration_loss_v2(df, time_col, event_col, calibrated_col, threshold):

    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    
    time_points, predicted_survival = create_predicted_survival_curve_with_overlap(
        df, time_col, calibrated_col, event_col, threshold
    )
    
    max_observed_time = df[time_col].max()
    eval_times = np.linspace(0, max_observed_time, 30)
    
    observed_survival = np.array([kmf.predict(t) for t in eval_times])
    
    pred_interp = interp1d(time_points, predicted_survival, kind='linear', fill_value='extrapolate')
    predicted_at_eval = pred_interp(eval_times)
    
    loss_observed = np.mean((observed_survival - predicted_at_eval) ** 2)
    
    if threshold > max_observed_time:
        eval_times_extra = np.linspace(max_observed_time, threshold, 20)[1:]
        predicted_at_extra = pred_interp(eval_times_extra)
        
        monotonicity_violations = np.sum(np.diff(predicted_at_extra) > 0)
        monotonicity_penalty = monotonicity_violations * 0.01
        
        unrealistic_penalty = 0
        if predicted_at_extra[-1] > 0.95:
            unrealistic_penalty += 0.01
        if predicted_at_extra[-1] < 0.01:
            unrealistic_penalty += 0.01
        
        total_loss = loss_observed + monotonicity_penalty + unrealistic_penalty
    else:
        total_loss = loss_observed
    
    return total_loss, eval_times, observed_survival, predicted_at_eval


def calculate_survival_probabilities_at_threshold(df, time_col, event_col, calibrated_col, threshold):

    from scipy.interpolate import interp1d
    
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    
    max_observed_time = df[time_col].max()

    evaluation_time = threshold  

    try:
        observed_surv_at_threshold = kmf.predict(threshold)
        if np.isnan(observed_surv_at_threshold):
          
            observed_surv_at_threshold = kmf.survival_function_.iloc[-1].values[0]
    except:
        observed_surv_at_threshold = np.nan

    time_points, predicted_survival = create_predicted_survival_curve_with_overlap(
        df, time_col, calibrated_col, event_col, threshold
    )

    pred_curve_interp = interp1d(
        time_points, predicted_survival, 
        kind='linear', 
        fill_value='extrapolate'
    )

    predicted_surv_at_threshold = float(pred_curve_interp(threshold))
    predicted_surv_at_threshold = np.clip(predicted_surv_at_threshold, 0, 1)
    
    predicted_surv_at_max_obs = float(pred_curve_interp(max_observed_time))
    predicted_surv_at_max_obs = np.clip(predicted_surv_at_max_obs, 0, 1)
    
    event_mask = df[event_col] == 1
    censored_mask = df[event_col] == 0
    
    censored_surv_prob = df.loc[censored_mask, calibrated_col].mean() if censored_mask.sum() > 0 else np.nan
    deceased_surv_prob = df.loc[event_mask, calibrated_col].mean() if event_mask.sum() > 0 else np.nan
    
    high_surv_count = (df[calibrated_col] > 0.70).sum()
    high_surv_pct = (high_surv_count / len(df)) * 100
    
    low_surv_count = (df[calibrated_col] < 0.30).sum()
    low_surv_pct = (low_surv_count / len(df)) * 100
    

    results = {
        
        'threshold': threshold,
        'max_observed_time': max_observed_time,
        'evaluation_time': threshold,  
        'used_max_observed': False,  
        'observed_surv_at_eval': observed_surv_at_threshold * 100 if not np.isnan(observed_surv_at_threshold) else np.nan,
        'predicted_surv_at_eval': predicted_surv_at_threshold * 100,  
        'difference': abs(predicted_surv_at_threshold * 100 - observed_surv_at_threshold * 100) if not np.isnan(observed_surv_at_threshold) else np.nan,
        'predicted_surv_at_max_obs': predicted_surv_at_max_obs * 100,
        'observed_surv_at_max_obs': (kmf.predict(max_observed_time) * 100) if max_observed_time <= kmf.survival_function_.index.max() else np.nan,
        'censored_surv_prob': censored_surv_prob * 100 if not np.isnan(censored_surv_prob) else np.nan,
        'deceased_surv_prob': deceased_surv_prob * 100 if not np.isnan(deceased_surv_prob) else np.nan,
        'high_surv_count': high_surv_count,
        'high_surv_pct': high_surv_pct,
        'low_surv_count': low_surv_count,
        'low_surv_pct': low_surv_pct,
        'mean_individual_prob': df[calibrated_col].mean() * 100
    }
    
    return results
    
# ============================================================================
# Plot function predicted curve
# ============================================================================

def plot_observed_vs_predicted_km_curves(df, iteration, time_col, event_col, 
                                        calibrated_col, threshold, output_dir, subgroup='Overall'):

    print(f"\n{'='*60}")
    print(f"Generating KM Curves - {subgroup} - Threshold {threshold}m ")
    print(f"{'='*60}")
    
    time_observed = df[time_col].values
    event_observed = df[event_col].values
    max_observed_time = time_observed.max()
    
    kmf_observed = KaplanMeierFitter()
    kmf_observed.fit(time_observed, event_observed, label='Observed')
    
    time_points, predicted_survival = create_predicted_survival_curve_with_overlap(
        df, time_col, calibrated_col, event_col, threshold
    )

    km_loss, eval_times, obs_at_eval, pred_at_eval = calculate_km_calibration_loss_v2(
        df, time_col, event_col, calibrated_col, threshold
    )
    
    try:
        obs_median = kmf_observed.median_survival_time_
        
        if not np.isnan(obs_median) and obs_median > 0:
            kmf_ci = kmf_observed.confidence_interval_survival_function_
            
            lower_col = kmf_ci.columns[0]  
            upper_col = kmf_ci.columns[1]  
            
            lower_surv = kmf_ci[lower_col]
            lower_idx = np.where(lower_surv.values <= 0.5)[0]
            if len(lower_idx) > 0:
                obs_median_lower = lower_surv.index[lower_idx[0]]
            else:
                obs_median_lower = obs_median * 0.85
            
            upper_surv = kmf_ci[upper_col]
            upper_idx = np.where(upper_surv.values <= 0.5)[0]
            if len(upper_idx) > 0:
                obs_median_upper = upper_surv.index[upper_idx[0]]
            else:
                obs_median_upper = obs_median * 1.15
        else:
            obs_median = np.nan
            obs_median_lower = np.nan
            obs_median_upper = np.nan
    except Exception as e:
        obs_median = kmf_observed.median_survival_time_
        if not np.isnan(obs_median) and obs_median > 0:
            obs_median_lower = obs_median * 0.85
            obs_median_upper = obs_median * 1.15
        else:
            obs_median = np.nan
            obs_median_lower = np.nan
            obs_median_upper = np.nan
    
    try:
        pred_median_idx = np.where(predicted_survival <= 0.5)[0]
        if len(pred_median_idx) > 0:
            pred_median = time_points[pred_median_idx[0]]
            
            curve_std = np.std(predicted_survival[max(0, pred_median_idx[0]-10):min(len(predicted_survival), pred_median_idx[0]+10)])
            pred_ci_range = curve_std * 20  
            
            pred_median_lower = max(0, pred_median - pred_ci_range)
            pred_median_upper = pred_median + pred_ci_range
            
            pred_median_lower = min(pred_median_lower, pred_median)
        else:
            pred_median = np.nan
            pred_median_lower = np.nan
            pred_median_upper = np.nan
    except Exception as e:
        pred_median = np.nan
        pred_median_lower = np.nan
        pred_median_upper = np.nan
    
    fig = plt.figure(figsize=(20, 8), dpi=600)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[:, :2])
    
    kmf_observed.plot_survival_function(ax=ax1, ci_show=True, color='#FF6B6B', 
                                       linewidth=3, label='Observed (KM)', alpha=0.8)
    
    ax1.plot(time_points, predicted_survival, color='#4ECDC4', 
            linewidth=3, linestyle='--', label='Predicted', alpha=0.9)
    
    ax1.axvline(x=max_observed_time, color='gray', linestyle=':', 
               linewidth=2, alpha=0.5, label=f'Max Observed: {max_observed_time:.1f}m')
    
    if abs(max_observed_time - threshold) > 1:  
        ax1.axvline(x=threshold, color='purple', linestyle='-.', 
                   linewidth=2.5, alpha=0.6, label=f'Threshold: {threshold}m')
    
    if not np.isnan(obs_median) and obs_median > 0:
        ax1.axvline(x=obs_median, color='#FF6B6B', linestyle=':', 
                   linewidth=2.5, alpha=0.8, label=f'Observed Median: {obs_median:.1f} ({obs_median_lower:.2f}, {obs_median_upper:.2f}) months')
    
    if not np.isnan(pred_median) and pred_median > 0:
        ax1.axvline(x=pred_median, color='#4ECDC4', linestyle=':', 
                   linewidth=2.5, alpha=0.8, label=f'Predicted Median: {pred_median:.1f} ({pred_median_lower:.2f}, {pred_median_upper:.2f}) months')
    
    ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    ax1.set_xlabel('Time (months)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Survival Probability', fontsize=14, fontweight='bold')
    
    SUBGROUP_DISPLAY_NAMES = {
        'Overall': 'Overall',
        'C1': 'TPS < 1%',
        'C2': 'TPS 1-49%', 
        'C3': 'TPS = 50%'
    }
    
    display_name = SUBGROUP_DISPLAY_NAMES.get(subgroup, subgroup)
    title_text = f'Observed vs Predicted Survival Curves - {display_name}\n(Threshold: {threshold}m - KM Loss: {km_loss:.6f})'
    ax1.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    plot_xlim = max(threshold, max_observed_time) * 1.15
    ax1.set_xlim(0, plot_xlim)
    ax1.set_ylim(-0.02, 1.05)
    
    survival_probs = df[calibrated_col].values
    beyond_threshold_count = (df[time_col] >= threshold).sum()
    stats_text = f'Subgroup: {subgroup}\n'
    stats_text += f'Threshold: {threshold} months\n'
    stats_text += f'Max Observed: {max_observed_time:.1f} months\n'
    if max_observed_time > threshold:
        stats_text += f'Beyond threshold: {beyond_threshold_count}\n'
    stats_text += f'Observed Events: {event_observed.sum()}/{len(event_observed)}\n'
    stats_text += f'Mean Survival Prob: {survival_probs.mean():.3f}'
    ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2.5, alpha=0.7, label='Perfect Calibration')
    scatter = ax2.scatter(obs_at_eval, pred_at_eval, c=eval_times, cmap='viridis',
                         s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    for i, t in enumerate(eval_times[::10]):
        idx = i * 10
        if idx < len(eval_times):
            ax2.annotate(f'{int(t)}m', (obs_at_eval[idx], pred_at_eval[idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Observed Survival', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Survival', fontsize=12, fontweight='bold')
    ax2.set_title('Time-wise Calibration', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Time (months)', fontsize=10)
    
    ax3 = fig.add_subplot(gs[1, 2])
    abs_diff = np.abs(obs_at_eval - pred_at_eval)
    ax3.plot(eval_times, abs_diff, 'o-', color='#E74C3C', linewidth=2, markersize=6)
    ax3.fill_between(eval_times, 0, abs_diff, alpha=0.3, color='#E74C3C')
    ax3.axhline(y=0.05, color='orange', linestyle='--', linewidth=2, 
               label='5% threshold', alpha=0.7)
    ax3.axhline(y=0.10, color='red', linestyle='--', linewidth=2, 
               label='10% threshold', alpha=0.7)
    
    ax3.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('|Observed - Predicted|', fontsize=12, fontweight='bold')
    ax3.set_title('Absolute Difference', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max(eval_times))
    
    mean_diff = np.mean(abs_diff)
    ax3.text(0.98, 0.98, f'Mean |Diff|: {mean_diff:.4f}', 
            transform=ax3.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    suffix = f"{threshold}m" if subgroup == 'Overall' else f"{threshold}m_{subgroup}"
    plt.savefig(os.path.join(output_dir, f'km_curves_comparison_{suffix}.png'), 
                dpi=600, bbox_inches='tight')
    plt.close()
    
    return {
        'obs_median': obs_median,
        'obs_median_lower': obs_median_lower,
        'obs_median_upper': obs_median_upper,
        'pred_median': pred_median,
        'pred_median_lower': pred_median_lower,
        'pred_median_upper': pred_median_upper,
        'km_loss': km_loss,
        'mean_abs_diff': mean_diff,
        'predicted_survival_at_0': predicted_survival[0]
    }

def calculate_threshold_specific_importance(models, preprocessor, X, y):

    from sklearn.metrics import mean_absolute_error
    
    X_processed = preprocessor.transform(X)
    
    baseline_preds = np.mean([model.predict(X_processed, verbose=0).flatten() for model in models], axis=0)
    baseline_mae = mean_absolute_error(y, baseline_preds)

    feature_names = []
    feature_to_original_map = {}
    
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    for feat in numeric_features:
        feature_names.append(feat)
        feature_to_original_map[feat] = feat
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features:
        try:
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
            for cat_feat in cat_feature_names:
                feature_names.append(cat_feat)
                orig_feat = cat_feat.split('_')[0] if '_' in cat_feat else cat_feat
                feature_to_original_map[cat_feat] = orig_feat
        except:
            for feat in categorical_features:
                feature_names.append(feat)
                feature_to_original_map[feat] = feat
    
    n_repeats = 3
    feature_importance = {fname: [] for fname in feature_names}
    
    for repeat in range(n_repeats):
        for idx, feature_name in enumerate(feature_names):
            X_permuted = X_processed.copy()
            np.random.seed(42 + repeat)
            X_permuted[:, idx] = np.random.permutation(X_permuted[:, idx])
            
            permuted_preds = np.mean([model.predict(X_permuted, verbose=0).flatten() for model in models], axis=0)
            permuted_mae = mean_absolute_error(y, permuted_preds)
            
            feature_importance[feature_name].append(permuted_mae - baseline_mae)
    
    feature_importance_avg = {k: np.mean(v) for k, v in feature_importance.items()}
    
    original_feature_importance = {}
    for feat_name, importance in feature_importance_avg.items():
        orig_feat = feature_to_original_map.get(feat_name, feat_name)
        if orig_feat in original_feature_importance:
            original_feature_importance[orig_feat] += abs(importance)
        else:
            original_feature_importance[orig_feat] = abs(importance)
    
    return original_feature_importance

def calculate_generalized_feature_importance(all_results, output_dir, subgroup='Overall'):

    print(f"\n{'='*70}")
    print(f"Calculating GENERALIZED Feature Importance - {subgroup}")
    print(f"{'='*70}")
    
    time_feature_keywords = ['time', 'log_time', 'squared', 'cubic', 'sqrt', 'reciprocal', 
                            'normalized', 'progression', 'remaining', 'ratio', 'decay', 'bins',
                            'event_interaction']
    
    all_threshold_importances = []
    valid_thresholds = []
    
    for threshold in THRESHOLDS:
        key = (threshold, subgroup)
        if key not in all_results or all_results[key] is None:
            print(f"  Skipping threshold {threshold}m - no results available")
            continue
        
        paths = get_threshold_paths(threshold, subgroup)
        weights_dir = paths['weights_dir']
        calibrated_col = paths['calibrated_col']
        
        try:

            models = []
            for i in range(ENSEMBLE_SIZE):
                model_path = os.path.join(weights_dir, f'best_model_{i}.keras')
                if os.path.exists(model_path):
                    model = load_model(model_path, compile=False)
                    models.append(model)
            
            if len(models) == 0:
                print(f"  Skipping threshold {threshold}m - no saved models")
                continue
            
            preprocessor_path = os.path.join(weights_dir, 'best_preprocessor.pkl')
            if not os.path.exists(preprocessor_path):
                print(f"  Skipping threshold {threshold}m - no preprocessor")
                continue
            preprocessor = joblib.load(preprocessor_path)
            
            suffix = f"{threshold}m" if subgroup == 'Overall' else f"{threshold}m_{subgroup}"
            feature_file = os.path.join(weights_dir, f'optimized_feature_cols_{suffix}.pkl')
            if not os.path.exists(feature_file):
                print(f"  Skipping threshold {threshold}m - no feature file")
                continue
            feature_cols = joblib.load(feature_file)
            
            df = all_results[key]
            X = df[feature_cols]
            y = df[calibrated_col]
            
            print(f"  Processing threshold {threshold}m...")
            threshold_importance = calculate_threshold_specific_importance(models, preprocessor, X, y)
            
            all_threshold_importances.append(threshold_importance)
            valid_thresholds.append(threshold)
            
            for model in models:
                del model
            tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"  Error at threshold {threshold}m: {e}")
            continue
    
    if len(all_threshold_importances) == 0:
        print("  No valid data for generalized importance")
        return None
    
    print(f"\n  Aggregating across {len(valid_thresholds)} thresholds...")
    
    all_features = set()
    for imp_dict in all_threshold_importances:
        all_features.update(imp_dict.keys())
    
    aggregated_importance = {}
    for feature in all_features:
        importances = [imp_dict[feature] for imp_dict in all_threshold_importances if feature in imp_dict]
        if importances:
            aggregated_importance[feature] = np.mean(importances)
    
    clinical_importance = {}
    engineered_importance = {}
    
    for feat, importance in aggregated_importance.items():
        if any(kw in feat.lower() for kw in time_feature_keywords):
            engineered_importance[feat] = importance
        else:
            clinical_importance[feat] = importance
    
    sorted_clinical = sorted(clinical_importance.items(), key=lambda x: x[1], reverse=True)
    sorted_engineered = sorted(engineered_importance.items(), key=lambda x: x[1], reverse=True)
    
    SUBGROUP_DISPLAY_NAMES = {
        'Overall': 'Overall Population',
        'C1': 'TPS < 1%',
        'C2': 'TPS 1-49%', 
        'C3': 'TPS = 50%'
    }
    display_name = SUBGROUP_DISPLAY_NAMES.get(subgroup, subgroup)
    thresholds_str = ', '.join([f'{t}m' for t in valid_thresholds])
    
    fig = plt.figure(figsize=(20, 16), dpi=300)
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    if sorted_clinical:
        names = [f[0] for f in sorted_clinical]
        vals = [f[1] for f in sorted_clinical]
        total = sum(vals) if sum(vals) > 0 else 1
        pcts = [(v / total) * 100 for v in vals]
        
        x_pos = np.arange(len(names))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
        
        bars = ax1.bar(x_pos, pcts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Clinical Features', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Relative Importance (%)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Clinical Features Importance - {display_name}', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, pct in zip(bars, pcts):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x_pos, np.cumsum(pcts), 'ro-', linewidth=2.5, markersize=8, label='Cumulative %', alpha=0.7)
        ax1_twin.set_ylabel('Cumulative (%)', fontsize=12, fontweight='bold')
        ax1_twin.set_ylim(0, 105)
        ax1_twin.legend(loc='lower right')
        ax1_twin.axhline(y=80, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    
    ax2 = fig.add_subplot(gs[1, 0])
    if sorted_engineered:
        names = [f[0] for f in sorted_engineered]
        vals = [f[1] for f in sorted_engineered]
        total = sum(vals) if sum(vals) > 0 else 1
        pcts = [(v / total) * 100 for v in vals]
        
        x_pos = np.arange(len(names))
        colors = plt.cm.Blues(np.linspace(0.2, 0.8, len(names)))
        
        bars = ax2.bar(x_pos, pcts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Engineered Time Features', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Relative Importance (%)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Engineered Features Importance - {display_name}', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, pct in zip(bars, pcts):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x_pos, np.cumsum(pcts), 'bo-', linewidth=2.5, markersize=8, label='Cumulative %', alpha=0.7)
        ax2_twin.set_ylabel('Cumulative (%)', fontsize=12, fontweight='bold')
        ax2_twin.set_ylim(0, 105)
        ax2_twin.legend(loc='lower right')
        ax2_twin.axhline(y=80, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'feature_importance_generalized_{subgroup}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    clinical_df = pd.DataFrame(sorted_clinical, columns=['Feature', 'Importance'])
    clinical_df['Category'] = 'Clinical'
    engineered_df = pd.DataFrame(sorted_engineered, columns=['Feature', 'Importance'])
    engineered_df['Category'] = 'Engineered'
    combined_df = pd.concat([clinical_df, engineered_df]).sort_values('Importance', ascending=False)
    combined_df['Subgroup'] = subgroup
    combined_df['Thresholds'] = thresholds_str
    combined_df.to_csv(os.path.join(output_dir, f'feature_importance_generalized_{subgroup}.csv'), index=False)
    
    print(f"\n  Top 10 Clinical Features:")
    for i, (feat, imp) in enumerate(sorted_clinical[:10], 1):
        print(f"    {i:2d}. {feat:30s}: {imp:.6f}")
    
    print(f"\n  Saved: feature_importance_generalized_{subgroup}.png/csv")
    
    return {'clinical': clinical_importance, 'engineered': engineered_importance, 'thresholds': valid_thresholds}

def generate_all_generalized_feature_importance(all_results):

    print("\n" + "="*80)
    print("GENERATING GENERALIZED FEATURE IMPORTANCE")
    print("="*80)
    
    results = {}
    for subgroup in SUBGROUPS.keys():
        results[subgroup] = calculate_generalized_feature_importance(all_results, BASE_OUTPUT_DIR, subgroup)
    
    return results

def ranking_preserving_calibration(y_pred, y_true, event, time):

    original_ranking = np.argsort(-y_pred)  
    
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
    y_calibrated = iso.fit_transform(y_pred, y_true)
    
    new_ranking = np.argsort(-y_calibrated)
    ranking_changed = not np.array_equal(original_ranking, new_ranking)
    
    if ranking_changed:

        sorted_idx = np.argsort(y_pred)
        y_pred_sorted = y_pred[sorted_idx]
        y_true_sorted = y_true[sorted_idx]
        
        calibrator = PchipInterpolator(y_pred_sorted, y_true_sorted)
        y_calibrated = calibrator(y_pred)
        y_calibrated = np.clip(y_calibrated, 0.001, 0.999)
    
    return y_calibrated


def km_aware_calibration(df, y_pred, iteration, time_col, event_col, threshold):

    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    
    original_order = np.argsort(-y_pred)
    c_index_before = calculate_c_index(df[event_col], df[time_col], y_pred, threshold)
    
    time_bins = np.percentile(df[time_col], np.linspace(0, 100, 11))
    calibrated = y_pred.copy()
    
    for i in range(len(time_bins) - 1):
        time_mask = (df[time_col] >= time_bins[i]) & (df[time_col] < time_bins[i + 1])
        
        if time_mask.sum() > 10:
            mid_time = (time_bins[i] + time_bins[i + 1]) / 2
            obs_survival_at_mid = kmf.predict(mid_time)
            current_mean = calibrated[time_mask].mean()
            
            adjustment_strength = min(0.15, 0.3 - 0.02 * iteration)
            
            if obs_survival_at_mid > 0.01:
                obs_surv_at_threshold = kmf.predict(threshold)
                target_mean = obs_surv_at_threshold / obs_survival_at_mid
                target_mean = np.clip(target_mean, 0.01, 0.99)
                
                adjustment = (target_mean - current_mean) * adjustment_strength
                calibrated[time_mask] += adjustment  
    
    calibrated = np.clip(calibrated, 0.001, 0.999)

    c_index_after = calculate_c_index(df[event_col], df[time_col], calibrated, threshold)
    
    if c_index_after < c_index_before - 0.02:
        print(f"    KM calibration reduced C-index: {c_index_before:.4f}  {c_index_after:.4f}")
        print(f"     Reducing calibration strength")
        calibrated = 0.7 * y_pred + 0.3 * calibrated
        c_index_final = calculate_c_index(df[event_col], df[time_col], calibrated, threshold)
        print(f"     After rollback: {c_index_final:.4f}")
    
    return calibrated

def enforce_constraints_v2(df, iteration, event_col, target_col, calibrated_col, 
                          time_col, expected_time_col):
    df = df.copy()
    deceased_mask = (df[event_col] == 1)
    censored_mask = (df[event_col] == 0)
    
    if deceased_mask.sum() > 0:
        df.loc[deceased_mask, expected_time_col] = df.loc[deceased_mask, time_col]
        
        max_deceased_prob = max(0.05, 0.15 - 0.01 * min(iteration, 10))
        df.loc[deceased_mask, calibrated_col] = np.minimum(
            df.loc[deceased_mask, calibrated_col], 
            max_deceased_prob
        )
        df.loc[deceased_mask, target_col] = df.loc[deceased_mask, calibrated_col]
    
    if censored_mask.sum() > 0:
        df.loc[censored_mask, expected_time_col] = np.maximum(
            df.loc[censored_mask, expected_time_col],
            df.loc[censored_mask, time_col]
        )
        
        min_censored_prob = 0.20
        df.loc[censored_mask, calibrated_col] = np.maximum(
            df.loc[censored_mask, calibrated_col],
            min_censored_prob
        )
    
    df[calibrated_col] = np.clip(df[calibrated_col], 0.001, 0.999)
    df[target_col] = np.clip(df[target_col], 0.001, 0.999)
    df[expected_time_col] = np.maximum(df[expected_time_col], df[time_col])
    
    return df
# ============================================================================
# Helper functions 
# ============================================================================

def add_time_features(df, time_col, threshold):
    df = df.copy()
    
    df['log_time'] = np.log1p(df[time_col])
    df['time_squared'] = df[time_col] ** 2
    df['time_normalized'] = df[time_col] / threshold
    df['time_cubic'] = df[time_col] ** 3
    df['time_sqrt'] = np.sqrt(df[time_col])
    df['time_reciprocal'] = 1 / (1 + df[time_col])
    
    df['time_progression'] = df[time_col] / (df[time_col] + threshold)
    df['time_remaining'] = (threshold - df[time_col]) / threshold
    df['time_ratio'] = df[time_col] / (df[time_col].max() + 1e-8)
    
    df['time_decay_short'] = np.exp(-df[time_col] / 60)
    df['time_decay_long'] = np.exp(-df[time_col] / threshold)
    
    df['time_event_interaction'] = df[time_col] * df[EVENT_COL]
    df['time_remaining_event'] = df['time_remaining'] * df[EVENT_COL]
    
    df['time_bins'] = pd.cut(df[time_col], bins=np.linspace(0, threshold, 13), labels=False)
    
    return df

def build_time_aware_model(input_dim):

    inputs = Input(shape=(input_dim,))
    
    x = BatchNormalization()(inputs)
    x = Dense(128, activation='relu', kernel_initializer='he_normal', 
              kernel_regularizer=l2(1e-5))(x)
    x = Dropout(0.4)(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-5))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Nadam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    return model

def build_model_ensemble(input_dim, n_models=ENSEMBLE_SIZE):
    models = []
    for _ in range(n_models):
        models.append(build_time_aware_model(input_dim))
    return models


def time_aware_refinement(df, pred_probs, event, time, threshold, iteration=1):

    refined = pred_probs.copy()
    deceased_mask = (event == 1)
    censored_mask = (event == 0)
    
    c_index_before = calculate_c_index(event, time, refined, threshold)
    
    kmf = KaplanMeierFitter()
    kmf.fit(time, event)
    
    if deceased_mask.sum() > 0:
        death_times = time[deceased_mask]
        
        for i, death_time in enumerate(death_times):
            idx = np.where(deceased_mask)[0][i]
            surv_at_death = kmf.predict(death_time)
            time_ratio = death_time / threshold
            base_penalty = min(0.95, 0.6 + 0.07 * iteration)
            time_factor = (1 - surv_at_death) * 0.3
            refined[idx] = refined[idx] * (1 - base_penalty) * time_factor
        
        max_deceased_prob = max(0.05, 0.2 - 0.02 * iteration)
        refined[deceased_mask] = np.minimum(refined[deceased_mask], max_deceased_prob)
    
    if censored_mask.sum() > 0:
        cens_times = time[censored_mask]
        
        for i, cens_time in enumerate(cens_times):
            idx = np.where(censored_mask)[0][i]
            surv_at_cens = kmf.predict(cens_time)
            surv_at_threshold = kmf.predict(threshold)
            
            if surv_at_cens > 0.01:
                km_cond_prob = surv_at_threshold / surv_at_cens
                blend_factor = min(0.5, 0.3 + 0.05 * iteration)
                refined[idx] = blend_factor * km_cond_prob + (1 - blend_factor) * refined[idx]
        
        very_long_term = censored_mask & (time >= threshold * 0.9)
        if very_long_term.sum() > 0:
            refined[very_long_term] = np.maximum(refined[very_long_term], 0.8)
    
    refined = np.clip(refined, 0.001, 0.999)
    
    c_index_after = calculate_c_index(event, time, refined, threshold)
    
    if c_index_after < c_index_before - 0.03:
        print(f"     Time-aware refinement reduced C-index: {c_index_before:.4f}  {c_index_after:.4f}")
        print(f"     Using lighter blend")
        refined = 0.6 * pred_probs + 0.4 * refined
    
    return refined
    
def update_expected_survival_time(df, pred_probs, event, time, expected_time_col, 
                                 time_col, event_col, threshold):

    df = df.copy()
    deceased_mask = (event == 1)
    censored_mask = (event == 0)
    
    if deceased_mask.sum() > 0:
        df.loc[deceased_mask, expected_time_col] = df.loc[deceased_mask, time_col]
    
    if censored_mask.sum() > 0:
        probs_censored = pred_probs[censored_mask]
        time_censored = time[censored_mask]
        beyond_threshold_mask = time_censored >= threshold
        within_threshold_mask = time_censored < threshold
        
        if beyond_threshold_mask.any():
            df.loc[censored_mask & (time >= threshold), expected_time_col] = time[censored_mask & (time >= threshold)]
        
        if within_threshold_mask.any():
            probs_within = probs_censored[within_threshold_mask]
            time_within = time_censored[within_threshold_mask]
            remaining_time = threshold - time_within
            remaining_time = np.maximum(remaining_time, 0)
            time_factor = probs_within ** 0.7
            additional_survival = remaining_time * time_factor
            expected_time_within = time_within + additional_survival
            expected_time_within = np.maximum(expected_time_within, time_within)
            expected_time_within = np.minimum(expected_time_within, threshold)
            
            high_prob_mask = probs_within > 0.7
            if high_prob_mask.any():
                expected_time_within[high_prob_mask] = np.maximum(
                    expected_time_within[high_prob_mask],
                    time_within[high_prob_mask] + remaining_time[high_prob_mask] * 0.9)
            
            df.loc[censored_mask & (time < threshold), expected_time_col] = expected_time_within
    
    df[expected_time_col] = np.maximum(df[expected_time_col], df[time_col])
    return df

def time_dependent_calibration(y_true, y_pred, event, time, time_bins=15):

    calibrated = y_pred.copy()
    time_quantiles = np.unique(np.percentile(time, np.linspace(0, 100, time_bins + 1)))
    
    for i in range(len(time_quantiles) - 1):
        time_mask = (time >= time_quantiles[i]) & (time < time_quantiles[i + 1])
        
        if np.sum(time_mask) > 20:
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
            calibrated_local = iso.fit_transform(y_pred[time_mask], y_true[time_mask])
            sample_ratio = np.sum(time_mask) / len(time)
            blend_factor = min(0.8, 0.3 + 0.5 * sample_ratio)
            calibrated[time_mask] = (blend_factor * calibrated_local + 
                                   (1 - blend_factor) * y_pred[time_mask])
    
    return calibrated


def track_deceased_convergence(df, iteration, event_col, calibrated_col, expected_time_col, time_col, threshold, subgroup='Overall'):
    deceased_mask = (df[event_col] == 1)
    
    if deceased_mask.sum() > 0:
        avg_prob = df.loc[deceased_mask, calibrated_col].mean()
        max_prob = df.loc[deceased_mask, calibrated_col].max()
        time_error = np.mean(np.abs(df.loc[deceased_mask, expected_time_col] - 
                                   df.loc[deceased_mask, time_col]))
        
        print(f"Deceased patients - {subgroup} (Threshold: {threshold}m):")
        print(f"  Avg probability: {avg_prob:.4f}, Max: {max_prob:.4f}")
        print(f"  Avg time error: {time_error:.2f} months")

def analyze_final_results(df, convergence_history, event_col, calibrated_col, time_col, threshold, subgroup='Overall'):
    print("\n" + "="*60)
    print(f"DETAILED FINAL ANALYSIS - {subgroup.upper()} - THRESHOLD: {threshold} MONTHS")
    print("="*60)
    
    event_mask = df[event_col] == 1
    censored_mask = df[event_col] == 0
    
    print(f"\n1. DISTRIBUTION ANALYSIS:")
    print(f"   Subgroup: {subgroup}")
    print(f"   Total samples: {len(df)}")
    print(f"   Events: {event_mask.sum()} ({event_mask.sum()/len(df)*100:.1f}%)")
    print(f"   Censored: {censored_mask.sum()} ({censored_mask.sum()/len(df)*100:.1f}%)")
    
    print(f"\n2. PREDICTION RANGE:")
    print(f"   Min probability: {df[calibrated_col].min():.4f}")
    print(f"   Max probability: {df[calibrated_col].max():.4f}")
    print(f"   Mean probability: {df[calibrated_col].mean():.4f}")
    print(f"   Median probability: {df[calibrated_col].median():.4f}")
    print(f"   Std deviation: {df[calibrated_col].std():.4f}")
    
    print(f"\n3. SURVIVAL PROBABILITY AT END OF STUDY:")
    surv_probs = calculate_survival_probabilities_at_threshold(
        df, time_col, event_col, calibrated_col, threshold
    )
    
    print(f"   {'-'*50}")
    print(f"   EVALUATION TIME: {surv_probs['evaluation_time']:.2f} MONTHS")
    if surv_probs['used_max_observed']:
        print(f"   (Using max observed time: {surv_probs['max_observed_time']:.2f} months > threshold: {threshold} months)")
    else:
        print(f"   (Using threshold: {threshold} months)")
    print(f"   {'-'*50}")
    
    print(f"\n   SURVIVAL AT {surv_probs['evaluation_time']:.2f} MONTHS:")
    print(f"     Observed (Kaplan-Meier):  {surv_probs['observed_surv_at_eval']:.2f}%")
    print(f"     Predicted (Model):        {surv_probs['predicted_surv_at_eval']:.2f}%")
    if not np.isnan(surv_probs['difference']):
        print(f"     Absolute Difference:      {surv_probs['difference']:.2f}%")
    
    print(f"\n   MEAN INDIVIDUAL PATIENT PREDICTIONS:")
    if not np.isnan(surv_probs['censored_surv_prob']):
        print(f"     Censored patients:        {surv_probs['censored_surv_prob']:.2f}%")
    if not np.isnan(surv_probs['deceased_surv_prob']):
        print(f"     Deceased patients:        {surv_probs['deceased_surv_prob']:.2f}%")
    print(f"     Overall mean:             {surv_probs['mean_individual_prob']:.2f}%")
    
    print(f"\n   RISK STRATIFICATION:")
    print(f"     High survival (>70%):     {surv_probs['high_surv_count']} patients ({surv_probs['high_surv_pct']:.1f}%)")
    print(f"     Low survival (<30%):      {surv_probs['low_surv_count']} patients ({surv_probs['low_surv_pct']:.1f}%)")
    print(f"   {'-'*50}")
    
    c_index = calculate_c_index(df[event_col], df[time_col], df[calibrated_col], threshold)
    print(f"\n4. C-INDEX (Discrimination): {c_index:.4f}")
    
    if convergence_history:
        final_km_loss = convergence_history[-1].get('km_loss', 'N/A')
        print(f"\n5. KM CALIBRATION LOSS: {final_km_loss}")
    
    print(f"\n6. ADDITIONAL SURVIVAL METRICS:")
    
    q25 = np.percentile(df[calibrated_col], 25)
    q75 = np.percentile(df[calibrated_col], 75)
    iqr = q75 - q25
    
    print(f"   25th percentile: {q25:.3f}")
    print(f"   75th percentile: {q75:.3f}")
    print(f"   IQR: {iqr:.3f}")
    
    very_high_risk = (df[calibrated_col] < 0.3).sum()
    high_risk = ((df[calibrated_col] >= 0.3) & (df[calibrated_col] < 0.5)).sum()
    moderate_risk = ((df[calibrated_col] >= 0.5) & (df[calibrated_col] < 0.7)).sum()
    low_risk = (df[calibrated_col] >= 0.7).sum()
    
    print(f"\n   DETAILED RISK STRATIFICATION:")
    print(f"     Very High Risk (<30%):    {very_high_risk} ({very_high_risk/len(df)*100:.1f}%)")
    print(f"     High Risk (30-50%):       {high_risk} ({high_risk/len(df)*100:.1f}%)")
    print(f"     Moderate Risk (50-70%):   {moderate_risk} ({moderate_risk/len(df)*100:.1f}%)")
    print(f"     Low Risk (>=70%):          {low_risk} ({low_risk/len(df)*100:.1f}%)")
    
    return surv_probs

# ==================== MAIN EXECUTION ====================

def should_stop_training(convergence_history, patience=5):

    if len(convergence_history) < patience + 1:
        return False, None, None
    

    c_indices = [h['c_index'] for h in convergence_history]
    best_c_index = max(c_indices)
    best_iteration = c_indices.index(best_c_index) + 1  
    recent_c_indices = c_indices[-patience:]
    all_worse = all(c < best_c_index - 0.001 for c in recent_c_indices)
    
    if all_worse:
        avg_recent = np.mean(recent_c_indices)
        degradation = best_c_index - avg_recent
        reason = f"C-index degraded for {patience} consecutive iterations (best: {best_c_index:.4f} at iter {best_iteration}, recent avg: {avg_recent:.4f}, degradation: {degradation:.4f})"
        return True, best_iteration, reason
    
    return False, None, None


def train_for_threshold(threshold, base_df, subgroup='Overall'):
    
    print("\n" + "="*80)
    print(f"{'='*80}")
    print(f"STARTING TRAINING - SUBGROUP: {subgroup} - THRESHOLD: {threshold} MONTHS")
    print(f"{'='*80}")
    print("="*80 + "\n")
    
    df = filter_subgroup(base_df, subgroup)
    
    if len(df) == 0:
        print(f"ERROR: No data available for subgroup '{subgroup}'")
        print("Skipping this subgroup...")
        return None
    
    if len(df) < 30:
        print(f"WARNING: Very small sample size ({len(df)}) for subgroup '{subgroup}'")
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
                df, TIME_COL, EVENT_COL, target_col, expected_time_col, threshold
            )
    else:
        print(f"Creating initial pseudo labels for {subgroup} (leave-one-out method)")
        
        df = create_generalized_pseudo_labels_corrected(
            df, TIME_COL, EVENT_COL, target_col, expected_time_col, threshold
        )
        df[calibrated_col] = df[target_col].copy()
        
        is_valid = validate_initial_pseudolabels(df, target_col, EVENT_COL, TIME_COL, threshold)
        
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
    df = add_time_features(df, TIME_COL, threshold)
    
    exclude_cols = ['ID', 'year','Age at Diagnosis', target_col, calibrated_col, expected_time_col, TIME_COL, EVENT_COL,
                   'TPS_L1', 'TPS_1_49', 'TPS_50L']
    
    for other_threshold in THRESHOLDS:
        for other_subgroup in SUBGROUPS.keys():
            if other_threshold != threshold or other_subgroup != subgroup:
                other_paths = get_threshold_paths(other_threshold, other_subgroup)
                exclude_cols.extend([
                    other_paths['target_col'],
                    other_paths['calibrated_col'],
                    other_paths['expected_time_col']
                ])
    
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('Pseudo_Label') 
                   and not c.startswith('Expected_Survival')]
    
    joblib.dump(feature_cols, feature_file)
    print(f"Using {len(feature_cols)} features for training")
    
    X = df[feature_cols]
    
    if X.shape[1] == 0:
        print("No valid features remaining!")
        return None
   
    models = None
    best_r2 = -float('inf')
    best_mae = float('inf')
    best_km_loss = float('inf')
    best_c_index = -float('inf')  
    best_c_index_iteration = 0  
    stagnation_count = 0
    convergence_history = []
    
    best_df_state = None
    
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
    
    tf.keras.backend.clear_session()
    
    for iteration in range(1, MAX_ITER + 1):
        print(f"\n{'#'*60}")
        print(f"### {subgroup} - Threshold: {threshold}m - Iteration {iteration} ###")
        print(f"{'#'*60}")
        
        X = df[feature_cols]
        y = df[calibrated_col].copy()
        event = df[EVENT_COL]
        time = df[TIME_COL]
        
        indices = np.arange(len(X))
        
        try:
            train_idx, test_idx = train_test_split(
                indices, test_size=0.15, random_state=42, stratify=event
            )
        except ValueError:
            print("Warning: Cannot stratify. Using random split.")
            train_idx, test_idx = train_test_split(
                indices, test_size=0.15, random_state=42
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
            
            initial_lr = 0.0005 * (0.95 ** (iteration // 5))
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
            
            val_size = int(0.15 * len(X_train_processed))
            if val_size < 5:
                val_size = min(5, len(X_train_processed) // 4)
            
            X_train_final = X_train_processed[:-val_size]
            X_val = X_train_processed[-val_size:]
            y_train_final = y_train[:-val_size]
            y_val = y_train[-val_size:]
            
            history = model.fit(
                X_train_final, y_train_final,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=min(32, len(X_train_final) // 4),
                callbacks=[early_stopping, reduce_lr, TerminateOnNaN()],
                verbose=0,
                sample_weight=sample_weights[:-val_size] if len(sample_weights) > val_size else sample_weights
            )
            
            test_pred = model.predict(X_test_processed, verbose=0).flatten()
            all_test_preds.append(test_pred)
            
            full_pred = model.predict(preprocessor.transform(X), verbose=0).flatten()
            all_preds.append(full_pred)
        
        y_pred_test = np.mean(all_test_preds, axis=0)
        y_pred_full = np.mean(all_preds, axis=0)
        
        valid_mask = ~np.isnan(y_pred_test) & ~np.isinf(y_pred_test)
        
        r2_after = r2_score(y_test[valid_mask], y_pred_test[valid_mask])
        mae_after = mean_absolute_error(y_test[valid_mask], y_pred_test[valid_mask])
        c_index_raw = calculate_c_index(event_test[valid_mask], time_test[valid_mask], 
                                        y_pred_test[valid_mask], threshold)
        
        print(f"\n  RAW PREDICTIONS (before calibration):")
        print(f"    R2: {r2_after:.4f}")
        print(f"    MAE: {mae_after:.4f}")
        print(f"    C-index: {c_index_raw:.4f}")
        print(f"  CALIBRATION PIPELINE (with C-index tracking):")
        print(f"  {'='*50}")

        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
        iso.fit(y_pred_full[train_idx], y.iloc[train_idx].values)
        y_full_calibrated = iso.predict(y_pred_full)
        
        c_index_after_iso = calculate_c_index(event.values, time.values, y_full_calibrated, threshold)
        print(f"    After Isotonic:       C-index = {c_index_after_iso:.4f}")
        
        y_full_calibrated = km_aware_calibration(df, y_full_calibrated, iteration, 
                                                 TIME_COL, EVENT_COL, threshold)
        
        c_index_after_km = calculate_c_index(event.values, time.values, y_full_calibrated, threshold)
        print(f"    After KM calibration: C-index = {c_index_after_km:.4f}")
        
        y_full_calibrated = time_aware_refinement(df, y_full_calibrated, event.values, time.values, 
                                                  threshold, iteration=iteration)
        
        c_index_after_time = calculate_c_index(event.values, time.values, y_full_calibrated, threshold)
        print(f"    After Time-aware:     C-index = {c_index_after_time:.4f}")
        
        if c_index_after_time < c_index_raw - 0.05:
            print(f"\n      MAJOR C-INDEX DROP: {c_index_raw:.4f}  {c_index_after_time:.4f}")
            print(f"    Rolling back to lighter calibration")
            y_full_calibrated = 0.8 * y_pred_full + 0.2 * y_full_calibrated
            c_index_final = calculate_c_index(event.values, time.values, y_full_calibrated, threshold)
            print(f"    After rollback: {c_index_final:.4f}")
        else:
            c_index_final = c_index_after_time
        
        print(f"  {'='*50}\n")
        
        blend_factor = 0.7
        df[target_col] = blend_factor * y_pred_full + (1 - blend_factor) * df[target_col]
        df[calibrated_col] = blend_factor * y_full_calibrated + (1 - blend_factor) * df[calibrated_col]
        df = update_expected_survival_time(df, df[calibrated_col].values, event.values, time.values, 
                                          expected_time_col, TIME_COL, EVENT_COL, threshold)
        df = enforce_constraints_v2(df, iteration, EVENT_COL, target_col, calibrated_col, 
                                   TIME_COL, expected_time_col)
        
        df.to_csv(input_file, index=False)
        
        km_loss, _, _, _ = calculate_km_calibration_loss_v2(
            df, TIME_COL, EVENT_COL, calibrated_col, threshold
        )
        
        convergence_history.append({
            'iteration': iteration,
            'r2_after': r2_after,
            'mae_after': mae_after,
            'c_index': c_index_final,
            'km_loss': km_loss
        })
        
        print(f"\n  ITERATION {iteration} SUMMARY:")
        print(f"    R2:      {r2_after:.4f}")
        print(f"    MAE:     {mae_after:.4f}")
        print(f"    C-index: {c_index_final:.4f}")
        print(f"    KM Loss: {km_loss:.6f}")
        
        improved = False
        if r2_after > best_r2 + MIN_DELTA:
            best_r2 = r2_after
            improved = True
            print(f"     New best R2: {best_r2:.4f}")
        if mae_after < best_mae - MIN_DELTA:
            best_mae = mae_after
            improved = True
            print(f"     New best MAE: {best_mae:.4f}")
        if c_index_final > best_c_index + MIN_DELTA:
            best_c_index = c_index_final
            best_c_index_iteration = iteration  
            improved = True
            print(f"     New best C-index: {best_c_index:.4f}")

            best_df_state = df.copy()
            
        if km_loss < best_km_loss - 0.0001:
            best_km_loss = km_loss
            improved = True
            print(f"     New best KM Loss: {best_km_loss:.6f}")
        
        if improved:
            stagnation_count = 0
            print(f"     Improvement detected! Saving models...")
            
            for i, model in enumerate(models):
                model.save(os.path.join(weights_dir, f'best_model_{i}.keras'))
            joblib.dump(preprocessor, os.path.join(weights_dir, 'best_preprocessor.pkl'))
        else:
            stagnation_count += 1
            print(f"    No improvement for {stagnation_count} iterations")
        
        should_stop, best_iter, stop_reason = should_stop_training(convergence_history, patience=5)
        
        if should_stop:
            print(f"\n{'='*70}")
            print(f"   EARLY STOPPING: C-INDEX DEGRADATION DETECTED")
            print(f"{'='*70}")
            print(f"  {stop_reason}")
            print(f"  Reverting to best C-index iteration: {best_iter}")
            print(f"{'='*70}")
            
            if best_df_state is not None:
                df = best_df_state.copy()
                print(f"   Reverted DataFrame to iteration {best_c_index_iteration}")
            
            break
        
        r2_target_met = r2_after >= TARGET_R2
        c_index_target_met = c_index_final >= TARGET_C_INDEX
        c_index_not_degraded = c_index_final >= (best_c_index - C_INDEX_TOLERANCE)
        km_loss_target_met = km_loss < 0.001
        
        print(f"\n  Progress toward targets:")
        print(f"    R2 = {TARGET_R2}:        {r2_after:.4f} {'' if r2_target_met else ''}")
        print(f"    C-index = {TARGET_C_INDEX}:   {c_index_final:.4f} {'' if c_index_target_met else ''}")
        print(f"    C-index stable:   {'' if c_index_not_degraded else ''} (within {C_INDEX_TOLERANCE:.3f} of best)")
        print(f"    KM Loss < 0.001:  {km_loss:.6f} {'' if km_loss_target_met else ''}")
        print(f"    Best C-index so far: {best_c_index:.4f} (iteration {best_c_index_iteration})")
        
        if r2_target_met and c_index_target_met and c_index_not_degraded and km_loss_target_met:
            print(f"\n{'='*60}")
            print(f"  ALL TARGETS ACHIEVED ")
            print(f"{'='*60}")
            print(f"  R2:      {r2_after:.4f} (target: ={TARGET_R2})")
            print(f"  C-index: {c_index_final:.4f} (target: ={TARGET_C_INDEX})")
            print(f"  KM Loss: {km_loss:.6f} (target: <0.001)")
            print(f"{'='*60}")
            break
        
        if stagnation_count >= PATIENCE:
            print(f"\n  Stopping due to stagnation")
            print(f"   Current metrics: R2={r2_after:.4f}, C-index={c_index_final:.4f}, KM Loss={km_loss:.6f}")
            
            if c_index_final < best_c_index - 0.02 and best_df_state is not None:
                print(f"   Reverting to best C-index iteration: {best_c_index_iteration}")
                df = best_df_state.copy()
            
            break
    
    print(f"\n{'='*70}")
    print(f"SAVING FINAL OPTIMIZED RESULTS")
    print(f"{'='*70}")
    print(f"  Best C-index: {best_c_index:.4f} (achieved at iteration {best_c_index_iteration})")
    print(f"  Total iterations run: {len(convergence_history)}")
    
    df.to_csv(input_file, index=False)
    print(f"   Saved optimized predictions to: {input_file}")
    print(f"{'='*70}\n")
    
    if convergence_history:
        suffix = f"{threshold}m" if subgroup == 'Overall' else f"{threshold}m_{subgroup}"
        history_file = os.path.join(output_dir, f'convergence_history_{suffix}.csv')
        history_df = pd.DataFrame(convergence_history)
        history_df['Iteration'] = history_df['iteration']
        history_df['C_Index'] = history_df['c_index']
        history_df['Best_C_Index'] = history_df['iteration'] == best_c_index_iteration
        
        history_df.to_csv(history_file, index=False)
        print(f" Saved convergence history to: {history_file}")

    print(f"\n{'='*70}")
    print(f"GENERATING KAPLAN-MEIER COMPARISON PLOTS")
    print(f"{'='*70}")
    
    plot_results = plot_observed_vs_predicted_km_curves(
        df=df,
        iteration=len(convergence_history),  
        time_col=TIME_COL,
        event_col=EVENT_COL,
        calibrated_col=calibrated_col,
        threshold=threshold,
        output_dir=output_dir,
        subgroup=subgroup
    )
    
    print(f"\n KM curves plot saved to: {output_dir}/km_curves_comparison_{suffix}.png")
    
    analyze_final_results(
        df=df,
        convergence_history=convergence_history,
        event_col=EVENT_COL,
        calibrated_col=calibrated_col,
        time_col=TIME_COL,
        threshold=threshold,
        subgroup=subgroup
    )
    
    print(f"\n Training complete for {subgroup} - threshold: {threshold} months!")
    
    tf.keras.backend.clear_session()
    
    return df
    
def generate_subgroup_summary(all_results):

    print("\n" + "="*80)
    print("COMPREHENSIVE SUBGROUP ANALYSIS SUMMARY")
    print("="*80)
    summary_file = os.path.join(BASE_OUTPUT_DIR, "subgroup_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-THRESHOLD SURVIVAL ANALYSIS - SUBGROUP SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for threshold in THRESHOLDS:
            f.write(f"\n{'='*80}\n")
            f.write(f"THRESHOLD: {threshold} MONTHS\n")
            f.write(f"{'='*80}\n\n")
            
            for subgroup in SUBGROUPS.keys():
                key = (threshold, subgroup)
                if key in all_results and all_results[key] is not None:
                    df = all_results[key]
                    paths = get_threshold_paths(threshold, subgroup)
                    calibrated_col = paths['calibrated_col']
                    
                    event_mask = df[EVENT_COL] == 1
                    
                    f.write(f"\nSubgroup: {subgroup}\n")
                    f.write(f"{'-'*40}\n")
                    f.write(f"Sample Size: {len(df)}\n")
                    f.write(f"Events: {event_mask.sum()} ({event_mask.sum()/len(df)*100:.1f}%)\n")
                    f.write(f"Censored: {(~event_mask).sum()} ({(~event_mask).sum()/len(df)*100:.1f}%)\n")
                    
                    surv_probs = calculate_survival_probabilities_at_threshold(
                        df, TIME_COL, EVENT_COL, calibrated_col, threshold
                    )
                    
                    f.write(f"\n SURVIVAL AT END OF STUDY ({surv_probs['evaluation_time']:.2f} months):\n")
                    if surv_probs['used_max_observed']:
                        f.write(f"  [Evaluation time = max observed time > threshold]\n")
                    else:
                        f.write(f"  [Evaluation time = threshold]\n")
                    
                    f.write(f"  Observed (Kaplan-Meier): {surv_probs['observed_surv_at_eval']:.2f}%\n")
                    f.write(f"  Predicted (Model):       {surv_probs['predicted_surv_at_eval']:.2f}%\n")
                    if not np.isnan(surv_probs['difference']):
                        f.write(f"  Absolute Difference:     {surv_probs['difference']:.2f}%\n")
                    
                    f.write(f"\n Mean Individual Patient Predictions (at threshold {threshold}m):\n")
                    if not np.isnan(surv_probs['censored_surv_prob']):
                        f.write(f"  Censored patients: {surv_probs['censored_surv_prob']:.2f}%\n")
                    if not np.isnan(surv_probs['deceased_surv_prob']):
                        f.write(f"  Deceased patients: {surv_probs['deceased_surv_prob']:.2f}%\n")
                    f.write(f"  Overall mean: {surv_probs['mean_individual_prob']:.2f}%\n")
                    
                    f.write(f"\n Risk Stratification:\n")
                    f.write(f"  High survival (>70%): {surv_probs['high_surv_count']} ({surv_probs['high_surv_pct']:.1f}%)\n")
                    f.write(f"  Low survival (<30%):  {surv_probs['low_surv_count']} ({surv_probs['low_surv_pct']:.1f}%)\n")
                    
                    f.write(f"\n Other Metrics:\n")
                    f.write(f"  Mean Survival Probability: {df[calibrated_col].mean():.3f}\n")
                    f.write(f"  Median Survival Probability: {df[calibrated_col].median():.3f}\n")
                    
                    c_index = calculate_c_index(df[EVENT_COL], df[TIME_COL], 
                                               df[calibrated_col], threshold)
                    f.write(f"  C-index: {c_index:.4f}\n")
                    
                    kmf = KaplanMeierFitter()
                    kmf.fit(df[TIME_COL], df[EVENT_COL])
                    median_surv = kmf.median_survival_time_
                    if not np.isnan(median_surv):
                        f.write(f"  Median Survival Time: {median_surv:.2f} months\n")
                    else:
                        f.write(f"  Median Survival Time: Not reached\n")
                    
                    f.write(f"Output Directory: {paths['output_dir']}\n")
                    f.write(f"Weights Directory: {paths['weights_dir']}\n")
                    f.write("\n")
                else:
                    f.write(f"\nSubgroup: {subgroup}\n")
                    f.write(f"{'-'*40}\n")
                    f.write(f"Training failed or skipped\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("="*80 + "\n")
    
    print(f"\n Comprehensive summary saved to: {summary_file}")
    
    with open(summary_file, 'r') as f:
        print(f.read())
    
if __name__ == "__main__":
    print("="*80)
    print("SPARC TRAINING WITH CROSS-VALIDATION SUPPORT")
    print("="*80)
    print(f"Training mode: {'CROSS-VALIDATION' if USE_CROSS_VALIDATION else 'FULL DATASET'}")
    print(f"Thresholds: {THRESHOLDS} months")
    print(f"Subgroups: {list(SUBGROUPS.keys())}")
    
    if USE_CROSS_VALIDATION:
        print(f"CV folds: {N_CV_FOLDS}")
        print(f"CV random state: {CV_RANDOM_STATE}")
        print(f"Output directory: {BASE_OUTPUT_DIR}")
        print("")
        print(" This mode produces results compatible with baseline comparison!")
        print(" DeLong tests can be performed using the same CV folds!")
    else:
        print(f"Output directory: {BASE_OUTPUT_DIR}")
        print("")
        print("Full dataset training mode")
        print("   Results will NOT be comparable to baseline models")
    
    print("="*80 + "\n")
    
    base_dataset_file = "final_OS_dataset_01_final.csv"
    
    if not os.path.exists(base_dataset_file):
        print(f"Dataset file '{base_dataset_file}' not found!")
        print("Please update the path in the code.")
        exit(1)
    
    base_df = pd.read_csv(base_dataset_file)
    print(f"Loaded base dataset: {len(base_df)} samples\n")
    
    
    all_results = {}
    cv_comprehensive_summary = []
    
    for threshold in THRESHOLDS:
        for subgroup in SUBGROUPS.keys():
            print(f"\n{'='*80}")
            print(f"PROCESSING: {subgroup} - {threshold}m")
            print(f"{'='*80}")
            
            try:
                if USE_CROSS_VALIDATION:
                    
                    cv_res = cross_validate_threshold_comprehensive(
                        threshold=threshold,
                        df=base_df,
                        subgroup=subgroup
                    )
                    
                    if cv_res:
                        cv_comprehensive_summary.append({
                            'threshold': threshold,
                            'subgroup': subgroup,
                            **cv_res
                        })
                        
                        all_results[(threshold, subgroup)] = cv_res
                else:

                    print("Full dataset training (no CV)")
                    print("Results will NOT be comparable to baseline models")
                    result_df = train_for_threshold(threshold, base_df, subgroup)
                    all_results[(threshold, subgroup)] = result_df
                
            except Exception as e:
                print(f"\n Threshold={threshold}m, Subgroup={subgroup}")
                print(f"{str(e)}")
                import traceback
                traceback.print_exc()
                all_results[(threshold, subgroup)] = None
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    
    if USE_CROSS_VALIDATION:
        print(f"\nResults saved to: {BASE_OUTPUT_DIR}")
        print("\nPer threshold/subgroup:")
        print("  - sparc_cv_predictions_*.csv (individual predictions)")
        print("  - sparc_cv_fold_results_*.csv (fold-wise metrics)")
        print("  - optimized_survival_probabilities_*.csv (compatibility format)")
        print("  - convergence_history_*.csv (compatibility format)")
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"{'Threshold':<12} {'Subgroup':<12} {'Mean C-Index':<15} {'Std':<12}")
        print(f"{'-'*80}")
        
        for (threshold, subgroup), result in all_results.items():
            if result is not None and isinstance(result, dict) and 'mean_c_index' in result:
                print(f"{threshold:<12} {subgroup:<12} "
                      f"{result['mean_c_index']:<15.4f} "
                      f"{result['std_c_index']:<12.4f}")
        
    else:
        print(f"\nResults saved to: {BASE_OUTPUT_DIR}")
        print("\nNote: These results use full-dataset training and are")
        print("NOT compatible with baseline comparison for statistical tests.")
        
    if cv_comprehensive_summary and USE_CROSS_VALIDATION:
        master_summary_file = os.path.join(BASE_OUTPUT_DIR, "comprehensive_cv_master_summary.txt")
        with open(master_summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE CROSS-VALIDATION SUMMARY - ALL 3 METHODS\n")
            f.write("="*80 + "\n\n")
            
            for item in cv_comprehensive_summary:
                f.write(f"\nThreshold: {item['threshold']}m | Subgroup: {item['subgroup']}\n")
                f.write(f"{'-'*80}\n")
                
                f.write(f"\n  NESTED K-FOLD:\n")
                f.write(f"    Overall C-index (combined):  {item['nested_results']['overall_c_index_final']:.4f}\n")
                f.write(f"    Mean C-index (across folds): {item['nested_results']['mean_c_index_final']:.4f} +/- {item['nested_results']['std_c_index_final']:.4f}\n")
                
                f.write(f"\n  REPEATED K-FOLD:\n")
                f.write(f"    Mean C-index: {item['repeated_results']['mean_c_index_final']:.4f} +/- {item['repeated_results']['std_c_index_final']:.4f}\n")
                
                f.write(f"\n  BOOTSTRAP:\n")
                f.write(f"    Optimism-corrected C-index: {item['bootstrap_results']['optimism_corrected_iso']:.4f}\n")
                
                f.write(f"\n")
        
        print(f"\n Master CV summary saved to: {master_summary_file}")
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE CROSS-VALIDATION SUMMARY - ALL 3 METHODS")
    print(f"{'='*80}")
    
    if USE_CROSS_VALIDATION and cv_comprehensive_summary:
        print(f"\n{'Threshold':<12} {'Subgroup':<12} {'Nested':<12} {'Repeated':<12} {'Bootstrap':<12}")
        print(f"{'-'*80}")
        
        for item in cv_comprehensive_summary:
            print(f"{item['threshold']:<12} {item['subgroup']:<12} "
                  f"{item['nested_results']['mean_c_index_final']:<12.4f} "
                  f"{item['repeated_results']['mean_c_index_final']:<12.4f} "
                  f"{item['bootstrap_results']['optimism_corrected_iso']:<12.4f}")
    
    print(f"{'='*80}\n")
    print("COMPREHENSIVE CV PIPELINE COMPLETED")
    print("="*80)