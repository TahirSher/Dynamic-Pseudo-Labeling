import pandas as pd
import numpy as np
import os
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
THRESHOLDS = [60, 84, 120]
SUBGROUPS = {
    'Overall': None,
}
TIME_COL = 'Time'
EVENT_COL = 'Event'
N_FOLDS = 5  
CV_RANDOM_STATE = 42  

ORIGINAL_DATASET_FILE = "final_OS_dataset_01_final.csv"

# SPARC output directory
SPARC_OUTPUT_DIR = "14SPARC_CV_multi_threshold_survival_output"

# Results output directory
BASE_OUTPUT_DIR = "16Jan-baseline_comparison_results_CV_MATCHED"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


def detect_sparc_training_mode(threshold, subgroup='Overall'):

    suffix = f"{threshold}m" if subgroup == 'Overall' else f"{threshold}m_{subgroup}"
    
    cv_predictions_file = os.path.join(
        SPARC_OUTPUT_DIR,
        f"threshold_{threshold}m" + (f"_{subgroup}" if subgroup != 'Overall' else ""),
        f"sparc_cv_predictions_{suffix}.csv"
    )
    
    cv_fold_results_file = os.path.join(
        SPARC_OUTPUT_DIR,
        f"threshold_{threshold}m" + (f"_{subgroup}" if subgroup != 'Overall' else ""),
        f"sparc_cv_fold_results_{suffix}.csv"
    )
    
    used_cv = os.path.exists(cv_predictions_file) and os.path.exists(cv_fold_results_file)
    
    if used_cv:
        print(f"\n SPARC used CROSS-VALIDATION for {subgroup} - {threshold}m")
        print(f"   Valid statistical comparison is possible!")
        return True
    else:
        print(f"\n SPARC used FULL DATASET training for {subgroup} - {threshold}m")
        print(f"   Statistical comparison will be INVALID")
        return False

def load_original_dataset(threshold, subgroup='Overall'):

    print(f"\n Loading ORIGINAL dataset (no SPARC-derived features)")
    
    if not os.path.exists(ORIGINAL_DATASET_FILE):
        print(f" ERROR: Original file not found: {ORIGINAL_DATASET_FILE}")
        return None
    
    df = pd.read_csv(ORIGINAL_DATASET_FILE)
    print(f"Loaded {len(df)} samples from original dataset")
    
    if subgroup != 'Overall' and SUBGROUPS[subgroup] is not None:
        mask = pd.Series(True, index=df.index)
        for col, val in SUBGROUPS[subgroup].items():
            if col in df.columns:
                mask &= (df[col] == val)
        df = df[mask].reset_index(drop=True)
        print(f"   After subgroup filtering: {len(df)} samples")
    
    required_cols = [TIME_COL, EVENT_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return None
    
    sparc_keywords = ['Pseudo', 'Calibrated', 'Expected', 'Survival_Prob', 
                      'Risk_Score', 'Predicted', 'log_time', 'time_squared',
                      'time_normalized', 'time_cubic', 'time_sqrt', 'time_reciprocal',
                      'time_progression', 'time_remaining', 'time_ratio', 'time_decay',
                      'time_event_interaction', 'time_bins', 'SPARC']
    
    cols_to_drop = []
    for col in df.columns:
        if any(keyword.lower() in col.lower() for keyword in sparc_keywords):
            cols_to_drop.append(col)
    
    if cols_to_drop:
        print(f"Removing {len(cols_to_drop)} potentially derived columns")
        df = df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"Final dataset: {len(df)} samples, {len(df.columns)} features")
    return df

class DeepSurvDataset(Dataset):
    def __init__(self, X, time, event):
        self.X = torch.FloatTensor(X)
        self.time = torch.FloatTensor(time)
        self.event = torch.FloatTensor(event)
   
    def __len__(self):
        return len(self.X)
   
    def __getitem__(self, idx):
        return self.X[idx], self.time[idx], self.event[idx]

class DeepSurvModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16]):
        super(DeepSurvModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
   
    def forward(self, x):
        return self.network(x)

def neg_partial_log_likelihood(risk_scores, time, event):
    sorted_indices = torch.argsort(time, descending=False)
    risk_scores = risk_scores[sorted_indices]
    event = event[sorted_indices]
   
    hazard_ratio = torch.exp(risk_scores)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
   
    uncensored_likelihood = risk_scores - log_risk
    censored_likelihood = uncensored_likelihood * event
   
    neg_likelihood = -torch.sum(censored_likelihood) / torch.sum(event)
    return neg_likelihood

def train_deepsurv(X_train, time_train, event_train, X_test, time_test, event_test,
                   epochs=300, batch_size=64, learning_rate=1e-4, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    train_dataset = DeepSurvDataset(X_train, time_train, event_train)
    test_dataset = DeepSurvDataset(X_test, time_test, event_test)
   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
   
    model = DeepSurvModel(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
   
    best_c_index = 0.0
    best_risk_scores = np.zeros(len(time_test))
   
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
       
        for batch_X, batch_time, batch_event in train_loader:
            batch_X = batch_X.to(device)
            batch_time = batch_time.to(device)
            batch_event = batch_event.to(device)
           
            optimizer.zero_grad()
            risk_scores = model(batch_X).squeeze()
            risk_scores = torch.clamp(risk_scores, -20, 20)
           
            loss = neg_partial_log_likelihood(risk_scores, batch_time, batch_event)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
           
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        test_risk_scores_list = []
        for batch_X, _, _ in test_loader:
            batch_X = batch_X.to(device)
            risk = model(batch_X).squeeze()
            risk = torch.clamp(risk, -50, 50)
            test_risk_scores_list.append(risk.cpu().numpy())
        
        test_risk_scores = np.concatenate(test_risk_scores_list)
        
        if not np.any(np.isnan(test_risk_scores)):
            try:
                c_index = concordance_index_censored(
                    event_test.astype(bool),
                    time_test,
                    test_risk_scores
                )[0]
                if c_index > best_c_index:
                    best_c_index = c_index
                    best_risk_scores = test_risk_scores.copy()
            except:
                pass
   
    if verbose:
        print(f"DeepSurv training completed. Best Test C-index: {best_c_index:.4f}")
    
    return best_risk_scores, best_c_index


# ============================================================================
# BASELINE MODELS
# ============================================================================
def train_cox_ph(X_train, time_train, event_train, X_test, time_test, event_test, feature_names, verbose=False):
    if verbose:
        print("Training Cox PH (lifelines)")
    
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df[TIME_COL] = time_train
    train_df[EVENT_COL] = event_train
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df[TIME_COL] = time_test
    test_df[EVENT_COL] = event_test
   
    for penalizer in [0.1, 0.01, 0.001, 0.0]:
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(train_df, duration_col=TIME_COL, event_col=EVENT_COL, show_progress=False)
            test_risk = cph.predict_partial_hazard(test_df).values.flatten()
           
            test_c = concordance_index_censored(event_test.astype(bool), time_test, test_risk)[0]
           
            if verbose:
                print(f"  Success (penalizer={penalizer}), Test: {test_c:.4f}")
            return {'test_c_index': test_c, 'test_risk_scores': test_risk}
        except Exception as e:
            continue
    
    if verbose:
        print("Cox PH failed to converge")
    return None

def train_random_survival_forest(X_train, time_train, event_train, X_test, time_test, event_test, verbose=False):
    if verbose:
        print("Training Random Survival Forest")
    
    y_train = Surv.from_arrays(event_train.astype(bool), time_train)
    y_test = Surv.from_arrays(event_test.astype(bool), time_test)
   
    rsf = RandomSurvivalForest(
        n_estimators=200, 
        min_samples_leaf=10, 
        max_features="sqrt", 
        n_jobs=-1, 
        random_state=42
    )
    rsf.fit(X_train, y_train)
   
    test_risk = rsf.predict(X_test)
    test_c = concordance_index_censored(event_test.astype(bool), time_test, test_risk)[0]
   
    if verbose:
        print(f"Test: {test_c:.4f}")
    
    return {'test_c_index': test_c, 'test_risk_scores': test_risk}

def train_cox_sksurv(X_train, time_train, event_train, X_test, time_test, event_test, verbose=False):
    if verbose:
        print(" Training Cox PH (scikit-survival)")
    
    y_train = Surv.from_arrays(event_train.astype(bool), time_train)
    y_test = Surv.from_arrays(event_test.astype(bool), time_test)
   
    for alpha in [1.0, 0.1, 0.01]:
        try:
            cox = CoxPHSurvivalAnalysis(alpha=alpha, n_iter=200)
            cox.fit(X_train, y_train)
            test_risk = cox.predict(X_test)
           
            test_c = concordance_index_censored(event_test.astype(bool), time_test, test_risk)[0]
           
            if verbose:
                print(f" Success (alpha={alpha}) , Test: {test_c:.4f}")
            return {'test_c_index': test_c, 'test_risk_scores': test_risk}
        except:
            continue
    
    if verbose:
        print(" Cox (sksurv) failed")
    return None

def train_deepsurv_wrapper(X_train, time_train, event_train, X_test, time_test, event_test, verbose=False):
    if verbose:
        print("Training DeepSurv")
    
    test_risk, test_c = train_deepsurv(
        X_train, time_train, event_train,
        X_test, time_test, event_test,
        epochs=300,
        batch_size=64,
        learning_rate=1e-4,
        verbose=verbose
    )
    
    if verbose:
        print(f"Best Test C-index: {test_c:.4f}")
    
    return {'test_c_index': test_c, 'test_risk_scores': test_risk}


# ============================================================================
# STANDARD PSEUDO-OBSERVATION 
# ============================================================================
def compute_standard_pseudo_observations_train_only(time, event, threshold):

    n = len(time)
    
    kmf = KaplanMeierFitter()
    kmf.fit(time, event)
    
    try:
        S_all = kmf.predict(threshold)
        if np.isnan(S_all):
            S_all = kmf.survival_function_.iloc[-1].values[0]
    except:
        S_all = kmf.survival_function_.iloc[-1].values[0]
    
    pseudo_obs = np.zeros(n)
    
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        kmf_loo = KaplanMeierFitter()
        kmf_loo.fit(time[mask], event[mask])
        
        try:
            S_loo = kmf_loo.predict(threshold)
            if np.isnan(S_loo):
                S_loo = kmf_loo.survival_function_.iloc[-1].values[0]
        except:
            S_loo = kmf_loo.survival_function_.iloc[-1].values[0]
        
        pseudo_obs[i] = n * S_all - (n - 1) * S_loo
    
    pseudo_obs = np.clip(pseudo_obs, 0, 1)
    return pseudo_obs

def train_standard_pseudo_observation(X_train, time_train, event_train, 
                                     X_test, time_test, event_test, 
                                     threshold, feature_names, verbose=False):

    if verbose:
        print(f"Training Standard Pseudo-Observation (threshold={threshold}m)")
    
    pseudo_train = compute_standard_pseudo_observations_train_only(time_train, event_train, threshold)
    
    best_c_test = 0
    best_test_pred = None
    
    for alpha in [1.0, 0.1, 0.01, 0.001]:
        ridge = Ridge(alpha=alpha, random_state=42)
        ridge.fit(X_train, pseudo_train)
        
        test_pred = ridge.predict(X_test)
        test_pred = np.clip(test_pred, 0, 1)
        
        try:
            test_c = concordance_index_censored(
                event_test.astype(bool), time_test, 1 - test_pred
            )[0]
            
            if test_c > best_c_test:
                best_c_test = test_c
                best_test_pred = test_pred
        except:
            continue
    
    if verbose:
        print(f"Best Test C-index: {best_c_test:.4f}")
    
    return {
        'test_c_index': best_c_test, 
        'test_risk_scores': 1 - best_test_pred,
    }


# ============================================================================
# LOAD SPARC CV RESULTS
# ============================================================================
def load_sparc_cv_results(threshold, subgroup='Overall'):

    suffix = f"{threshold}m" if subgroup == 'Overall' else f"{threshold}m_{subgroup}"
    subgroup_suffix = f"_{subgroup}" if subgroup != 'Overall' else ""
    
    fold_results_file = os.path.join(
        SPARC_OUTPUT_DIR,
        f"threshold_{threshold}m{subgroup_suffix}",
        f"sparc_cv_fold_results_{suffix}.csv"
    )
    
    print(f"\n Loading SPARC CV results:")
    print(f"  File: {fold_results_file}")
    print(f"  Exists: {os.path.exists(fold_results_file)}")
    
    if not os.path.exists(fold_results_file):
        print(f"File not found")
        return None
    
    try:
        df_folds = pd.read_csv(fold_results_file)
        
        print(f" File loaded successfully")
        print(f"    Shape: {df_folds.shape}")
        print(f"    Columns: {df_folds.columns.tolist()}")
        
        c_index_columns = [
            'C_Index_Final',      
            'C_Index_TimeAware',  
            'C_Index_KM',         
            'C_Index_Isotonic',  
            'C_Index_Raw',        
            'C_Index'            
        ]
        
        c_index_col = None
        for col in c_index_columns:
            if col in df_folds.columns:
                c_index_col = col
                print(f" Using C-index column: {c_index_col}")
                break
        
        if c_index_col is None:
            print(f" No valid C-index column found in: {df_folds.columns.tolist()}")
            return None

        if 'Fold' not in df_folds.columns:
            print(f" 'Fold' column not found")
            return None
        
        fold_c_indices = df_folds[c_index_col].values
        
        overall_c_col_candidates = [
            'Overall_C_Index_Final',
            'Overall_C_Index',
            f'Mean_{c_index_col}'
        ]
        
        overall_c = None
        for col in overall_c_col_candidates:
            if col in df_folds.columns and not pd.isna(df_folds[col].iloc[0]):
                overall_c = df_folds[col].iloc[0]
                print(f"Using overall C-index from column: {col}")
                break
        
        if overall_c is None:
            overall_c = np.mean(fold_c_indices)
            print(f"  Computed overall C-index from fold means: {overall_c:.4f}")
        
        print(f"\n  SPARC CV Results Summary:")
        print(f"    Number of folds: {len(fold_c_indices)}")
        print(f"    Fold-wise C-indices: {fold_c_indices}")
        print(f"    Mean C-index: {np.mean(fold_c_indices):.4f}")
        print(f"    Std C-index: {np.std(fold_c_indices):.4f}")
        print(f"    Overall C-index: {overall_c:.4f}")
        
        result = {
            'fold_c_indices': fold_c_indices,
            'overall_c_index': overall_c,
            'mean_c_index': np.mean(fold_c_indices),
            'std_c_index': np.std(fold_c_indices),
            'c_index_type': c_index_col, 
            'n_folds': len(fold_c_indices)
        }
        
        return result
        
    except Exception as e:
        print(f"  Failed to load SPARC CV results: {e}")
        import traceback
        traceback.print_exc()
        return None


def cross_validation_comparison_matched(df, threshold, subgroup='Overall', n_folds=5):

    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION COMPARISON (MATCHED FOLDS) - {subgroup} - {threshold}m")
    print(f"{'='*80}")
   
    exclude_cols = ['ID', TIME_COL, EVENT_COL, 'TPS_L1', 'TPS_1_49', 'TPS_50L']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X_df = df[feature_cols].select_dtypes(include=[np.number])
    feature_cols = X_df.columns.tolist()
    X = X_df.values
    time = df[TIME_COL].values
    event = df[EVENT_COL].values

    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
   
    selector = VarianceThreshold(threshold=0.01)
    X = selector.fit_transform(X)
    feature_cols = [f for f, s in zip(feature_cols, selector.get_support()) if s]
   
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    X = np.clip(X, -10, 10)
   
    print(f"Final feature count: {X.shape[1]}")
    print(f"Dataset size: {len(df)} samples")
   
    if X.shape[1] == 0:
        print("No valid features remaining!")
        return None

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CV_RANDOM_STATE)
    
    cv_results = {
        'Cox PH (lifelines)': [],
        'RSF': [],
        'Cox PH (L2)': [],
        'DeepSurv': [],
        'Standard Pseudo-Obs': []
    }
    
    all_baseline_predictions = {key: [] for key in cv_results.keys()}
    all_test_events = []
    all_test_times = []
    
    print(f"\n{'='*80}")
    print(f"RUNNING {n_folds}-FOLD CROSS-VALIDATION (random_state={CV_RANDOM_STATE})")
    print(f"{'='*80}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, event), 1):
        print(f"\n--- Fold {fold_idx}/{n_folds} ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        time_train, time_test = time[train_idx], time[test_idx]
        event_train, event_test = event[train_idx], event[test_idx]
        
        all_test_events.append(event_test)
        all_test_times.append(time_test)
        
        cox_res = train_cox_ph(X_train, time_train, event_train, X_test, time_test, event_test, feature_cols, verbose=False)
        if cox_res:
            cv_results['Cox PH (lifelines)'].append(cox_res['test_c_index'])
            all_baseline_predictions['Cox PH (lifelines)'].append(cox_res['test_risk_scores'])
            print(f"  Cox PH: {cox_res['test_c_index']:.4f}")
        
        rsf_res = train_random_survival_forest(X_train, time_train, event_train, X_test, time_test, event_test, verbose=False)
        if rsf_res:
            cv_results['RSF'].append(rsf_res['test_c_index'])
            all_baseline_predictions['RSF'].append(rsf_res['test_risk_scores'])
            print(f"  RSF: {rsf_res['test_c_index']:.4f}")
        
        cox_sk_res = train_cox_sksurv(X_train, time_train, event_train, X_test, time_test, event_test, verbose=False)
        if cox_sk_res:
            cv_results['Cox PH (L2)'].append(cox_sk_res['test_c_index'])
            all_baseline_predictions['Cox PH (L2)'].append(cox_sk_res['test_risk_scores'])
            print(f"  Cox PH (L2): {cox_sk_res['test_c_index']:.4f}")
        
        deep_res = train_deepsurv_wrapper(X_train, time_train, event_train, X_test, time_test, event_test, verbose=False)
        if deep_res:
            cv_results['DeepSurv'].append(deep_res['test_c_index'])
            all_baseline_predictions['DeepSurv'].append(deep_res['test_risk_scores'])
            print(f"  DeepSurv: {deep_res['test_c_index']:.4f}")
        
        std_pseudo_res = train_standard_pseudo_observation(
            X_train, time_train, event_train, 
            X_test, time_test, event_test,
            threshold, feature_cols, verbose=False
        )
        if std_pseudo_res:
            cv_results['Standard Pseudo-Obs'].append(std_pseudo_res['test_c_index'])
            all_baseline_predictions['Standard Pseudo-Obs'].append(std_pseudo_res['test_risk_scores'])
            print(f"  Standard Pseudo-Obs: {std_pseudo_res['test_c_index']:.4f}")
    
    all_test_events = np.concatenate(all_test_events)
    all_test_times = np.concatenate(all_test_times)
    
    for key in all_baseline_predictions:
        if all_baseline_predictions[key]:
            all_baseline_predictions[key] = np.concatenate(all_baseline_predictions[key])
    
    print(f"\n{'='*80}")
    print("LOADING SPARC CV RESULTS")
    print(f"{'='*80}")
    
    sparc_cv = load_sparc_cv_results(threshold, subgroup)
    sparc_all_risk = None
    sparc_test_predictions = None
    
    if sparc_cv is not None and isinstance(sparc_cv, dict):
        if 'fold_c_indices' in sparc_cv:
            fold_c_indices = sparc_cv['fold_c_indices']
            
            if len(fold_c_indices) == n_folds:
            
                if isinstance(fold_c_indices, np.ndarray):
                    fold_c_list = fold_c_indices.tolist()
                else:
                    fold_c_list = list(fold_c_indices)

                cv_results['SPARC (CV-matched)'] = fold_c_list
                
                print(f"\n SPARC CV results successfully added to cv_results")
                print(f"   C-index type: {sparc_cv.get('c_index_type', 'Unknown')}")
                print(f"   Fold-wise values: {fold_c_list}")
                print(f"   Mean: {np.mean(fold_c_list):.4f} +/- {np.std(fold_c_list):.4f}")
                
                if 'SPARC (CV-matched)' in cv_results:
                    print(f"   Verification: Key exists in cv_results")
                else:
                    print(f"   ERROR: Key NOT found in cv_results after addition!")
                
            else:
                print(f"\n WARNING: SPARC has {len(fold_c_indices)} folds, expected {n_folds}")
                print(f"   Cannot add to cv_results (fold count mismatch)")
        else:
            print(f"\n 'fold_c_indices' not found in SPARC results")
            print(f"   Available keys: {list(sparc_cv.keys())}")
    else:
        print(f"\n SPARC CV results not available")
        if sparc_cv is None:
            print(f"   Reason: load_sparc_cv_results returned None")
        else:
            print(f"   Reason: Unexpected return type ({type(sparc_cv)})")
    
    print(f"\n Final cv_results keys: {list(cv_results.keys())}")
    
    if sparc_cv is not None:
        suffix = f"{threshold}m" if subgroup == 'Overall' else f"{threshold}m_{subgroup}"
        subgroup_suffix = f"_{subgroup}" if subgroup != 'Overall' else ""
        sparc_pred_file = os.path.join(
            SPARC_OUTPUT_DIR,
            f"threshold_{threshold}m{subgroup_suffix}",
            f"sparc_cv_predictions_{suffix}.csv"
        )
        
        if os.path.exists(sparc_pred_file):
            try:
                df_sparc = pd.read_csv(sparc_pred_file)
                risk_col = f'SPARC_CV_Risk_Score_{suffix}'
                
                print(f"\n   SPARC predictions file found:")
                print(f"     File: {sparc_pred_file}")
                print(f"     Shape: {df_sparc.shape}")
                print(f"     Current dataset size: {len(df)}")
                
                if len(df_sparc) != len(df):
                    print(f"\n   Dataset size mismatch!")
                    print(f"     SPARC dataset: {len(df_sparc)} samples")
                    print(f"     Current dataset: {len(df)} samples")
                    print(f"     Cannot perform DeLong test (different datasets)")
                    sparc_all_risk = None
                elif risk_col in df_sparc.columns:
                    sparc_all_risk = df_sparc[risk_col].values
                    print(f" SPARC predictions loaded (will extract by CV folds)")
                    
                    
                    print(f"\n Extracting SPARC predictions using same CV folds...")
                    sparc_fold_predictions = []
                    
                    skf_sparc = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CV_RANDOM_STATE)
                    for fold_idx, (_, test_idx) in enumerate(skf_sparc.split(X, event), 1):
                        sparc_fold_predictions.append(sparc_all_risk[test_idx])
                        print(f"     Fold {fold_idx}: extracted {len(test_idx)} predictions")
                    
                    sparc_test_predictions = np.concatenate(sparc_fold_predictions)
                    print(f" Total SPARC test predictions: {len(sparc_test_predictions)}")
                    print(f" Total baseline test predictions: {len(all_test_events)}")
                    
                    if len(sparc_test_predictions) == len(all_test_events):
                        print(f"Sizes match! DeLong test is possible")
                    else:
                        print(f"Size mismatch - DeLong test not possible")
                        sparc_test_predictions = None
                else:
                    print(f"Column {risk_col} not found")
                    print(f"     Available columns: {df_sparc.columns.tolist()}")
                    sparc_all_risk = None
            except Exception as e:
                print(f"Failed to load SPARC predictions: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n SPARC predictions file not found: {sparc_pred_file}")
    
    return cv_results, all_baseline_predictions, sparc_test_predictions, all_test_events, all_test_times


def delong_test(y_true, y_pred1, y_pred2):

    from sklearn.metrics import roc_auc_score
    
    y_true = np.array(y_true, dtype=bool)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    
    n = len(y_true)
    pos_idx = np.where(y_true)[0]
    neg_idx = np.where(~y_true)[0]
    
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan
    
    V10 = np.zeros(n_pos)
    V01 = np.zeros(n_neg)
    V20 = np.zeros(n_pos)
    V02 = np.zeros(n_neg)
    
    for i, pos in enumerate(pos_idx):
        V10[i] = np.mean(y_pred1[pos] > y_pred1[neg_idx])
        V20[i] = np.mean(y_pred2[pos] > y_pred2[neg_idx])
    
    for i, neg in enumerate(neg_idx):
        V01[i] = np.mean(y_pred1[pos_idx] > y_pred1[neg])
        V02[i] = np.mean(y_pred2[pos_idx] > y_pred2[neg])

    auc1 = np.mean(V10)
    auc2 = np.mean(V20)
    
    S10 = np.var(V10, ddof=1) / n_pos
    S01 = np.var(V01, ddof=1) / n_neg
    S20 = np.var(V20, ddof=1) / n_pos
    S02 = np.var(V02, ddof=1) / n_neg
    
    S_pos = np.cov(V10, V20)[0, 1] / n_pos
    S_neg = np.cov(V01, V02)[0, 1] / n_neg
    
    var_diff = S10 + S01 + S20 + S02 - 2 * (S_pos + S_neg)
    
    if var_diff <= 0:
        return np.nan, np.nan
    
    z_score = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return z_score, p_value

def compute_statistical_tests_with_delong(cv_results, baseline_predictions, sparc_predictions,
                                         all_test_events, all_test_times, output_dir, threshold, subgroup='Overall'):

    print(f"\n{'='*80}")
    print(f"STATISTICAL ANALYSIS - {subgroup} - Threshold {threshold}m")
    print(f"{'='*80}")
    
    df_cv = pd.DataFrame(cv_results)
    
    print("\n Cross-Validation Results Summary:")
    print(df_cv.describe())
    
    cv_file = os.path.join(output_dir, f'cv_results_{threshold}m_{subgroup}.csv')
    df_cv.to_csv(cv_file, index=False)
    print(f"\n CV results saved to: {cv_file}")

    print(f"\n{'='*80}")
    print("95% CONFIDENCE INTERVALS")
    print(f"{'='*80}")
    
    ci_results = []
    for model in df_cv.columns:
        scores = df_cv[model].values
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        ci_lower = mean - 1.96 * std / np.sqrt(len(scores))
        ci_upper = mean + 1.96 * std / np.sqrt(len(scores))
        
        ci_results.append({
            'Model': model,
            'Mean': mean,
            'Std': std,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper
        })
        
        print(f"{model:30s}: {mean:.4f} +/- {std:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    df_ci = pd.DataFrame(ci_results)
    ci_file = os.path.join(output_dir, f'confidence_intervals_{threshold}m_{subgroup}.csv')
    df_ci.to_csv(ci_file, index=False)

    if 'SPARC (CV-matched)' not in df_cv.columns:
        print("\n SPARC CV results not available - skipping statistical tests")
        return df_ci, None, None
    
    print(f"\n{'='*80}")
    print(f"PAIRED STATISTICAL TESTS: SPARC vs BASELINES")
    print(f"{'='*80}")
    
    sparc_scores = df_cv['SPARC (CV-matched)'].values
    
    stat_test_results = []
    
    for model in df_cv.columns:
        if 'SPARC' in model:
            continue
        
        baseline_scores = df_cv[model].values
        
        try:
            t_stat, t_pval = ttest_rel(sparc_scores, baseline_scores)
            t_result = "?" if t_pval < 0.05 else "?"
        except:
            t_stat, t_pval = np.nan, np.nan
            t_result = "N/A"
        
        try:
            differences = sparc_scores - baseline_scores
            if np.all(differences == 0):
                w_stat, w_pval = np.nan, 1.0
                w_result = "?"
            else:
                w_stat, w_pval = wilcoxon(sparc_scores, baseline_scores)
                w_result = "?" if w_pval < 0.05 else "?"
        except:
            w_stat, w_pval = np.nan, np.nan
            w_result = "N/A"
        
        mean_diff = np.mean(sparc_scores - baseline_scores)
        
        stat_test_results.append({
            'Baseline_Model': model,
            'SPARC_Mean': np.mean(sparc_scores),
            'Baseline_Mean': np.mean(baseline_scores),
            'Mean_Difference': mean_diff,
            'T_Statistic': t_stat,
            'T_Test_P_Value': t_pval,
            'T_Test_Significant': t_result,
            'Wilcoxon_Statistic': w_stat,
            'Wilcoxon_P_Value': w_pval,
            'Wilcoxon_Significant': w_result
        })
        
        print(f"\nSPARC vs {model}:")
        print(f"  Mean Difference: {mean_diff:+.4f}")
        print(f"  Paired t-test: t={t_stat:.3f}, p={t_pval:.4f} {t_result}")
        print(f"  Wilcoxon test: W={w_stat:.3f}, p={w_pval:.4f} {w_result}")
    
    df_stat = pd.DataFrame(stat_test_results)
    stat_file = os.path.join(output_dir, f'statistical_tests_{threshold}m_{subgroup}.csv')
    df_stat.to_csv(stat_file, index=False)
    print(f"\nStatistical tests saved to: {stat_file}")
    
    print(f"\n{'='*80}")
    print(f"DELONG TESTS: SPARC vs BASELINES (Same Test Set)")
    print(f"{'='*80}")
    
    delong_results = []
    
    if sparc_predictions is not None and len(sparc_predictions) > 0:
        print(f"SPARC test predictions available: {len(sparc_predictions)} samples")
        print(f"Test events available: {len(all_test_events)} samples")
        
        for model in baseline_predictions:
            baseline_risk = baseline_predictions[model]
            
            if baseline_risk is None:
                print(f"\n  No predictions for {model}: skipping")
                continue
            
            if isinstance(baseline_risk, (list, np.ndarray)):
                if len(baseline_risk) == 0:
                    print(f"\n  Empty predictions for {model}: skipping")
                    continue
            
            if len(baseline_risk) != len(sparc_predictions):
                print(f"\n Size mismatch for {model}:")
                print(f"   Baseline: {len(baseline_risk)} samples")
                print(f"   SPARC: {len(sparc_predictions)} samples")
                print(f"   Skipping DeLong test")
                continue
            
            if len(baseline_risk) != len(all_test_events):
                print(f"\n  Size mismatch for {model} vs events:")
                print(f"    Baseline: {len(baseline_risk)} samples")
                print(f"    Events: {len(all_test_events)} samples")
                print(f"    Skipping DeLong test")
                continue
            
            try:
                print(f"\n  Performing DeLong test: SPARC vs {model}")
                z_score, p_value = delong_test(all_test_events, sparc_predictions, baseline_risk)
                
                if np.isnan(z_score) or np.isnan(p_value):
                    print(f" DeLong test returned NaN - skipping")
                    continue
                
                delong_results.append({
                    'Baseline_Model': model,
                    'SPARC_C_Index': np.mean(sparc_scores),
                    'Baseline_C_Index': np.mean(df_cv[model].values) if model in df_cv.columns else np.nan,
                    'DeLong_Z_Score': z_score,
                    'DeLong_P_Value': p_value,
                    'Significant': "?" if p_value < 0.05 else "?"
                })
                
                print(f"      DeLong Z-score: {z_score:.3f}")
                print(f"      DeLong p-value: {p_value:.4f}")
                if p_value < 0.001:
                    print(f"      Result: HIGHLY SIGNIFICANT (p < 0.001)")
                elif p_value < 0.01:
                    print(f"      Result: VERY SIGNIFICANT (p < 0.01)")
                elif p_value < 0.05:
                    print(f"      Result: SIGNIFICANT (p < 0.05)")
                else:
                    print(f"      Result: Not significant (p = {p_value:.4f})")
                
            except Exception as e:
                print(f"      DeLong test failed: {e}")
                import traceback
                traceback.print_exc()
        
        if delong_results:
            df_delong = pd.DataFrame(delong_results)
            delong_file = os.path.join(output_dir, f'delong_tests_{threshold}m_{subgroup}.csv')
            df_delong.to_csv(delong_file, index=False)
            print(f"\n DeLong tests saved to: {delong_file}")
            
            # Print summary
            print(f"\n{'='*80}")
            print("DELONG TEST SUMMARY")
            print(f"{'='*80}")
            for _, row in df_delong.iterrows():
                print(f"{row['Baseline_Model']:30s} | Z={row['DeLong_Z_Score']:6.3f} | p={row['DeLong_P_Value']:.4f} | {row['Significant']}")
            print(f"{'='*80}")
            
            return df_ci, df_stat, df_delong
    else:
        print("\n SPARC test predictions not available for DeLong test")
    
    return df_ci, df_stat, None


def plot_cv_results(cv_results, output_dir, threshold, subgroup='Overall'):

    df_cv = pd.DataFrame(cv_results)
    
    means = df_cv.mean()
    stds = df_cv.std()
    sorted_idx = means.sort_values(ascending=False).index
    
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    
    x_pos = np.arange(len(sorted_idx))
    colors = ['#e74c3c' if 'SPARC' in m else '#3498db' for m in sorted_idx]
    
    bars = ax.bar(x_pos, means[sorted_idx], yerr=stds[sorted_idx], 
                  capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('C-index', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_title(f'{subgroup} - {threshold}m - Cross-Validation Results (Matched Folds)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_idx, rotation=45, ha='right', fontsize=12)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random (0.5)', linewidth=2)
    ax.axhline(y=0.7, color='orange', linestyle='--', label='Good (0.7)', linewidth=2)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, mean, std) in enumerate(zip(bars, means[sorted_idx], stds[sorted_idx])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}+/-{std:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f'cv_comparison_{threshold}m_{subgroup}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n CV plot saved to: {plot_file}")

def run_comprehensive_analysis():
    print("="*80)
    print("COMPREHENSIVE BASELINE COMPARISON WITH CV-MATCHED FOLDS")
    print("="*80)
    print(f"Using CV random_state: {CV_RANDOM_STATE} (must match SPARC!)")
    
    all_cv_results = {}
    all_ci_results = {}
    all_stat_results = {}
    all_delong_results = {}
    
    for threshold in THRESHOLDS:
        for subgroup_name in SUBGROUPS.keys():
            print(f"\n{'='*80}")
            print(f"Processing: {subgroup_name} - {threshold}m")
            print(f"{'='*80}")
            
            sparc_used_cv = detect_sparc_training_mode(threshold, subgroup_name)
            
            df = load_original_dataset(threshold, subgroup_name)
            
            if df is None:
                print(f"  failed to load dataset")
                continue
            
            if len(df) < 30:
                print(f"  Skipping (n={len(df)} too small)")
                continue
            
            cv_output = cross_validation_comparison_matched(df, threshold, subgroup_name, n_folds=N_FOLDS)
            
            if cv_output is None:
                continue
            
            cv_results, baseline_preds, sparc_preds, test_events, test_times = cv_output
            all_cv_results[(threshold, subgroup_name)] = cv_results
            
            df_ci, df_stat, df_delong = compute_statistical_tests_with_delong(
                cv_results, baseline_preds, sparc_preds, test_events, test_times,
                BASE_OUTPUT_DIR, threshold, subgroup_name
            )
            
            if df_ci is not None:
                all_ci_results[(threshold, subgroup_name)] = df_ci
            if df_stat is not None:
                all_stat_results[(threshold, subgroup_name)] = df_stat
            if df_delong is not None:
                all_delong_results[(threshold, subgroup_name)] = df_delong
            
            plot_cv_results(cv_results, BASE_OUTPUT_DIR, threshold, subgroup_name)
    
    print(f"\n{'='*80}")
    print("SAVING COMBINED RESULTS")
    print(f"{'='*80}")
    
    if all_ci_results:
        combined_ci = []
        for (th, sg), df_ci in all_ci_results.items():
            df_ci['Threshold'] = th
            df_ci['Subgroup'] = sg
            combined_ci.append(df_ci)
        df_combined_ci = pd.concat(combined_ci, ignore_index=True)
        df_combined_ci.to_csv(os.path.join(BASE_OUTPUT_DIR, 'combined_confidence_intervals.csv'), index=False)
    
    if all_stat_results:
        combined_stat = []
        for (th, sg), df_stat in all_stat_results.items():
            df_stat['Threshold'] = th
            df_stat['Subgroup'] = sg
            combined_stat.append(df_stat)
        df_combined_stat = pd.concat(combined_stat, ignore_index=True)
        df_combined_stat.to_csv(os.path.join(BASE_OUTPUT_DIR, 'combined_statistical_tests.csv'), index=False)
    
    if all_delong_results:
        combined_delong = []
        for (th, sg), df_delong in all_delong_results.items():
            df_delong['Threshold'] = th
            df_delong['Subgroup'] = sg
            combined_delong.append(df_delong)
        df_combined_delong = pd.concat(combined_delong, ignore_index=True)
        df_combined_delong.to_csv(os.path.join(BASE_OUTPUT_DIR, 'combined_delong_tests.csv'), index=False)
    
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE WITH VALID STATISTICAL TESTS")
    print("="*80)
    print(f"Results saved in: {BASE_OUTPUT_DIR}")

if __name__ == "__main__":

    run_comprehensive_analysis()
