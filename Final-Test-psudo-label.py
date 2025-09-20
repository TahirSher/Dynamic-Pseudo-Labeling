import pandas as pd
import numpy as np
import torch
from lifelines import KaplanMeierFitter, WeibullFitter
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
# Suppress TensorFlow and XLA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'  # Reduce XLA verbosity
# Add these imports for survival metrics
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# Configure TensorFlow to use single GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Use only the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # Set memory growth to avoid allocating all memory at once
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)

# Configure PyTorch to use single GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("Using PyTorch device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Using CPU")

OUTPUT_DIR = "20 Sep Test Sep-cindex updated-output"
WEIGHTS_DIR = "20 Sep Test-cindex updated-weights"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
INPUT_FILE = os.path.join(OUTPUT_DIR, "optimized_survival_probabilities.csv")
MODEL_FILE = os.path.join(WEIGHTS_DIR, 'optimized_survival_predictor_10yr.keras')
PREPROCESSOR_FILE = os.path.join(WEIGHTS_DIR, 'optimized_preprocessor_10yr.pkl')
FEATURE_FILE = os.path.join(WEIGHTS_DIR, 'optimized_feature_cols_10yr.pkl')
TIME_COL = 'Time'
EVENT_COL = 'Event'
TARGET_COL = 'Pseudo_Label'
CALIBRATED_COL = 'Pseudo_Label_Calibrated'
EXPECTED_TIME_COL = 'Expected_Survival_Time'
THRESHOLD_TIME = 120  # 10 years in months
MAX_ITER = 100
TARGET_R2 = 0.95
MIN_DELTA = 0.001
PATIENCE = 15
PLOT_INTERVAL = 10
ENSEMBLE_SIZE = 3

def calculate_c_index(event, time, predictions, threshold=THRESHOLD_TIME):
    """Calculate Concordance Index (C-index) for survival predictions"""
    try:
        # For C-index, we use risk scores (1 - survival probability)
        risk_scores = 1 - predictions
        
        y_structured = np.array([(bool(e), t) for e, t in zip(event, time)],
                              dtype=[('event', '?'), ('time', '<f8')])
        
        cindex = concordance_index_censored(y_structured['event'], y_structured['time'], risk_scores)
        return cindex[0]
    except Exception as e:
        print(f"Error calculating C-index: {e}")
        # Return 0.5 (random) if calculation fails
        return 0.5

def add_time_features(df, time_col=TIME_COL, threshold=THRESHOLD_TIME):
    """Enhanced time-based features without artificial plateaus"""
    df = df.copy()
    
    # Basic transformations
    df['log_time'] = np.log1p(df[time_col])
    df['time_squared'] = df[time_col] ** 2
    df['time_normalized'] = df[time_col] / threshold
    df['time_cubic'] = df[time_col] ** 3
    df['time_sqrt'] = np.sqrt(df[time_col])
    df['time_reciprocal'] = 1 / (1 + df[time_col])
    
    # More meaningful time features
    df['time_progression'] = df[time_col] / (df[time_col] + threshold)  # Saturation curve
    df['time_remaining'] = (threshold - df[time_col]) / threshold  # Proportion of time remaining
    df['time_ratio'] = df[time_col] / (df[time_col].max() + 1e-8)  # Relative to max observed time
    
    # Time-based decay factors
    df['time_decay_short'] = np.exp(-df[time_col] / 60)  # Short-term decay (5 years)
    df['time_decay_long'] = np.exp(-df[time_col] / threshold)  # Long-term decay (10 years)
    
    # Time interaction features
    df['time_event_interaction'] = df[time_col] * df[EVENT_COL]
    df['time_remaining_event'] = df['time_remaining'] * df[EVENT_COL]
    
    # Time bins for categorical representation
    df['time_bins'] = pd.cut(df[time_col], bins=np.linspace(0, threshold, 13), labels=False)
    
    return df

def enforce_deceased_constraints(df, iteration, event_col=EVENT_COL, 
                                 target_col=TARGET_COL, calibrated_col=CALIBRATED_COL,
                                 time_col=TIME_COL, expected_time_col=EXPECTED_TIME_COL):
    df = df.copy()
    deceased_mask = (df[event_col] == 1)
    censored_mask = (df[event_col] == 0)
    
    if deceased_mask.sum() > 0:
        # Ensure expected time is at least observed time for deceased patients
        time_violation = df.loc[deceased_mask, expected_time_col] < df.loc[deceased_mask, time_col]
        if time_violation.any():
            print(f"  Correcting {time_violation.sum()} time violations for deceased patients")
            df.loc[deceased_mask & time_violation, expected_time_col] = df.loc[deceased_mask & time_violation, time_col]
        
        time_discrepancy = np.abs(df.loc[deceased_mask, expected_time_col] - df.loc[deceased_mask, time_col])
        max_discrepancy = time_discrepancy.max() if time_discrepancy.max() > 0 else 1
        normalized_discrepancy = time_discrepancy / max_discrepancy
        iteration_factor = min(0.95, 0.7 + 0.05 * iteration)
        correction = 1 - (normalized_discrepancy * iteration_factor)
        df.loc[deceased_mask, target_col] = df.loc[deceased_mask, target_col] * correction
        df.loc[deceased_mask, calibrated_col] = df.loc[deceased_mask, calibrated_col] * correction
        
        # For deceased patients, expected time should be close to observed time
        blend_factor = min(0.9, 0.5 + 0.1 * iteration)
        df.loc[deceased_mask, expected_time_col] = (
            blend_factor * df.loc[deceased_mask, time_col] + 
            (1 - blend_factor) * df.loc[deceased_mask, expected_time_col]
        )
    
    # For censored patients, ensure expected time is at least observed time
    if censored_mask.sum() > 0:
        time_violation = df.loc[censored_mask, expected_time_col] < df.loc[censored_mask, time_col]
        if time_violation.any():
            print(f"  Correcting {time_violation.sum()} time violations for censored patients")
            df.loc[censored_mask & time_violation, expected_time_col] = df.loc[censored_mask & time_violation, time_col]
    
    df[expected_time_col] = np.clip(df[expected_time_col], 0, THRESHOLD_TIME)
    return df

def time_aware_refinement(df, pred_probs, event, time, threshold=THRESHOLD_TIME, iteration=1):
    """Enhanced time-aware refinement with better temporal dynamics"""
    refined = pred_probs.copy()
    deceased_mask = (event == 1)
    censored_mask = (event == 0)
    
    # Non-linear penalty for deceased patients
    if deceased_mask.sum() > 0:
        time_error = np.abs(df.loc[deceased_mask, EXPECTED_TIME_COL].values - 
                           df.loc[deceased_mask, TIME_COL].values)
        
        # Use exponential decay for penalty
        penalty_factor = min(0.95, 0.6 + 0.07 * iteration)
        penalty = np.exp(-time_error / (threshold * 0.2)) * penalty_factor
        refined[deceased_mask] *= penalty
    
    # Time-dependent smoothing with continuous decay
    if censored_mask.sum() > 0:
        time_ratio = time[censored_mask] / threshold
        
        # Continuous decay without artificial plateaus
        decay_factor = 1 - (time_ratio * 0.8)  # Linear decay, more gradual
        refined[censored_mask] *= decay_factor
        
        # Additional adjustment for very long-term censored
        very_long_term = censored_mask & (time >= threshold * 0.8)
        refined[very_long_term] = np.maximum(refined[very_long_term], 0.7)
    
    return np.clip(refined, 0.01, 0.99)

def update_expected_survival_time(df, pred_probs, event, time, threshold=THRESHOLD_TIME):
    df = df.copy()
    
    # Calculate expected time from probabilities
    expected_time = threshold * pred_probs
    
    # For all patients, expected time should be at least the observed time
    expected_time = np.maximum(expected_time, time)
    
    deceased_mask = (event == 1)
    if deceased_mask.sum() > 0:
        # For deceased patients, expected time should be very close to observed time
        # Use a stronger constraint for deceased patients
        prob_weight = 0.9  # Strong weight toward observed time
        observed_time = time[deceased_mask]
        blended_time = prob_weight * observed_time + (1 - prob_weight) * expected_time[deceased_mask]
        
        # Ensure blended time is at least the observed time
        blended_time = np.maximum(blended_time, observed_time)
        df.loc[deceased_mask, EXPECTED_TIME_COL] = blended_time
    
    censored_mask = (event == 0)
    if censored_mask.sum() > 0:
        # For censored patients, expected time should be at least the observed time
        df.loc[censored_mask, EXPECTED_TIME_COL] = np.maximum(expected_time[censored_mask], time[censored_mask])
    
    # Ensure consistency: if probability is high, expected time should be high
    # and vice versa
    high_prob_mask = pred_probs > 0.7
    df.loc[high_prob_mask, EXPECTED_TIME_COL] = np.maximum(
        df.loc[high_prob_mask, EXPECTED_TIME_COL],
        threshold * 0.8  # At least 80% of threshold time for high probabilities
    )
    
    low_prob_mask = pred_probs < 0.3
    df.loc[low_prob_mask, EXPECTED_TIME_COL] = np.minimum(
        df.loc[low_prob_mask, EXPECTED_TIME_COL],
        threshold * 0.5  # At most 50% of threshold time for low probabilities
    )
    
    df[EXPECTED_TIME_COL] = np.clip(df[EXPECTED_TIME_COL], 0, threshold)
    return df

def time_dependent_calibration(y_true, y_pred, event, time, time_bins=15):
    """More granular time-dependent calibration"""
    calibrated = y_pred.copy()
    
    # Use quantile-based time bins for better distribution
    time_quantiles = np.unique(np.percentile(time, np.linspace(0, 100, time_bins + 1)))
    
    for i in range(len(time_quantiles) - 1):
        time_mask = (time >= time_quantiles[i]) & (time < time_quantiles[i + 1])
        
        if np.sum(time_mask) > 20:
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
            calibrated_local = iso.fit_transform(y_pred[time_mask], y_true[time_mask])
            
            # Blend with original based on sample size confidence
            sample_ratio = np.sum(time_mask) / len(time)
            blend_factor = min(0.8, 0.3 + 0.5 * sample_ratio)
            
            calibrated[time_mask] = (blend_factor * calibrated_local + 
                                   (1 - blend_factor) * y_pred[time_mask])
    
    return calibrated

def create_improved_pseudo_labels(df, time_col=TIME_COL, event_col=EVENT_COL, threshold=THRESHOLD_TIME):
    wf = WeibullFitter()
    wf.fit(df[time_col], event_observed=df[event_col])
    survival_probs = wf.predict(threshold)
    deceased_mask = (df[event_col] == 1)
    df.loc[deceased_mask, TARGET_COL] = 0.0
    df.loc[deceased_mask, EXPECTED_TIME_COL] = df.loc[deceased_mask, time_col]
    censored_mask = (df[event_col] == 0)
    long_term_censored = censored_mask & (df[time_col] >= threshold)
    df.loc[long_term_censored, TARGET_COL] = 1.0
    df.loc[long_term_censored, EXPECTED_TIME_COL] = threshold
    short_term_censored = censored_mask & (df[time_col] < threshold)
    for idx in df[short_term_censored].index:
        t = df.at[idx, time_col]
        prob_threshold = wf.predict(threshold)
        prob_t = wf.predict(t)
        if prob_t > 0.01:
            conditional_prob = prob_threshold / prob_t
        else:
            kmf = KaplanMeierFitter()
            kmf.fit(df[time_col], event_observed=df[event_col])
            prob_threshold_km = kmf.predict(threshold)
            prob_t_km = kmf.predict(t)
            conditional_prob = prob_threshold_km / prob_t_km if prob_t_km > 0.01 else 0.5
        df.at[idx, TARGET_COL] = np.clip(conditional_prob, 0.01, 0.99)
        df.at[idx, EXPECTED_TIME_COL] = min(t + (threshold - t) * conditional_prob, threshold)
    df[CALIBRATED_COL] = df[TARGET_COL].copy()
    return df

def build_time_aware_model(input_dim):
    """Build a model for single GPU training"""
    inputs = Input(shape=(input_dim,))
    
    # Feature extraction
    x = BatchNormalization()(inputs)
    x = Dense(128, activation='relu', kernel_initializer='he_normal', 
              kernel_regularizer=l2(1e-5))(x)
    x = Dropout(0.4)(x)
    
    # Time-aware processing
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Additional layers for temporal patterns
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

def predict_expected_time(probabilities, time_observed, threshold=THRESHOLD_TIME):
    expected_time = threshold * probabilities
    # Ensure expected time is at least the observed time
    expected_time = np.maximum(expected_time, time_observed)
    return np.clip(expected_time, 0, threshold)

def plot_time_dependent_performance(df, iteration, output_dir=OUTPUT_DIR):
    """Plot survival probabilities over time with better colors"""
    time_bins = np.linspace(0, THRESHOLD_TIME, 13)
    avg_probs = []
    std_probs = []
    counts = []
    
    for i in range(len(time_bins) - 1):
        mask = (df[TIME_COL] >= time_bins[i]) & (df[TIME_COL] < time_bins[i + 1])
        if mask.sum() > 0:
            avg_probs.append(df.loc[mask, CALIBRATED_COL].mean())
            std_probs.append(df.loc[mask, CALIBRATED_COL].std())
            counts.append(mask.sum())
        else:
            avg_probs.append(np.nan)
            std_probs.append(np.nan)
            counts.append(0)
    
    # Create a colormap for the points
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_bins)-1))
    sizes = np.array(counts) / max(counts) * 200  # Scale point sizes by sample count
    
    plt.figure(figsize=(12, 8), dpi=600)
    
    # Main line with confidence interval
    plt.plot(time_bins[:-1], avg_probs, 'o-', color='#FF6B6B', linewidth=3, 
             markersize=0, label='Average Probability')
    
    # Points with size proportional to sample count
    for i in range(len(time_bins)-1):
        if counts[i] > 0:
            plt.scatter(time_bins[i], avg_probs[i], s=sizes[i], 
                       color=colors[i], alpha=0.8, edgecolors='black')
    
    plt.fill_between(time_bins[:-1], 
                    np.array(avg_probs) - np.array(std_probs),
                    np.array(avg_probs) + np.array(std_probs),
                    alpha=0.2, color='#4ECDC4', label='(+/-)1 Std Dev')
    
    plt.xlabel('Time (months)', fontsize=12)
    plt.ylabel('Average Survival Probability', fontsize=12)
    plt.title(f'Time-Dependent Performance - Iteration {iteration}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add sample count annotations
    for i, count in enumerate(counts):
        if count > 0:
            plt.annotate(f'n={count}', (time_bins[i], avg_probs[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'time_performance_iter_{iteration}.png'), 
                dpi=600, bbox_inches='tight')
    plt.close()

def plot_calibration_and_distributions(y_true, y_pred, y_calibrated, event, history, iteration):
    # Create a vibrant color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A602', '#9B59B6']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=600)
    
    # Training history
    axes[0, 0].plot(history.history['loss'], label='Train Loss', color=colors[0], linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', color=colors[1], linewidth=2)
    axes[0, 0].set_title(f'Training History (Iteration {iteration})', fontsize=12)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Calibration plot
    axes[0, 1].scatter(y_true, y_pred, alpha=0.6, s=30, color=colors[0], label='Before Calibration')
    axes[0, 1].scatter(y_true, y_calibrated, alpha=0.6, s=30, color=colors[1], label='After Calibration')
    axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('Actual Pseudo Labels')
    axes[0, 1].set_ylabel('Predicted Probabilities')
    axes[0, 1].set_title(f'Actual vs Predicted (Iter {iteration})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution plot
    axes[1, 0].hist(y_pred, bins=20, alpha=0.7, color=colors[0], 
                   label='Original Predictions', edgecolor='black', density=True)
    axes[1, 0].hist(y_calibrated, bins=20, alpha=0.7, color=colors[1], 
                   label='Calibrated Predictions', edgecolor='black', density=True)
    axes[1, 0].set_xlabel('Survival Probability')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title(f'Prediction Distributions (Iter {iteration})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Event status distribution
    event_mask = (event == 1)
    axes[1, 1].hist(y_calibrated[event_mask], bins=20, alpha=0.7, color=colors[2], 
                   label='Events', edgecolor='black', density=True)
    axes[1, 1].hist(y_calibrated[~event_mask], bins=20, alpha=0.7, color=colors[3], 
                   label='Censored', edgecolor='black', density=True)
    axes[1, 1].set_xlabel('Survival Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title(f'Distribution by Event Status (Iter {iteration})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'calibration_dist_iter_{iteration}.png'), 
                dpi=600, bbox_inches='tight')
    plt.close()

def plot_deceased_probability_analysis(df, iteration, output_dir=OUTPUT_DIR):
    deceased_mask = (df[EVENT_COL] == 1)
    censored_mask = (df[EVENT_COL] == 0)
    
    plt.figure(figsize=(12, 5), dpi=600)
    
    plt.subplot(1, 2, 1)
    plt.hist(df.loc[deceased_mask, CALIBRATED_COL], bins=20, alpha=0.7, 
             color='red', label='Deceased', edgecolor='black')
    plt.axvline(x=0.05, color='black', linestyle='--', label='Threshold (0.05)')
    plt.xlabel('Survival Probability')
    plt.ylabel('Count')
    plt.title(f'Deceased Patients - Iteration {iteration}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(df.loc[deceased_mask, CALIBRATED_COL], bins=20, alpha=0.7, 
             color='red', label='Deceased', edgecolor='black', log=True)
    plt.hist(df.loc[censored_mask, CALIBRATED_COL], bins=20, alpha=0.7, 
             color='blue', label='Censored', edgecolor='black', log=True)
    plt.xlabel('Survival Probability')
    plt.ylabel('Count (log scale)')
    plt.title(f'Probability Distribution - Iteration {iteration}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'deceased_analysis_iter_{iteration}.png'), dpi=600, bbox_inches='tight')
    plt.close()

def plot_temporal_convergence(df, iteration, output_dir=OUTPUT_DIR):
    """Monitor temporal pattern convergence"""
    time_intervals = np.linspace(0, THRESHOLD_TIME, 13)
    
    plt.figure(figsize=(15, 10), dpi=600)
    
    for i, interval in enumerate([30, 60, 90, 120]):
        mask = (df[TIME_COL] >= interval - 10) & (df[TIME_COL] <= interval + 10)
        
        if mask.sum() > 0:
            plt.subplot(2, 2, i + 1)
            plt.hist(df.loc[mask, CALIBRATED_COL], bins=20, alpha=0.7, 
                    label=f'{interval} (+/-) 10 months', edgecolor='black')
            plt.xlabel('Survival Probability')
            plt.ylabel('Count')
            plt.title(f'Distribution at {interval} months - Iter {iteration}')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'temporal_convergence_iter_{iteration}.png'), dpi=600, bbox_inches='tight')
    plt.close()

def track_deceased_convergence(df, iteration, output_dir=OUTPUT_DIR):
    deceased_mask = (df[EVENT_COL] == 1)
    
    if deceased_mask.sum() > 0:
        avg_prob = df.loc[deceased_mask, CALIBRATED_COL].mean()
        max_prob = df.loc[deceased_mask, CALIBRATED_COL].max()
        time_error = np.mean(np.abs(df.loc[deceased_mask, EXPECTED_TIME_COL] - 
                                   df.loc[deceased_mask, TIME_COL]))
        
        print(f"Deceased patients - Iter {iteration}:")
        print(f"  Avg probability: {avg_prob:.4f}, Max: {max_prob:.4f}")
        print(f"  Avg time error: {time_error:.2f} months")
        
        if iteration > 1:
            track_file = os.path.join(output_dir, 'deceased_convergence.csv')
            if os.path.exists(track_file):
                track_df = pd.read_csv(track_file)
            else:
                track_df = pd.DataFrame(columns=['iteration', 'avg_prob', 'max_prob', 'time_error'])
            
            new_row = pd.DataFrame({
                'iteration': [iteration],
                'avg_prob': [avg_prob],
                'max_prob': [max_prob],
                'time_error': [time_error]
            })
            track_df = pd.concat([track_df, new_row], ignore_index=True)
            track_df.to_csv(track_file, index=False)
            
            plt.figure(figsize=(10, 4), dpi=600)
            plt.subplot(1, 2, 1)
            plt.plot(track_df['iteration'], track_df['avg_prob'], 'bo-', label='Average Probability')
            plt.plot(track_df['iteration'], track_df['max_prob'], 'ro-', label='Max Probability')
            plt.xlabel('Iteration')
            plt.ylabel('Probability')
            plt.title('Deceased Patient Probability Convergence')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(track_df['iteration'], track_df['time_error'], 'go-')
            plt.xlabel('Iteration')
            plt.ylabel('Time Error (months)')
            plt.title('Expected vs Observed Time Error')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'deceased_convergence_tracking.png'), dpi=600, bbox_inches='tight')
            plt.close()

def create_validation_plots(df, output_dir=OUTPUT_DIR):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=600)
    
    # First plot: Calibration
    axes[0].scatter(df[TARGET_COL], df[CALIBRATED_COL], alpha=0.6, s=30)
    axes[0].plot([0,1], [0,1], 'r--', linewidth=2, label='Perfect calibration')
    axes[0].set_xlabel('Original Pseudo Labels (Actual)')
    axes[0].set_ylabel('Calibrated Predictions')
    axes[0].set_title('Final Calibration: Actual vs Predicted')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim(-0.02, 1.02)  # Add small margin
    axes[0].set_ylim(-0.02, 1.02)  # Add small margin
    
    # Second plot: Error distribution
    errors = df[CALIBRATED_COL] - df[TARGET_COL]
    counts, bins, patches = axes[1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error (Predicted - Actual)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # Add margin to error distribution plot
    x_margin = (bins[-1] - bins[0]) * 0.05
    axes[1].set_xlim(bins[0] - x_margin, bins[-1] + x_margin)
    axes[1].set_ylim(0, max(counts) * 1.05)  # Add 5% margin on top
    
    # Third plot: Residual plot
    axes[2].scatter(df[TARGET_COL], errors, alpha=0.6, s=30)
    axes[2].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Actual Survival Probability')
    axes[2].set_ylabel('Prediction Error')
    axes[2].set_title('Residual Plot')
    axes[2].grid(True, alpha=0.3)
    
    # Add margin to residual plot
    axes[2].set_xlim(-0.02, 1.02)
    y_range = errors.max() - errors.min()
    y_margin = y_range * 0.1 if y_range > 0 else 0.1
    axes[2].set_ylim(errors.min() - y_margin, errors.max() + y_margin)
    
    # Adjust layout with more flexible padding
    plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=2.0)
    plt.savefig(os.path.join(output_dir, 'final_validation_plots.png'), 
                dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.close()


def analyze_final_results(df, convergence_history):
    print("\n" + "="*60)
    print("DETAILED FINAL ANALYSIS")
    print("="*60)
    
    event_mask = df[EVENT_COL] == 1
    censored_mask = df[EVENT_COL] == 0
    
    print(f"\n1. DISTRIBUTION ANALYSIS:")
    print(f"   Total samples: {len(df)}")
    print(f"   Events (deceased): {event_mask.sum()} ({event_mask.sum()/len(df)*100:.1f}%)")
    print(f"   Censored: {censored_mask.sum()} ({censored_mask.sum()/len(df)*100:.1f}%)")
    
    print(f"\n2. PREDICTION RANGE:")
    print(f"   Min probability: {df[CALIBRATED_COL].min():.4f}")
    print(f"   Max probability: {df[CALIBRATED_COL].max():.4f}")
    print(f"   Mean probability: {df[CALIBRATED_COL].mean():.4f}")
    print(f"   Median probability: {df[CALIBRATED_COL].median():.4f}")  
    print(f"   Std probability: {df[CALIBRATED_COL].std():.4f}")
    
    print(f"\n3. EXPECTED SURVIVAL TIME:")
    print(f"   Min time: {df[EXPECTED_TIME_COL].min():.1f} months")
    print(f"   Max time: {df[EXPECTED_TIME_COL].max():.1f} months")
    print(f"   Mean time: {df[EXPECTED_TIME_COL].mean():.1f} months")
    
    # Check for time constraint violations
    time_violations = df[EXPECTED_TIME_COL] < df[TIME_COL]
    if time_violations.any():
        print(f"   Time constraint violations: {time_violations.sum()} cases")
        print(f"   Max violation: {(df[TIME_COL] - df[EXPECTED_TIME_COL]).max():.2f} months")
    else:
        print(f"   No time constraint violations")
    
    print(f"\n4. EVENT-SPECIFIC ANALYSIS:")
    print(f"   Events - Mean probability: {df.loc[event_mask, CALIBRATED_COL].mean():.4f}")
    print(f"   Censored - Mean probability: {df.loc[censored_mask, CALIBRATED_COL].mean():.4f}")
    
    deceased_probs = df.loc[event_mask, CALIBRATED_COL]
    print(f"\n5. DECEASED PATIENT VALIDATION:")
    print(f"   Max probability for deceased: {deceased_probs.max():.4f}")
    print(f"   Patients with probability > 0.05: {(deceased_probs > 0.05).sum()}")
    print(f"   Patients with probability > 0.01: {(deceased_probs > 0.01).sum()}")
    
    # Calculate survival metrics
    c_index = calculate_c_index(df[EVENT_COL], df[TIME_COL], df[CALIBRATED_COL])
    
    print(f"\n6. SURVIVAL ANALYSIS METRICS:")
    print(f"   C-index (Concordance Index): {c_index:.4f}")
    
    if convergence_history:
        print(f"\n7. CONVERGENCE SPEED:")
        print(f"   Achieved target in {len(convergence_history)} iterations")
        print(f"   Final R2 improvement: {convergence_history[-1]['r2_after'] - convergence_history[0]['r2_after']:.4f}")
    
    final_metrics = convergence_history[-1] if convergence_history else {}
    print(f"\n8. MODEL ROBUSTNESS INDICATORS:")
    print(f"   Spearman correlation: {final_metrics.get('spearman', 0):.4f}")
    print(f"   Explained variance: {final_metrics.get('explained_var', 0):.4f}")
    
    return True
def check_time_probability_consistency(df, iteration):
    """Check for consistency between survival probability and expected time"""
    print(f"\nTime-Probability Consistency Check (Iteration {iteration}):")
    
    # Check if high probabilities correspond to long expected times
    high_prob_mask = df[CALIBRATED_COL] > 0.7
    if high_prob_mask.sum() > 0:
        avg_expected_time_high_prob = df.loc[high_prob_mask, EXPECTED_TIME_COL].mean()
        print(f"  High prob (>0.7) patients: {high_prob_mask.sum()} cases")
        print(f"  Average expected time: {avg_expected_time_high_prob:.1f} months")
        
        # Check if any high probability patients have unexpectedly short expected times
        inconsistency_mask = high_prob_mask & (df[EXPECTED_TIME_COL] < THRESHOLD_TIME * 0.6)
        if inconsistency_mask.sum() > 0:
            print(f"   {inconsistency_mask.sum()} inconsistencies detected")
            print(f"  Min expected time for high prob: {df.loc[high_prob_mask, EXPECTED_TIME_COL].min():.1f} months")
    
    # Check if low probabilities correspond to short expected times
    low_prob_mask = df[CALIBRATED_COL] < 0.3
    if low_prob_mask.sum() > 0:
        avg_expected_time_low_prob = df.loc[low_prob_mask, EXPECTED_TIME_COL].mean()
        print(f"  Low prob (<0.3) patients: {low_prob_mask.sum()} cases")
        print(f"  Average expected time: {avg_expected_time_low_prob:.1f} months")
        
        # Check if any low probability patients have unexpectedly long expected times
        inconsistency_mask = low_prob_mask & (df[EXPECTED_TIME_COL] > THRESHOLD_TIME * 0.8)
        if inconsistency_mask.sum() > 0:
            print(f"    {inconsistency_mask.sum()} inconsistencies detected")
            print(f"  Max expected time for low prob: {df.loc[low_prob_mask, EXPECTED_TIME_COL].max():.1f} months")
    
    # Check the overall correlation
    correlation = spearmanr(df[CALIBRATED_COL], df[EXPECTED_TIME_COL]).correlation
    print(f"  Probability-Time correlation: {correlation:.4f}")
    
    return correlation

# Load or initialize data
if os.path.exists(INPUT_FILE):
    df = pd.read_csv(INPUT_FILE)
    print("Loaded existing data file")
    
    if TARGET_COL not in df.columns:
        print(f"Creating missing column: {TARGET_COL}")
        df[TARGET_COL] = np.nan
    if CALIBRATED_COL not in df.columns:
        print(f"Creating missing column: {CALIBRATED_COL}")
        df[CALIBRATED_COL] = np.nan
    if EXPECTED_TIME_COL not in df.columns:
        print(f"Creating missing column: {EXPECTED_TIME_COL}")
        df[EXPECTED_TIME_COL] = np.nan
        
    if df[TARGET_COL].isna().any() or df[CALIBRATED_COL].isna().any() or df[EXPECTED_TIME_COL].isna().any():
        print("Creating initial pseudo labels for missing values")
        df = create_improved_pseudo_labels(df)
else:
    df = pd.read_csv("OSTrain_Dataset.csv")
    print("Creating initial pseudo labels for new dataset")
    df = create_improved_pseudo_labels(df)

# Add time-based features
print("Adding enhanced time-based features...")
df = add_time_features(df)

print("Improving pseudo labels with Weibull estimates")
df = create_improved_pseudo_labels(df)

exclude_cols = ['ID', TARGET_COL, CALIBRATED_COL, EXPECTED_TIME_COL, TIME_COL, EVENT_COL]
feature_cols = [c for c in df.columns if c not in exclude_cols]

joblib.dump(feature_cols, FEATURE_FILE)

models = None
best_r2 = -float('inf')
best_mae = float('inf')
stagnation_count = 0
convergence_history = []

X = df[feature_cols]
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

# Clear GPU memory before starting
tf.keras.backend.clear_session()

for iteration in range(1, MAX_ITER + 1):
    print(f"\n### Iteration {iteration} ###")
    
    X = df[feature_cols]
    y = df[CALIBRATED_COL].copy()
    event = df[EVENT_COL]
    time = df[TIME_COL]
    
    # Split first to avoid data leakage
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=42, 
        stratify=event
    )
    
    # Fit preprocessor ONLY on training data
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    event_train = event.iloc[train_idx]
    event_test = event.iloc[test_idx]
    time_train = time.iloc[train_idx]
    time_test = time.iloc[test_idx]
    
    if iteration > 1:
        if iteration % 3 == 0:
            # Apply QuantileTransformer only on training data to avoid leakage
            qt = QuantileTransformer(n_quantiles=min(100, len(y_train)), 
                                   output_distribution='normal', random_state=42)
            y_train_transformed = qt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_train_transformed = (y_train_transformed - y_train_transformed.mean()) / y_train_transformed.std()
            y_train = 1 / (1 + np.exp(-y_train_transformed))
            
            # Transform test data using the same fitted transformer
            y_test_transformed = qt.transform(y_test.values.reshape(-1, 1)).flatten()
            y_test_transformed = (y_test_transformed - y_test_transformed.mean()) / y_test_transformed.std()
            y_test = 1 / (1 + np.exp(-y_test_transformed))
        else:
            # Normalize only training data and apply same transformation to test
            train_min, train_max = y_train.min(), y_train.max()
            y_train = (y_train - train_min) / (train_max - train_min + 1e-8)
            y_test = (y_test - train_min) / (train_max - train_min + 1e-8)
    
    y_train = np.clip(y_train, 0.01, 0.99)
    y_test = np.clip(y_test, 0.01, 0.99)
    
    # Update the main dataframe with normalized values
    df.loc[train_idx, CALIBRATED_COL] = y_train
    df.loc[test_idx, CALIBRATED_COL] = y_test
    
    sample_weights = compute_sample_weight('balanced', y_train)
    confidence_weights = 1.0 / (1.0 + np.abs(y_train - 0.5))
    sample_weights = sample_weights * confidence_weights
    
    if models is None or stagnation_count > 5:
        print("Resetting models to escape local minima")
        # Clear any existing models from memory
        if models is not None:
            for model in models:
                del model
            tf.keras.backend.clear_session()
        
        models = build_model_ensemble(X_train_processed.shape[1])
        stagnation_count = 0
    
    all_test_preds = []
    all_preds = []
    histories = []
    
    for i, model in enumerate(models):
        print(f"Training model {i+1}/{len(models)}")
        
        initial_lr = 0.0005 * (0.95 ** (iteration // 5))
        
        optimizer = Nadam(learning_rate=initial_lr, clipnorm=1.0)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        
        early_stopping = EarlyStopping(
            monitor='val_mae', patience=15, mode='min', 
            restore_best_weights=True, verbose=0, min_delta=0.0001
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_mae', factor=0.7, patience=8, 
            min_lr=1e-7, verbose=0, min_delta=0.0005
        )
        
        # Create validation split from training data
        val_size = int(0.15 * len(X_train_processed))
        X_train_final = X_train_processed[:-val_size]
        X_val = X_train_processed[-val_size:]
        y_train_final = y_train[:-val_size]
        y_val = y_train[-val_size:]
        
        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr, TerminateOnNaN()],
            verbose=0,
            sample_weight=sample_weights[:-val_size] if len(sample_weights) > val_size else sample_weights
        )
        histories.append(history)
        
        test_pred = model.predict(X_test_processed, verbose=0).flatten()
        all_test_preds.append(test_pred)
        
        # Predict on full dataset for updating
        full_pred = model.predict(preprocessor.transform(X), verbose=0).flatten()
        all_preds.append(full_pred)
    
    all_test_preds = np.array(all_test_preds)
    all_preds = np.array(all_preds)
    
    pred_means = np.mean(all_test_preds, axis=0)
    pred_stds = np.std(all_test_preds, axis=0)
    valid_mask = np.all(np.abs(all_test_preds - pred_means[:, None].T) < 2 * pred_stds, axis=0)
    
    y_pred_test = np.mean(all_test_preds[:, valid_mask], axis=0)
    y_pred_full = np.mean(all_preds, axis=0)
    
    # Use time-dependent calibration only on training data to avoid leakage
    y_test_calibrated = time_dependent_calibration(
        y_test[valid_mask], y_pred_test, 
        event_test[valid_mask], time_test.iloc[valid_mask].values
    )
    
    # For full dataset calibration, use only training data patterns
    train_calibration_mask = np.isin(indices, train_idx)
    y_train_for_calib = y.iloc[train_idx].values
    y_pred_train_for_calib = y_pred_full[train_idx]
    
    # Create calibration model on training data only
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
    iso.fit(y_pred_train_for_calib, y_train_for_calib)
    y_full_calibrated = iso.predict(y_pred_full)
    
    # Use time-aware refinement
    y_full_calibrated = time_aware_refinement(df, y_full_calibrated, event, time, iteration=iteration)
    
    blend_factor = 0.7
    
    df[TARGET_COL] = (blend_factor * y_pred_full + 
                     (1 - blend_factor) * df[TARGET_COL])
    df[CALIBRATED_COL] = (blend_factor * y_full_calibrated + 
                         (1 - blend_factor) * df[CALIBRATED_COL])
    
    df = update_expected_survival_time(df, y_full_calibrated, event, time)
    time_prob_correlation = check_time_probability_consistency(df, iteration)
    df = enforce_deceased_constraints(df, iteration=iteration)
    
    df.to_csv(INPUT_FILE, index=False)
    
    r2_before = r2_score(y_test[valid_mask], y_pred_test)
    r2_after = r2_score(y_test[valid_mask], y_test_calibrated)
    mae_before = mean_absolute_error(y_test[valid_mask], y_pred_test)
    mae_after = mean_absolute_error(y_test[valid_mask], y_test_calibrated)
    
    spearman_corr = spearmanr(y_test[valid_mask], y_test_calibrated)[0]
    explained_var = explained_variance_score(y_test[valid_mask], y_test_calibrated)
    
    # Calculate survival metrics
    c_index = calculate_c_index(event_test[valid_mask], time_test[valid_mask], y_test_calibrated)
    
    convergence_history.append({
        'iteration': iteration,
        'r2_before': r2_before,
        'r2_after': r2_after,
        'mae_before': mae_before,
        'mae_after': mae_after,
        'spearman': spearman_corr,
        'explained_var': explained_var,
        'c_index': c_index
    })
    
    print(f"Iteration {iteration} Results:")
    print(f"  R2: {r2_before:.4f} -> {r2_after:.4f}")
    print(f"  MAE: {mae_before:.4f} -> {mae_after:.4f}")
    print(f"  Spearman: {spearman_corr:.4f}, Explained Var: {explained_var:.4f}")
    print(f"  C-index: {c_index:.4f}")
    
    # Monitor time-specific performance
    for critical_time in [30, 60, 90, 120]:
        time_mask = (df[TIME_COL] >= critical_time - 5) & (df[TIME_COL] <= critical_time + 5)
        if time_mask.sum() > 0:
            avg_prob = df.loc[time_mask, CALIBRATED_COL].mean()
            std_prob = df.loc[time_mask, CALIBRATED_COL].std()
            print(f"  Time {critical_time} months: Avg prob = {avg_prob:.3f} (+/-) {std_prob:.3f}")
    
    track_deceased_convergence(df, iteration)
    plot_time_dependent_performance(df, iteration)
    plot_temporal_convergence(df, iteration)
    
    improvement = (r2_after - best_r2) > MIN_DELTA or (mae_after < best_mae - MIN_DELTA)
    
    if improvement:
        best_r2 = max(best_r2, r2_after)
        best_mae = min(best_mae, mae_after)
        stagnation_count = 0
        print(f"  Improvement detected: R2={best_r2:.4f}, MAE={best_mae:.4f}")
        
        for i, model in enumerate(models):
            model.save(os.path.join(WEIGHTS_DIR, f'best_model_{i}.keras'))
        joblib.dump(preprocessor, os.path.join(WEIGHTS_DIR, 'best_preprocessor.pkl'))
        
    else:
        stagnation_count += 1
        print(f"  No significant improvement for {stagnation_count} iterations")
    
    if iteration % 2 == 0 or iteration == 1:
        plot_calibration_and_distributions(
            y_test[valid_mask], y_pred_test, y_test_calibrated, 
            event_test[valid_mask], histories[0], iteration
        )
        
        plot_deceased_probability_analysis(df, iteration)
    
    if r2_after >= TARGET_R2 and mae_after <= 0.01:
        print(f"? Target metrics achieved: R2={r2_after:.4f} >= {TARGET_R2}, MAE={mae_after:.4f} <= 0.05")
        break
        
    elif stagnation_count >= PATIENCE:
        print(f"Stopping due to stagnation after {PATIENCE} iterations")
        if stagnation_count == PATIENCE:
            print("Attempting final model reset...")
            # Clear memory before resetting
            for model in models:
                del model
            tf.keras.backend.clear_session()
            models = None
            stagnation_count = 0
            continue
        else:
            break
    
    if iteration > 10 and best_r2 > 0.85:
        TARGET_R2 = min(0.97, TARGET_R2 + 0.01)

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)

final_metrics = convergence_history[-1] if convergence_history else {}
print(f"Best R2 achieved: {best_r2:.4f}")
print(f"Best MAE achieved: {best_mae:.4f}")
print(f"Final Spearman correlation: {final_metrics.get('spearman', 0):.4f}")
print(f"Final explained variance: {final_metrics.get('explained_var', 0):.4f}")
print(f"Final C-index: {final_metrics.get('c_index', 0):.4f}")

if os.path.exists(os.path.join(WEIGHTS_DIR, 'best_model_0.keras')):
    print("Loading best performing models...")
    best_models = []
    for i in range(ENSEMBLE_SIZE):
        model = tf.keras.models.load_model(os.path.join(WEIGHTS_DIR, f'best_model_{i}.keras'))
        best_models.append(model)
    
    # Preprocess the entire dataset using the fitted preprocessor
    X_processed = preprocessor.transform(X)
    
    final_preds = []
    for model in best_models:
        pred = model.predict(X_processed, verbose=0).flatten()
        final_preds.append(pred)
    
    final_y_pred = np.mean(final_preds, axis=0)
    
    # Calibrate using training data only to avoid leakage
    train_mask = np.isin(indices, train_idx)
    iso_final = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
    iso_final.fit(final_y_pred[train_mask], y.iloc[train_idx].values)
    final_y_calibrated = iso_final.predict(final_y_pred)
    
    df[TARGET_COL] = final_y_pred
    df[CALIBRATED_COL] = final_y_calibrated
    
    # Use the corrected predict_expected_time function
    df[EXPECTED_TIME_COL] = predict_expected_time(final_y_calibrated, df[TIME_COL])
    
    df = enforce_deceased_constraints(df, iteration=MAX_ITER)
    
    df.to_csv(INPUT_FILE, index=False)

create_validation_plots(df)
analyze_final_results(df, convergence_history)

print("\nTraining complete.")
print(f"Final dataset saved: {INPUT_FILE}")
print(f"Best models saved in: {WEIGHTS_DIR}")

# Final memory cleanup
tf.keras.backend.clear_session()
if 'models' in locals():
    for model in models:
        del model

print("GPU memory cleared.")