import os
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.metrics import concordance_index_censored
import config


def filter_subgroup(df, subgroup_name):

    if subgroup_name == 'Overall' or config.SUBGROUPS[subgroup_name] is None:
        return df.copy().reset_index(drop=True)
    
    conditions = config.SUBGROUPS[subgroup_name]
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
        output_dir = os.path.join(config.BASE_OUTPUT_DIR, f"threshold_{threshold}m")
        weights_dir = os.path.join(config.BASE_WEIGHTS_DIR, f"threshold_{threshold}m")
    else:
        output_dir = os.path.join(config.BASE_OUTPUT_DIR, f"threshold_{threshold}m_{subgroup}")
        weights_dir = os.path.join(config.BASE_WEIGHTS_DIR, f"threshold_{threshold}m_{subgroup}")
    
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


def add_time_features(df, time_col, threshold):

    df = df.copy()
    
    # Basic transformations
    df['log_time'] = np.log1p(df[time_col])
    df['time_squared'] = df[time_col] ** 2
    df['time_cubic'] = df[time_col] ** 3
    df['time_sqrt'] = np.sqrt(df[time_col])
    df['time_reciprocal'] = 1 / (1 + df[time_col])
    
    # Normalized features
    df['time_normalized'] = df[time_col] / threshold
    df['time_progression'] = df[time_col] / (df[time_col] + threshold)
    df['time_remaining'] = (threshold - df[time_col]) / threshold
    df['time_ratio'] = df[time_col] / (df[time_col].max() + 1e-8)
    
    # Decay features
    df['time_decay_short'] = np.exp(-df[time_col] / 60)
    df['time_decay_long'] = np.exp(-df[time_col] / threshold)
    
    # Interaction features
    df['time_event_interaction'] = df[time_col] * df[config.EVENT_COL]
    df['time_remaining_event'] = df['time_remaining'] * df[config.EVENT_COL]
    
    # Binned features
    df['time_bins'] = pd.cut(df[time_col], bins=np.linspace(0, threshold, 13), labels=False)
    
    return df


def calculate_c_index(event, time, predictions, threshold):

    try:
        risk_scores = 1 - predictions
        y_structured = np.array(
            [(bool(e), t) for e, t in zip(event, time)],
            dtype=[('event', '?'), ('time', '<f8')]
        )
        cindex = concordance_index_censored(
            y_structured['event'], 
            y_structured['time'], 
            risk_scores
        )
        return cindex[0]
    except Exception as e:
        print(f"Error calculating C-index: {e}")
        return 0.5


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
        reason = (f"C-index degraded for {patience} consecutive iterations "
                 f"(best: {best_c_index:.4f} at iter {best_iteration}, "
                 f"recent avg: {avg_recent:.4f}, degradation: {degradation:.4f})")
        return True, best_iteration, reason
    
    return False, None, None


def track_deceased_convergence(df, iteration, threshold, subgroup='Overall'):

    paths = get_threshold_paths(threshold, subgroup)
    calibrated_col = paths['calibrated_col']
    expected_time_col = paths['expected_time_col']
    
    deceased_mask = (df[config.EVENT_COL] == 1)
    
    if deceased_mask.sum() > 0:
        avg_prob = df.loc[deceased_mask, calibrated_col].mean()
        max_prob = df.loc[deceased_mask, calibrated_col].max()
        time_error = np.mean(np.abs(
            df.loc[deceased_mask, expected_time_col] - 
            df.loc[deceased_mask, config.TIME_COL]
        ))
        
        print(f"Deceased patients - {subgroup} (Threshold: {threshold}m):")
        print(f"  Avg probability: {avg_prob:.4f}, Max: {max_prob:.4f}")
        print(f"  Avg time error: {time_error:.2f} months")


def create_predicted_survival_curve_with_overlap(df, time_col, calibrated_col, 
                                                  event_col, threshold):

    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    
    max_observed_time = df[time_col].max()
    curve_endpoint = max(threshold, max_observed_time)
    
    n_points_observed = 200
    n_points_extrapolate = 100
    
    time_points_observed = np.linspace(0, max_observed_time, n_points_observed)
    
    if curve_endpoint > max_observed_time:
        time_points_extrapolate = np.linspace(
            max_observed_time, curve_endpoint, n_points_extrapolate
        )[1:]
    else:
        time_points_extrapolate = np.array([])
    
    time_points = np.concatenate([time_points_observed, time_points_extrapolate])
    predicted_survival = np.zeros(len(time_points))
    
    # Use KM curve for observed range
    for idx, t in enumerate(time_points_observed):
        if t == 0:
            predicted_survival[idx] = 1.0
        else:
            predicted_survival[idx] = kmf.predict(t)
    
    # Extrapolate beyond observed range
    if len(time_points_extrapolate) > 0:
        surv_at_max_obs = kmf.predict(max_observed_time)
        
        censored_mask = df[event_col] == 0
        if censored_mask.sum() > 0:
            censored_probs = df.loc[censored_mask, calibrated_col].values
            censored_times = df.loc[censored_mask, time_col].values
            
            time_weights = np.exp(-(max_observed_time - censored_times) / 
                                 (max_observed_time / 3))
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
    
    # Ensure monotonicity
    for i in range(1, len(predicted_survival)):
        if predicted_survival[i] > predicted_survival[i-1]:
            predicted_survival[i] = predicted_survival[i-1]
    
    # Smooth extrapolated portion
    if len(time_points_extrapolate) > 0:
        from scipy.ndimage import gaussian_filter1d
        extrapolate_portion = predicted_survival[n_points_observed:]
        smoothed_extrapolate = gaussian_filter1d(extrapolate_portion, sigma=1.0)
        predicted_survival[n_points_observed:] = smoothed_extrapolate
    
    predicted_survival[0] = 1.0
    predicted_survival = np.clip(predicted_survival, 0, 1)
    
    return time_points, predicted_survival