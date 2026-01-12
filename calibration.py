import numpy as np
from lifelines import KaplanMeierFitter
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator
import config
from utils import calculate_c_index


def ranking_preserving_calibration(y_pred, y_true, event, time):

    original_ranking = np.argsort(-y_pred)
    
    iso = IsotonicRegression(
        out_of_bounds='clip', 
        y_min=config.ISOTONIC_MIN, 
        y_max=config.ISOTONIC_MAX
    )
    y_calibrated = iso.fit_transform(y_pred, y_true)
    
    new_ranking = np.argsort(-y_calibrated)
    ranking_changed = not np.array_equal(original_ranking, new_ranking)
    
    if ranking_changed:
        sorted_idx = np.argsort(y_pred)
        y_pred_sorted = y_pred[sorted_idx]
        y_true_sorted = y_true[sorted_idx]
        
        calibrator = PchipInterpolator(y_pred_sorted, y_true_sorted)
        y_calibrated = calibrator(y_pred)
        y_calibrated = np.clip(y_calibrated, config.ISOTONIC_MIN, config.ISOTONIC_MAX)
    
    return y_calibrated


def km_aware_calibration(df, y_pred, iteration, time_col, event_col, threshold):

    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    
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
    
    calibrated = np.clip(calibrated, config.ISOTONIC_MIN, config.ISOTONIC_MAX)
    

    c_index_after = calculate_c_index(df[event_col], df[time_col], calibrated, threshold)
    
    if c_index_after < c_index_before - 0.02:
        print(f"     KM calibration reduced C-index: "
              f"{c_index_before:.4f} and {c_index_after:.4f}")
        print(f"     Reducing calibration strength")

        calibrated = 0.7 * y_pred + 0.3 * calibrated
        c_index_final = calculate_c_index(df[event_col], df[time_col], calibrated, threshold)
        print(f"     After rollback: {c_index_final:.4f}")
    
    return calibrated


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
        
        max_deceased_prob = max(
            0.05, 
            config.MAX_DECEASED_PROB_BASE - config.MAX_DECEASED_PROB_DECAY * iteration
        )
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
                refined[idx] = (blend_factor * km_cond_prob + 
                              (1 - blend_factor) * refined[idx])
        
        very_long_term = censored_mask & (time >= threshold * 0.9)
        if very_long_term.sum() > 0:
            refined[very_long_term] = np.maximum(refined[very_long_term], 0.8)
    
    refined = np.clip(refined, config.ISOTONIC_MIN, config.ISOTONIC_MAX)
    
    c_index_after = calculate_c_index(event, time, refined, threshold)
    
    if c_index_after < c_index_before - 0.03:
        print(f"     Time-aware refinement reduced C-index: "
              f"{c_index_before:.4f} and {c_index_after:.4f}")
        print(f"     Using lighter blend")
        refined = 0.6 * pred_probs + 0.4 * refined
    
    return refined


def time_dependent_calibration(y_true, y_pred, event, time, time_bins=15):

    calibrated = y_pred.copy()
    time_quantiles = np.unique(
        np.percentile(time, np.linspace(0, 100, time_bins + 1))
    )
    
    for i in range(len(time_quantiles) - 1):
        time_mask = ((time >= time_quantiles[i]) & 
                    (time < time_quantiles[i + 1]))
        
        if np.sum(time_mask) > 20:
            iso = IsotonicRegression(
                out_of_bounds='clip', 
                y_min=config.ISOTONIC_MIN, 
                y_max=config.ISOTONIC_MAX
            )
            calibrated_local = iso.fit_transform(
                y_pred[time_mask], 
                y_true[time_mask]
            )
            
            sample_ratio = np.sum(time_mask) / len(time)
            blend_factor = min(0.8, 0.3 + 0.5 * sample_ratio)
            
            calibrated[time_mask] = (blend_factor * calibrated_local + 
                                   (1 - blend_factor) * y_pred[time_mask])
    
    return calibrated


def enforce_constraints_v2(df, iteration, event_col, target_col, 
                          calibrated_col, time_col, expected_time_col):

    df = df.copy()
    deceased_mask = (df[event_col] == 1)
    censored_mask = (df[event_col] == 0)
    
    # Deceased patients
    if deceased_mask.sum() > 0:
        df.loc[deceased_mask, expected_time_col] = df.loc[deceased_mask, time_col]
        
        max_deceased_prob = max(
            0.05, 
            config.MAX_DECEASED_PROB_BASE - config.MAX_DECEASED_PROB_DECAY * min(iteration, 10)
        )
        df.loc[deceased_mask, calibrated_col] = np.minimum(
            df.loc[deceased_mask, calibrated_col], 
            max_deceased_prob
        )
        df.loc[deceased_mask, target_col] = df.loc[deceased_mask, calibrated_col]
    
    # Censored patients
    if censored_mask.sum() > 0:
        df.loc[censored_mask, expected_time_col] = np.maximum(
            df.loc[censored_mask, expected_time_col],
            df.loc[censored_mask, time_col]
        )
        
        df.loc[censored_mask, calibrated_col] = np.maximum(
            df.loc[censored_mask, calibrated_col],
            config.MIN_CENSORED_PROB
        )
    
    df[calibrated_col] = np.clip(
        df[calibrated_col], 
        config.ISOTONIC_MIN, 
        config.ISOTONIC_MAX
    )
    df[target_col] = np.clip(
        df[target_col], 
        config.ISOTONIC_MIN, 
        config.ISOTONIC_MAX
    )
    df[expected_time_col] = np.maximum(df[expected_time_col], df[time_col])
    
    return df


def update_expected_survival_time(df, pred_probs, event, time, 
                                  expected_time_col, time_col, 
                                  event_col, threshold):

    df = df.copy()
    deceased_mask = (event == 1)
    censored_mask = (event == 0)
    
    # Deceased patients
    if deceased_mask.sum() > 0:
        df.loc[deceased_mask, expected_time_col] = df.loc[deceased_mask, time_col]
    
    # Censored patients
    if censored_mask.sum() > 0:
        probs_censored = pred_probs[censored_mask]
        time_censored = time[censored_mask]
        beyond_threshold_mask = time_censored >= threshold
        within_threshold_mask = time_censored < threshold
        
        if beyond_threshold_mask.any():
            df.loc[censored_mask & (time >= threshold), expected_time_col] = \
                time[censored_mask & (time >= threshold)]
        
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
                    time_within[high_prob_mask] + remaining_time[high_prob_mask] * 0.9
                )
            
            df.loc[censored_mask & (time < threshold), expected_time_col] = \
                expected_time_within
    
    df[expected_time_col] = np.maximum(df[expected_time_col], df[time_col])
    
    return df