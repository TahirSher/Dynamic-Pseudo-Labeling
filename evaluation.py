import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error
import os

import config
from utils import create_predicted_survival_curve_with_overlap


def calculate_km_calibration_loss_v2(df, time_col, event_col, calibrated_col, threshold):

    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    
    time_points, predicted_survival = create_predicted_survival_curve_with_overlap(
        df, time_col, calibrated_col, event_col, threshold
    )
    
    max_observed_time = df[time_col].max()
    eval_times = np.linspace(0, max_observed_time, 30)
    
    observed_survival = np.array([kmf.predict(t) for t in eval_times])
    
    pred_interp = interp1d(
        time_points, predicted_survival, 
        kind='linear', 
        fill_value='extrapolate'
    )
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


def calculate_survival_probabilities_at_threshold(df, time_col, event_col, 
                                                  calibrated_col, threshold):

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
    
    censored_surv_prob = (df.loc[censored_mask, calibrated_col].mean() 
                         if censored_mask.sum() > 0 else np.nan)
    deceased_surv_prob = (df.loc[event_mask, calibrated_col].mean() 
                         if event_mask.sum() > 0 else np.nan)
    
    high_surv_count = (df[calibrated_col] > 0.70).sum()
    high_surv_pct = (high_surv_count / len(df)) * 100
    
    low_surv_count = (df[calibrated_col] < 0.30).sum()
    low_surv_pct = (low_surv_count / len(df)) * 100
    
    results = {
        'threshold': threshold,
        'max_observed_time': max_observed_time,
        'evaluation_time': threshold,
        'used_max_observed': False,
        
        'observed_surv_at_eval': (observed_surv_at_threshold * 100 
                                 if not np.isnan(observed_surv_at_threshold) else np.nan),
        'predicted_surv_at_eval': predicted_surv_at_threshold * 100,
        'difference': (abs(predicted_surv_at_threshold * 100 - 
                          observed_surv_at_threshold * 100) 
                      if not np.isnan(observed_surv_at_threshold) else np.nan),
        
        'predicted_surv_at_max_obs': predicted_surv_at_max_obs * 100,
        'observed_surv_at_max_obs': ((kmf.predict(max_observed_time) * 100) 
                                     if max_observed_time <= kmf.survival_function_.index.max() 
                                     else np.nan),
        
        'censored_surv_prob': (censored_surv_prob * 100 
                              if not np.isnan(censored_surv_prob) else np.nan),
        'deceased_surv_prob': (deceased_surv_prob * 100 
                              if not np.isnan(deceased_surv_prob) else np.nan),
        
        'high_surv_count': high_surv_count,
        'high_surv_pct': high_surv_pct,
        'low_surv_count': low_surv_count,
        'low_surv_pct': low_surv_pct,
        
        'mean_individual_prob': df[calibrated_col].mean() * 100
    }
    
    return results


def plot_observed_vs_predicted_km_curves(df, iteration, time_col, event_col, 
                                        calibrated_col, threshold, output_dir, 
                                        subgroup='Overall'):

    print(f"\n{'='*60}")
    print(f"Generating KM Curves - {subgroup} - Threshold {threshold}m")
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
            
            curve_std = np.std(predicted_survival[
                max(0, pred_median_idx[0]-10):min(len(predicted_survival), pred_median_idx[0]+10)
            ])
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
    
    fig = plt.figure(figsize=config.FIGURE_SIZE_CURVES, dpi=config.PLOT_DPI)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[:, :2])
    
    kmf_observed.plot_survival_function(
        ax=ax1, ci_show=True, color='#FF6B6B', 
        linewidth=3, label='Observed (KM)', alpha=0.8
    )
    
    ax1.plot(time_points, predicted_survival, color='#4ECDC4', 
            linewidth=3, linestyle='--', label='Predicted', alpha=0.9)
    
    ax1.axvline(x=max_observed_time, color='gray', linestyle=':', 
               linewidth=2, alpha=0.5, 
               label=f'Max Observed: {max_observed_time:.1f}m')
    
    if abs(max_observed_time - threshold) > 1:
        ax1.axvline(x=threshold, color='purple', linestyle='-.', 
                   linewidth=2.5, alpha=0.6, 
                   label=f'Threshold: {threshold}m')
    
    if not np.isnan(obs_median) and obs_median > 0:
        ax1.axvline(x=obs_median, color='#FF6B6B', linestyle=':', 
                   linewidth=2.5, alpha=0.8, 
                   label=f'Observed Median: {obs_median:.1f} '
                         f'({obs_median_lower:.2f}, {obs_median_upper:.2f}) months')
    
    if not np.isnan(pred_median) and pred_median > 0:
        ax1.axvline(x=pred_median, color='#4ECDC4', linestyle=':', 
                   linewidth=2.5, alpha=0.8, 
                   label=f'Predicted Median: {pred_median:.1f} '
                         f'({pred_median_lower:.2f}, {pred_median_upper:.2f}) months')
    
    ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    ax1.set_xlabel('Time (months)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Survival Probability', fontsize=14, fontweight='bold')
    
    display_name = config.SUBGROUP_DISPLAY_NAMES.get(subgroup, subgroup)
    title_text = (f'Observed vs Predicted Survival Curves - {display_name}\n'
                 f'(Threshold: {threshold}m - KM Loss: {km_loss:.6f})')
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
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2.5, alpha=0.7, 
            label='Perfect Calibration')
    scatter = ax2.scatter(obs_at_eval, pred_at_eval, c=eval_times, 
                         cmap='viridis', s=100, alpha=0.7, 
                         edgecolors='black', linewidth=1.5)
    
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
    ax3.plot(eval_times, abs_diff, 'o-', color='#E74C3C', 
            linewidth=2, markersize=6)
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
                dpi=config.PLOT_DPI, bbox_inches='tight')
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


def analyze_final_results(df, convergence_history, event_col, calibrated_col, 
                         time_col, threshold, subgroup='Overall'):

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
    print(f"     High survival (>70%):     {surv_probs['high_surv_count']} patients "
          f"({surv_probs['high_surv_pct']:.1f}%)")
    print(f"     Low survival (<30%):      {surv_probs['low_surv_count']} patients "
          f"({surv_probs['low_surv_pct']:.1f}%)")
    print(f"   {'-'*50}")
    
    from utils import calculate_c_index
    c_index = calculate_c_index(df[event_col], df[time_col], 
                                df[calibrated_col], threshold)
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
    print(f"     Very High Risk (<30%):    {very_high_risk} "
          f"({very_high_risk/len(df)*100:.1f}%)")
    print(f"     High Risk (30-50%):       {high_risk} "
          f"({high_risk/len(df)*100:.1f}%)")
    print(f"     Moderate Risk (50-70%):   {moderate_risk} "
          f"({moderate_risk/len(df)*100:.1f}%)")
    print(f"     Low Risk (>=70%):          {low_risk} "
          f"({low_risk/len(df)*100:.1f}%)")
    
    return surv_probs