import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import tensorflow as tf

import config
from utils import get_threshold_paths, calculate_c_index
from lifelines import KaplanMeierFitter


def calculate_threshold_specific_importance(models, preprocessor, X, y):

    X_processed = preprocessor.transform(X)
    
    baseline_preds = np.mean([model.predict(X_processed, verbose=0).flatten() 
                             for model in models], axis=0)
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
            
            permuted_preds = np.mean([model.predict(X_permuted, verbose=0).flatten() 
                                     for model in models], axis=0)
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
    
    all_threshold_importances = []
    valid_thresholds = []
    
    for threshold in config.THRESHOLDS:
        key = (threshold, subgroup)
        if key not in all_results or all_results[key] is None:
            print(f"  Skipping threshold {threshold}m - no results available")
            continue
        
        paths = get_threshold_paths(threshold, subgroup)
        weights_dir = paths['weights_dir']
        calibrated_col = paths['calibrated_col']
        
        try:
            
            models = []
            for i in range(config.ENSEMBLE_SIZE):
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
            threshold_importance = calculate_threshold_specific_importance(
                models, preprocessor, X, y
            )
            
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
        importances = [imp_dict[feature] for imp_dict in all_threshold_importances 
                      if feature in imp_dict]
        if importances:
            aggregated_importance[feature] = np.mean(importances)
    

    clinical_importance = {}
    engineered_importance = {}
    
    for feat, importance in aggregated_importance.items():
        if any(kw in feat.lower() for kw in config.TIME_FEATURE_KEYWORDS):
            engineered_importance[feat] = importance
        else:
            clinical_importance[feat] = importance
    
    sorted_clinical = sorted(clinical_importance.items(), key=lambda x: x[1], reverse=True)
    sorted_engineered = sorted(engineered_importance.items(), key=lambda x: x[1], reverse=True)
    

    _plot_feature_importance(sorted_clinical, sorted_engineered, 
                            valid_thresholds, output_dir, subgroup)
    

    _save_feature_importance_csv(sorted_clinical, sorted_engineered,
                                valid_thresholds, output_dir, subgroup)
    

    print(f"\n  Top 10 Clinical Features:")
    for i, (feat, imp) in enumerate(sorted_clinical[:10], 1):
        print(f"    {i:2d}. {feat:30s}: {imp:.6f}")
    
    print(f"\n   Saved: feature_importance_generalized_{subgroup}.png/csv")
    
    return {
        'clinical': clinical_importance, 
        'engineered': engineered_importance, 
        'thresholds': valid_thresholds
    }


def _plot_feature_importance(sorted_clinical, sorted_engineered, 
                             valid_thresholds, output_dir, subgroup):
   
    display_name = config.SUBGROUP_DISPLAY_NAMES.get(subgroup, subgroup)
    thresholds_str = ', '.join([f'{t}m' for t in valid_thresholds])
    
    fig = plt.figure(figsize=config.FIGURE_SIZE_IMPORTANCE, dpi=300)
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    # Clinical Features
    ax1 = fig.add_subplot(gs[0, 0])
    if sorted_clinical:
        names = [f[0] for f in sorted_clinical]
        vals = [f[1] for f in sorted_clinical]
        total = sum(vals) if sum(vals) > 0 else 1
        pcts = [(v / total) * 100 for v in vals]
        
        x_pos = np.arange(len(names))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
        
        bars = ax1.bar(x_pos, pcts, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(names, rotation=45, ha='right', 
                           fontsize=11, fontweight='bold')
        ax1.set_xlabel('Clinical Features', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Relative Importance (%)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Clinical Features Importance - {display_name}',
                     fontsize=14, fontweight='bold', pad=20)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, pct in zip(bars, pcts):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{pct:.1f}%', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x_pos, np.cumsum(pcts), 'ro-', linewidth=2.5, 
                     markersize=8, label='Cumulative %', alpha=0.7)
        ax1_twin.set_ylabel('Cumulative (%)', fontsize=12, fontweight='bold')
        ax1_twin.set_ylim(0, 105)
        ax1_twin.legend(loc='lower right')
        ax1_twin.axhline(y=80, color='orange', linestyle='--', 
                        linewidth=2, alpha=0.5)
    
    # Engineered Features
    ax2 = fig.add_subplot(gs[1, 0])
    if sorted_engineered:
        names = [f[0] for f in sorted_engineered]
        vals = [f[1] for f in sorted_engineered]
        total = sum(vals) if sum(vals) > 0 else 1
        pcts = [(v / total) * 100 for v in vals]
        
        x_pos = np.arange(len(names))
        colors = plt.cm.Blues(np.linspace(0.2, 0.8, len(names)))
        
        bars = ax2.bar(x_pos, pcts, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(names, rotation=45, ha='right', 
                           fontsize=11, fontweight='bold')
        ax2.set_xlabel('Engineered Time Features', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Relative Importance (%)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Engineered Features Importance - {display_name}',
                     fontsize=14, fontweight='bold', pad=20)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, pct in zip(bars, pcts):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{pct:.1f}%', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x_pos, np.cumsum(pcts), 'bo-', linewidth=2.5, 
                     markersize=8, label='Cumulative %', alpha=0.7)
        ax2_twin.set_ylabel('Cumulative (%)', fontsize=12, fontweight='bold')
        ax2_twin.set_ylim(0, 105)
        ax2_twin.legend(loc='lower right')
        ax2_twin.axhline(y=80, color='orange', linestyle='--', 
                        linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'feature_importance_generalized_{subgroup}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def _save_feature_importance_csv(sorted_clinical, sorted_engineered,
                                 valid_thresholds, output_dir, subgroup):
    
    thresholds_str = ', '.join([f'{t}m' for t in valid_thresholds])
    
    clinical_df = pd.DataFrame(sorted_clinical, columns=['Feature', 'Importance'])
    clinical_df['Category'] = 'Clinical'
    engineered_df = pd.DataFrame(sorted_engineered, columns=['Feature', 'Importance'])
    engineered_df['Category'] = 'Engineered'
    
    combined_df = pd.concat([clinical_df, engineered_df]).sort_values(
        'Importance', ascending=False
    )
    combined_df['Subgroup'] = subgroup
    combined_df['Thresholds'] = thresholds_str
    combined_df.to_csv(
        os.path.join(output_dir, f'feature_importance_generalized_{subgroup}.csv'),
        index=False
    )


def generate_all_generalized_feature_importance(all_results):

    print("\n" + "="*80)
    print("GENERATING GENERALIZED FEATURE IMPORTANCE")
    print("="*80)
    
    results = {}
    for subgroup in config.SUBGROUPS.keys():
        results[subgroup] = calculate_generalized_feature_importance(
            all_results, config.BASE_OUTPUT_DIR, subgroup
        )
    
    return results


def generate_subgroup_summary(all_results):

    print("\n" + "="*80)
    print("COMPREHENSIVE SUBGROUP ANALYSIS SUMMARY")
    print("="*80)
    
    summary_file = os.path.join(config.BASE_OUTPUT_DIR, "subgroup_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-THRESHOLD SURVIVAL ANALYSIS - SUBGROUP SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for threshold in config.THRESHOLDS:
            f.write(f"\n{'='*80}\n")
            f.write(f"THRESHOLD: {threshold} MONTHS\n")
            f.write(f"{'='*80}\n\n")
            
            for subgroup in config.SUBGROUPS.keys():
                key = (threshold, subgroup)
                if key in all_results and all_results[key] is not None:
                    df = all_results[key]
                    paths = get_threshold_paths(threshold, subgroup)
                    calibrated_col = paths['calibrated_col']
                    
                    _write_subgroup_stats(f, df, threshold, subgroup, calibrated_col)
                else:
                    f.write(f"\nSubgroup: {subgroup}\n")
                    f.write(f"{'-'*40}\n")
                    f.write(f"Training failed or skipped\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("="*80 + "\n")
    
    print(f"\n Comprehensive summary saved to: {summary_file}")
    
    # Print to console
    with open(summary_file, 'r') as f:
        print(f.read())


def _write_subgroup_stats(f, df, threshold, subgroup, calibrated_col):
    """Write subgroup statistics to file."""
    from evaluation import calculate_survival_probabilities_at_threshold
    
    event_mask = df[config.EVENT_COL] == 1
    
    f.write(f"\nSubgroup: {subgroup}\n")
    f.write(f"{'-'*40}\n")
    f.write(f"Sample Size: {len(df)}\n")
    f.write(f"Events: {event_mask.sum()} ({event_mask.sum()/len(df)*100:.1f}%)\n")
    f.write(f"Censored: {(~event_mask).sum()} ({(~event_mask).sum()/len(df)*100:.1f}%)\n")
    
    surv_probs = calculate_survival_probabilities_at_threshold(
        df, config.TIME_COL, config.EVENT_COL, calibrated_col, threshold
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
    f.write(f"  High survival (>70%): {surv_probs['high_surv_count']} "
            f"({surv_probs['high_surv_pct']:.1f}%)\n")
    f.write(f"  Low survival (<30%):  {surv_probs['low_surv_count']} "
            f"({surv_probs['low_surv_pct']:.1f}%)\n")
    
    f.write(f"\n Other Metrics:\n")
    f.write(f"  Mean Survival Probability: {df[calibrated_col].mean():.3f}\n")
    f.write(f"  Median Survival Probability: {df[calibrated_col].median():.3f}\n")
    
    c_index = calculate_c_index(df[config.EVENT_COL], df[config.TIME_COL],
                                df[calibrated_col], threshold)
    f.write(f"  C-index: {c_index:.4f}\n")
    
    kmf = KaplanMeierFitter()
    kmf.fit(df[config.TIME_COL], df[config.EVENT_COL])
    median_surv = kmf.median_survival_time_
    if not np.isnan(median_surv):
        f.write(f"  Median Survival Time: {median_surv:.2f} months\n")
    else:
        f.write(f"  Median Survival Time: Not reached\n")
    
    paths = get_threshold_paths(threshold, subgroup)
    f.write(f"Output Directory: {paths['output_dir']}\n")
    f.write(f"Weights Directory: {paths['weights_dir']}\n")
    f.write("\n")