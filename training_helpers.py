import os
import numpy as np
import tensorflow as tf
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_absolute_error

import config
from utils import calculate_c_index, should_stop_training
from calibration import (km_aware_calibration, time_aware_refinement,
                        enforce_constraints_v2, update_expected_survival_time)
from evaluation import (calculate_km_calibration_loss_v2,
                       plot_observed_vs_predicted_km_curves,
                       analyze_final_results)


def _process_predictions(df, iteration, threshold, subgroup,
                        all_test_preds, all_preds, y_test, event, time,
                        train_idx, test_idx, event_test, time_test,
                        target_col, calibrated_col, expected_time_col,
                        models, preprocessor, weights_dir, input_file,
                        convergence_history, best_r2, best_mae, best_c_index,
                        best_km_loss, best_c_index_iteration, stagnation_count,
                        best_df_state):

    
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
    
    print(f"\n  {'='*50}")
    print(f"  CALIBRATION PIPELINE (with C-index tracking):")
    print(f"  {'='*50}")
    
    # Step 1: Isotonic calibration
    iso = IsotonicRegression(out_of_bounds='clip', 
                             y_min=config.ISOTONIC_MIN, 
                             y_max=config.ISOTONIC_MAX)
    iso.fit(y_pred_full[train_idx], df[target_col].iloc[train_idx].values)
    y_full_calibrated = iso.predict(y_pred_full)
    
    c_index_after_iso = calculate_c_index(event.values, time.values, 
                                          y_full_calibrated, threshold)
    print(f"    After Isotonic:       C-index = {c_index_after_iso:.4f}")
    
    # Step 2: KM-aware calibration 
    y_full_calibrated = km_aware_calibration(df, y_full_calibrated, iteration,
                                             config.TIME_COL, config.EVENT_COL, 
                                             threshold)
    
    c_index_after_km = calculate_c_index(event.values, time.values, 
                                         y_full_calibrated, threshold)
    print(f"    After KM calibration: C-index = {c_index_after_km:.4f}")
    
    # Step 3: Time-aware refinement 
    y_full_calibrated = time_aware_refinement(df, y_full_calibrated, 
                                              event.values, time.values,
                                              threshold, iteration=iteration)
    
    c_index_after_time = calculate_c_index(event.values, time.values, 
                                           y_full_calibrated, threshold)
    print(f"    After Time-aware:     C-index = {c_index_after_time:.4f}")
    
    if c_index_after_time < c_index_raw - 0.05:
        print(f"\n     MAJOR C-INDEX DROP: {c_index_raw:.4f} and {c_index_after_time:.4f}")
        print(f"    Rolling back to lighter calibration")
        y_full_calibrated = 0.8 * y_pred_full + 0.2 * y_full_calibrated
        c_index_final = calculate_c_index(event.values, time.values, 
                                          y_full_calibrated, threshold)
        print(f"    After rollback: {c_index_final:.4f}")
    else:
        c_index_final = c_index_after_time
    
    print(f"  {'='*50}\n")
    
    df[target_col] = config.BLEND_FACTOR * y_pred_full + \
                     (1 - config.BLEND_FACTOR) * df[target_col]
    df[calibrated_col] = config.BLEND_FACTOR * y_full_calibrated + \
                         (1 - config.BLEND_FACTOR) * df[calibrated_col]
    

    df = update_expected_survival_time(df, df[calibrated_col].values, 
                                      event.values, time.values,
                                      expected_time_col, config.TIME_COL,
                                      config.EVENT_COL, threshold)

    df = enforce_constraints_v2(df, iteration, config.EVENT_COL, 
                               target_col, calibrated_col,
                               config.TIME_COL, expected_time_col)
    
    df.to_csv(input_file, index=False)
    

    km_loss, _, _, _ = calculate_km_calibration_loss_v2(
        df, config.TIME_COL, config.EVENT_COL, calibrated_col, threshold
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
    if r2_after > best_r2 + config.MIN_DELTA:
        best_r2 = r2_after
        improved = True
        print(f"     New best R2: {best_r2:.4f}")
    if mae_after < best_mae - config.MIN_DELTA:
        best_mae = mae_after
        improved = True
        print(f"     New best MAE: {best_mae:.4f}")
    if c_index_final > best_c_index + config.MIN_DELTA:
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
        print(f"     No improvement for {stagnation_count} iterations")
    
    # Check C-index based early stopping
    should_stop, best_iter, stop_reason = should_stop_training(
        convergence_history, patience=config.EARLY_STOPPING_PATIENCE
    )
    
    if should_stop:
        print(f"\n{'='*70}")
        print(f"   EARLY STOPPING: C-INDEX DEGRADATION DETECTED")
        print(f"{'='*70}")
        print(f"  {stop_reason}")
        print(f"  Reverting to best C-index iteration: {best_iter}")
        print(f"{'='*70}")
        
        # Revert to best DataFrame state
        if best_df_state is not None:
            df = best_df_state.copy()
            print(f"   Reverted DataFrame to iteration {best_c_index_iteration}")
        
        return None  # Signal to stop training
    
    # Check stopping conditions
    r2_target_met = r2_after >= config.TARGET_R2
    c_index_target_met = c_index_final >= config.TARGET_C_INDEX
    c_index_not_degraded = c_index_final >= (best_c_index - config.C_INDEX_TOLERANCE)
    km_loss_target_met = km_loss < 0.001
    
    print(f"\n  Progress toward targets:")
    print(f"    R2 = {config.TARGET_R2}:        {r2_after:.4f} "
          f"{ if r2_target_met else ''}")
    print(f"    C-index = {config.TARGET_C_INDEX}:   {c_index_final:.4f} "
          f"{ if c_index_target_met else ''}")
    print(f"    C-index stable:   {'' if c_index_not_degraded else ''} "
          f"(within {config.C_INDEX_TOLERANCE:.3f} of best)")
    print(f"    KM Loss < 0.001:  {km_loss:.6f} "
          f"{ if km_loss_target_met else ''}")
    print(f"    Best C-index so far: {best_c_index:.4f} "
          f"(iteration {best_c_index_iteration})")
    
    should_break = False
    
    if (r2_target_met and c_index_target_met and 
        c_index_not_degraded and km_loss_target_met):
        print(f"\n{'='*60}")
        print(f"   ALL TARGETS ACHIEVED ")
        print(f"{'='*60}")
        print(f"  R2:      {r2_after:.4f} (target: ={config.TARGET_R2})")
        print(f"  C-index: {c_index_final:.4f} (target: ={config.TARGET_C_INDEX})")
        print(f"  KM Loss: {km_loss:.6f} (target: <0.001)")
        print(f"{'='*60}")
        should_break = True
    
    if stagnation_count >= config.PATIENCE:
        print(f"\n   Stopping due to stagnation")
        print(f"   Current metrics: R2={r2_after:.4f}, C-index={c_index_final:.4f}, "
              f"KM Loss={km_loss:.6f}")
        
        if c_index_final < best_c_index - 0.02 and best_df_state is not None:
            print(f"   Reverting to best C-index iteration: {best_c_index_iteration}")
            df = best_df_state.copy()
        
        should_break = True
    
    return (df, convergence_history, best_r2, best_mae, best_c_index,
            best_km_loss, best_c_index_iteration, stagnation_count,
            best_df_state, should_break)


def _save_final_results(df, convergence_history, output_dir, weights_dir, 
                       input_file, threshold, subgroup, target_col, 
                       calibrated_col, best_c_index, best_c_index_iteration):

    print(f"\n{'='*70}")
    print(f"SAVING FINAL OPTIMIZED RESULTS")
    print(f"{'='*70}")
    print(f"  Best C-index: {best_c_index:.4f} "
          f"(achieved at iteration {best_c_index_iteration})")
    print(f"  Total iterations run: {len(convergence_history)}")
    
    df.to_csv(input_file, index=False)
    print(f"   Saved optimized predictions to: {input_file}")
    print(f"{'='*70}\n")
    
    if convergence_history:
        import pandas as pd
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
        time_col=config.TIME_COL,
        event_col=config.EVENT_COL,
        calibrated_col=calibrated_col,
        threshold=threshold,
        output_dir=output_dir,
        subgroup=subgroup
    )
    
    suffix = f"{threshold}m" if subgroup == 'Overall' else f"{threshold}m_{subgroup}"
    print(f"\n KM curves plot saved to: {output_dir}/km_curves_comparison_{suffix}.png")
    

    analyze_final_results(
        df=df,
        convergence_history=convergence_history,
        event_col=config.EVENT_COL,
        calibrated_col=calibrated_col,
        time_col=config.TIME_COL,
        threshold=threshold,
        subgroup=subgroup
    )
    
    print(f"\n Training complete for {subgroup} - threshold: {threshold} months!")