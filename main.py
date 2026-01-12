import os
import sys
import warnings
import pandas as pd
import traceback

warnings.filterwarnings('ignore')

import config
from training import train_for_threshold
from results import (generate_all_generalized_feature_importance, 
                    generate_subgroup_summary)


def setup_directories():

    os.makedirs(config.BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.BASE_WEIGHTS_DIR, exist_ok=True)
    print(f" Created output directories:")
    print(f"  - {config.BASE_OUTPUT_DIR}")
    print(f"  - {config.BASE_WEIGHTS_DIR}")


def validate_dataset(dataset_path):

    if not os.path.exists(dataset_path):
        print(f"\n ERROR: Dataset file not found: {dataset_path}")
        print(f"Please ensure the file exists and update the path in main.py")
        return None
    
    try:
        df = pd.read_csv(dataset_path)
        print(f" Loaded dataset: {len(df)} samples")
        
        required_cols = [config.TIME_COL, config.EVENT_COL]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"\n ERROR: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        print(f" Required columns present: {required_cols}")
        
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  Events: {df[config.EVENT_COL].sum()} "
              f"({df[config.EVENT_COL].sum()/len(df)*100:.1f}%)")
        print(f"  Censored: {(df[config.EVENT_COL] == 0).sum()} "
              f"({(df[config.EVENT_COL] == 0).sum()/len(df)*100:.1f}%)")
        print(f"  Mean survival time: {df[config.TIME_COL].mean():.2f} months")
        print(f"  Median survival time: {df[config.TIME_COL].median():.2f} months")
        
        return df
        
    except Exception as e:
        print(f"\n ERROR: Failed to load dataset: {e}")
        return None


def main(dataset_path):
    
    print("="*80)
    print("MULTI-THRESHOLD + SUBGROUP SURVIVAL ANALYSIS")
    print("="*80)
    print(f"Thresholds: {config.THRESHOLDS} months")
    print(f"Subgroups: {list(config.SUBGROUPS.keys())}")
    print(f"Target C-index: {config.TARGET_C_INDEX:.2f} "
          f"(realistic for survival analysis)")
    print("="*80 + "\n")
    
    setup_directories()
    
    base_df = validate_dataset(dataset_path)
    if base_df is None:
        print("\n Exiting due to dataset validation failure")
        sys.exit(1)
    
    all_results = {}
    
    # ========================================================================
    # TRAIN ALL THRESHOLDS AND SUBGROUPS
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING PHASE")
    print("="*80)
    
    for threshold in config.THRESHOLDS:
        for subgroup in config.SUBGROUPS.keys():
            print(f"\n{'='*80}")
            print(f"Processing: Threshold={threshold}m, Subgroup={subgroup}")
            print(f"{'='*80}")
            
            try:
                result_df = train_for_threshold(threshold, base_df, subgroup)
                all_results[(threshold, subgroup)] = result_df
                
                if result_df is not None:
                    print(f"\n Successfully completed: "
                          f"Threshold={threshold}m, Subgroup={subgroup}")
                else:
                    print(f"\n Training returned None: "
                          f"Threshold={threshold}m, Subgroup={subgroup}")
                
            except Exception as e:
                print(f"\n ERROR: Threshold={threshold}m, Subgroup={subgroup}")
                print(f"Error message: {str(e)}")
                print(f"\nFull traceback:")
                traceback.print_exc()
                all_results[(threshold, subgroup)] = None
    
    # ========================================================================
    # GENERATE FEATURE IMPORTANCE PLOTS
    # ========================================================================
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    try:
        feature_importance_results = generate_all_generalized_feature_importance(all_results)
        print("\n Feature importance plots generated for all subgroups")
    except Exception as e:
        print(f"\n ERROR generating feature importance: {e}")
        traceback.print_exc()
    
    # ========================================================================
    # GENERATE COMPREHENSIVE SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    try:
        generate_subgroup_summary(all_results)
        print("\n Summary report generated")
    except Exception as e:
        print(f"\n ERROR generating summary: {e}")
        traceback.print_exc()
    
    # ========================================================================
    # FINAL OUTPUT
    # ========================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n Outputs saved to:")
    print(f"   Main directory: {config.BASE_OUTPUT_DIR}")
    print(f"   Model weights: {config.BASE_WEIGHTS_DIR}")
    print(f"   Summary: {os.path.join(config.BASE_OUTPUT_DIR, 'subgroup_summary.txt')}")
    
    print(f"\n Generated files per threshold:")
    print(f"   KM comparison plots: km_curves_comparison_*.png")
    print(f"   Convergence history: convergence_history_*.csv")
    print(f"   Predictions: optimized_survival_probabilities_*.csv")
    print(f"   Iteration 0 (baseline): iteration_0_predictions_*.csv")
    
    print(f"\n Generated files overall:")
    print(f"  Feature importance plots: feature_importance_generalized_*.png")
    print(f"  Feature importance data: feature_importance_generalized_*.csv")
    
    # Count successful trainings
    successful = sum(1 for v in all_results.values() if v is not None)
    total = len(all_results)
    
    print(f"\n Training Summary:")
    print(f"  Successfully completed: {successful}/{total}")
    
    if successful < total:
        print(f"    {total - successful} training(s) failed - check logs above")
    else:
        print(f"  All trainings completed successfully!")
    
    print("\n" + "="*80)



if __name__ == "__main__":

    # =======================================================================
    # Dataset Path
    # =======================================================================
    DATASET_PATH = "final_OS_dataset_01_final.csv"
    
    # Command-line argument
    if len(sys.argv) > 1:
        DATASET_PATH = sys.argv[1]
    
    print(f"\nUsing dataset: {DATASET_PATH}\n")
    
    # Main analysis

    main(DATASET_PATH)
