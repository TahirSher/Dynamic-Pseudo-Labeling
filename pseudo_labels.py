import numpy as np
from lifelines import KaplanMeierFitter
from sksurv.metrics import concordance_index_censored
import config


def create_generalized_pseudo_labels_corrected(df, time_col, event_col, target_col, 
                                               expected_time_col, threshold):

    print(f"\n{'='*70}")
    print(f"CREATING INITIAL PSEUDO-LABELS (LEAVE-ONE-OUT METHOD)")
    print(f"  Threshold: {threshold} months")
    print(f"{'='*70}")
    
    n = len(df)
    
    # ========================================================================
    # STEP 1: KM on FULL dataset
    # ========================================================================
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
    print(f"    Censored: {n - df[event_col].sum()} "
          f"({(n - df[event_col].sum())/n*100:.1f}%)")
    print(f"    Max observed time: {max_observed_time:.2f} months")
    print(f"    Threshold: {threshold} months")
    print(f"    Overall KM survival at threshold: {S_all:.4f}")
    
    # ========================================================================
    # STEP 2: Leave-One-Out Pseudo-Observation Computation
    # ========================================================================
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
    
    # ========================================================================
    # STEP 3: Assign to DataFrame
    # ========================================================================
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
    
    # ========================================================================
    # STEP 4: Statistics and Validation
    # ========================================================================
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
            print(f"       WARNING: Very low separation")
        elif separation > 0.7:
            print(f"       WARNING: Very high separation (potential leakage)")
        else:
            print(f"       Separation in reasonable range")
    
    # ========================================================================
    # STEP 5: Initial C-index 
    # ========================================================================
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
        print(f"\n      Could not calculate initial C-index: {e}")
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
    
    # ========================================================================
    # CHECK 1: Distribution Statistics
    # ========================================================================
    mean_prob = pseudo_labels.mean()
    std_prob = pseudo_labels.std()
    
    print(f"\n  1. DISTRIBUTION CHECK:")
    print(f"     Mean: {mean_prob:.4f}")
    print(f"     Std:  {std_prob:.4f}")
    
    if mean_prob < 0.05:
        print(f"      ? Mean too low (< 0.05)")
        validation_passed = False
    elif mean_prob > 0.95:
        print(f"       Mean too high (> 0.95)")
        validation_passed = False
    else:
        print(f"       Mean in reasonable range")
    
    if std_prob < 0.01:
        print(f"       Very low variance - pseudo-labels may be uninformative")
    
    # ========================================================================
    # CHECK 2: Group Separation
    # ========================================================================
    if deceased_mask.sum() > 0 and censored_mask.sum() > 0:
        deceased_mean = df.loc[deceased_mask, target_col].mean()
        censored_mean = df.loc[censored_mask, target_col].mean()
        separation = abs(censored_mean - deceased_mean)
        
        print(f"\n  2. GROUP SEPARATION CHECK:")
        print(f"     Deceased mean: {deceased_mean:.4f}")
        print(f"     Censored mean: {censored_mean:.4f}")
        print(f"     Separation:    {separation:.4f}")
        
        if separation > 0.75:
            print(f"       CRITICAL: Separation too large - LIKELY DATA LEAKAGE")
            validation_passed = False
        elif separation > 0.60:
            print(f"       Separation high - possible leakage")
        elif separation < 0.05:
            print(f"       Very low separation - pseudo-labels may be uninformative")
        else:
            print(f"       Separation in reasonable range (0.05-0.60)")
    
    # ========================================================================
    # CHECK 3: C-index
    # ========================================================================
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
            print(f"       Expected range: 0.55-0.75 for initial pseudo-observations")
            validation_passed = False
        elif c_index > 0.85:
            print(f"       WARNING: C-index > 0.85 - suspiciously high")
            print(f"       Expected range: 0.55-0.75")
        elif c_index < 0.50:
            print(f"       WARNING: C-index < 0.50 - below random performance")
            validation_passed = False
        elif 0.55 <= c_index <= 0.80:
            print(f"       EXCELLENT: C-index in expected/good range for initial pseudo-obs")
        else:
            print(f"       C-index outside typical range but may be acceptable")
            
    except Exception as e:
        print(f"\n  3. C-INDEX CHECK: Failed to compute ({e})")
        validation_passed = False
    
    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    print(f"\n  {'='*60}")
    if validation_passed:
        print(f"   VALIDATION PASSED - Pseudo-labels appear valid")
    else:
        print(f"   VALIDATION FAILED - Critical issues detected")
        print(f"     DO NOT PROCEED - Fix pseudo-label generation first")
    print(f"  {'='*60}\n")
    
    return validation_passed