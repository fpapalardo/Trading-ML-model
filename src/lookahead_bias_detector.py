import pandas as pd
import numpy as np
import inspect
from typing import Callable, List, Dict, Any, Union
import pandas_ta as ta
import sys
import os

# --- FIX: Add project root to Python path ---
# This allows the script to find and import your feature modules correctly.
# It assumes your project structure is something like:
# project_root/
#  ├── features/
#  │   ├── registry.py
#  │   └── ...
#  └── lookahead_bias_detector.py (this file)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


# --- The Core Checker Function (Improved Reporting) ---
def check_for_lookahead(
    indicator_function: Callable[[pd.DataFrame], pd.DataFrame],
    base_df: pd.DataFrame,
    func_name: str, # Added for better error reporting
    input_cols: List[str] = ['open', 'high', 'low', 'close', 'volume']
) -> Union[bool, str]:
    """
    Detects look-ahead bias in an indicator function by perturbing future data.
    Returns:
        True: If the check passes (no bias).
        False: If the check fails (bias detected).
        "SETUP_ERROR": If the function fails to run at all (e.g., dependency missing).
    """
    print(f"🔬 Checking: {func_name}...")

    df_original = base_df.copy(deep=True)
    try:
        # Pass a copy to prevent the function from modifying the original df in the loop
        features_original = indicator_function(df_original.copy())
        original_cols = features_original.columns.difference(base_df.columns)
        if original_cols.empty:
            # This is not an error, some functions might be conditional or have no effect on this data
            print("   - ⚠️  Function did not add any new columns in this run. Skipping.")
            return True
        features_original = features_original[original_cols]
    except Exception as e:
        print(f"   - ❌ ERROR: Function failed on initial run: {e}")
        return "SETUP_ERROR"

    for col_to_perturb in input_cols:
        if col_to_perturb not in base_df.columns:
            continue

        df_perturbed = base_df.copy(deep=True)
        original_value = df_perturbed.iloc[-1][col_to_perturb]
        perturbed_value = original_value * 1.1 + 0.1
        df_perturbed.iloc[-1, df_perturbed.columns.get_loc(col_to_perturb)] = perturbed_value

        try:
            # Pass a copy here as well
            features_perturbed = indicator_function(df_perturbed.copy())
            features_perturbed = features_perturbed[original_cols]
        except Exception as e:
            print(f"   - ❌ ERROR: Function failed on perturbed run for '{col_to_perturb}': {e}")
            return "SETUP_ERROR"

        # Compare all rows EXCEPT the last one. A change in the last row is expected.
        comparison_df = features_original.iloc[:-1].compare(features_perturbed.iloc[:-1])

        if not comparison_df.empty:
            leaky_timestamp = comparison_df.index.max()
            affected_columns = comparison_df.columns.get_level_values(0).unique().tolist()
            print("   - 🚨 FAILED: LOOK-AHEAD BIAS DETECTED!")
            print(f"     - Perturbing '{col_to_perturb}' in the future caused a change in the past.")
            print(f"     - Leaky Feature(s): {affected_columns}")
            print(f"     - Most recent leak at: {leaky_timestamp}")
            return False

    print("   - ✅ PASSED")
    return True


if __name__ == '__main__':
    # --- Create a sample DataFrame ---
    data = {
        'open': np.random.uniform(95, 105, size=300),
        'high': np.random.uniform(100, 110, size=300),
        'low': np.random.uniform(90, 100, size=300),
        'close': np.random.uniform(98, 108, size=300),
        'volume': np.random.uniform(1000, 5000, size=300)
    }
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=300, freq='5min'))
    sample_df = pd.DataFrame(data, index=dates)

    # --- Import your entire feature registry ---
    try:
        from features.registry import FEATURE_FUNCTIONS, SESSION_FUNCTIONS
    except ImportError as e:
        print("CRITICAL ERROR: Could not import your feature files.")
        print("Please ensure this script is in the project root directory, and your features are in a 'features' sub-directory.")
        print(f"Details: {e}")
        exit()

    # --- Pre-populate the sample DataFrame with common base indicators ---
    print("Pre-populating sample DataFrame with base indicators...")
    base_indicators = [
        'ema', 'sma', 'atr', 'daily_vwap', 'rsi', 'bollinger', 'adx',
        'choppiness', 'stochastic', 'macd', 'prev_swing',
        'day_high_low' # FIX: Add dependency for 'price_vs_open'
    ]
    for name in base_indicators:
        if name in FEATURE_FUNCTIONS:
            try:
                sample_df = FEATURE_FUNCTIONS[name](sample_df.copy())
            except Exception as e:
                print(f"Warning: Failed to pre-populate with '{name}': {e}")
    print("Base indicators added.\n")
    
    # --- Combine all functions into one dictionary for testing ---
    all_functions_to_test = {**FEATURE_FUNCTIONS, **SESSION_FUNCTIONS}
    
    print("="*80)
    print("🚀 Starting Full Causality Check Test Suite 🚀")
    print(f"Found {len(all_functions_to_test)} functions to test in your registry.")
    print("="*80)

    passed_checks = 0
    failed_checks = 0
    failed_list = []
    error_list = []

    # --- Iterate and test every function in your registry ---
    for name, func in all_functions_to_test.items():
        test_func = lambda df: func(df)
        
        # Use improved result reporting
        result = check_for_lookahead(test_func, sample_df, func_name=name)
        
        if result is True:
            passed_checks += 1
        elif result is False:
            failed_checks += 1
            failed_list.append(name)
        else: # result == "SETUP_ERROR"
            failed_checks += 1
            error_list.append(name)
        
        print("-" * 50)

    print("\n" + "="*80)
    print("🏁 Test Suite Finished 🏁")
    print(f"   Passed: {passed_checks}")
    print(f"   Failed: {failed_checks}")
    print("="*80)

    if failed_checks > 0:
        print("\nACTION REQUIRED: The following functions have issues:")
        if failed_list:
            print("\n--- Look-Ahead Bias Detected ---")
            for name in failed_list:
                print(f"  - {name}")
        if error_list:
            print("\n--- Errors During Testing (check dependencies or function logic) ---")
            for name in error_list:
                print(f"  - {name}")
    else:
        print("\n🎉 Congratulations! All tested functions are free of look-ahead bias.")