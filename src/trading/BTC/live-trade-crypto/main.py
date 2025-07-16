# main.py

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import joblib
#from sklearn.preprocessing import OneHotEncoder

from utils.pipeline import apply_feature_engineering_live

from .progressive_obv import ProgressiveResamplingOBV
from . import config
from . import data_handler
from . import binance_client
from . import trade_manager_mod

def load_model_and_encoder(model_path='src/trading/BTC/live-trade-crypto/lookahead57_03-gemini-custom-score-improved-crypto.pkl', encoder_path='src/trading/BTC/live-trade-crypto/lookahead57_03-encoder-crypto.pkl'):
    """Loads the ML model and the one-hot encoder from disk."""
    try:
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        print(f"Successfully loaded model from {model_path} and encoder from {encoder_path}")
        return model, encoder
    except FileNotFoundError:
        print(f"ERROR: Model or encoder file not found. Please ensure '{model_path}' and '{encoder_path}' are present.")
        return None, None
    
def load_model(model_path='src/trading/BTC/live-trade-crypto/lookahead57_03-gemini-custom-score-improved-crypto.pkl'):
    """Loads the ML model and the one-hot encoder from disk."""
    try:
        model = joblib.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except FileNotFoundError:
        print(f"ERROR: Model file not found. Please ensure '{model_path}' is present.")
        return None, None

# def get_model_prediction(latest_features: pd.Series, model, encoder) -> int:
#     """
#     Prepares the latest data, applies one-hot encoding, and returns the model's prediction.
#     This version gets the expected feature list directly from the model object.
#     """
#     #if model is None or encoder is None:
#     if model is None:
#         print("Model or encoder not loaded. Cannot make a prediction.")
#         return 0

#     model_features = set(model.feature_names_in_)
#     cat_cols = ['Regime_5min', 'session', 'Regime_15min', 'Regime_1h']

#     features_to_keep = [col for col in latest_features.index if col in model_features or col in cat_cols]
#     latest_features = latest_features[features_to_keep]
    
#     live_df = pd.DataFrame(latest_features).T
    
#     # Separate numeric and categorical, keeping only columns that will be used.
#     # We must separate original features before they are one-hot-encoded.
#     num_cols = [col for col in live_df.columns if col not in cat_cols]
#     live_num_features = live_df[num_cols]
#     #live_cat_features = live_df[cat_cols]

#     try:
#         # REORDER CATEGORICAL FEATURES TO MATCH ENCODER'S EXPECTATIONS
#         # Get the feature names and order the encoder was fitted with.
#         #encoder_expected_cols = encoder.feature_names_in_

#         # Reorder the live categorical features to match the expected order.
#         #live_cat_features_ordered = live_cat_features[encoder_expected_cols]
        
#         # One-Hot Encode the correctly ordered categorical features.
#         #live_cat_encoded = encoder.transform(live_cat_features_ordered)
        
#         # Also use the expected order when getting the new feature names.
#         #cat_feature_names = encoder.get_feature_names_out(encoder_expected_cols)

#         #live_cat_df = pd.DataFrame(live_cat_encoded, columns=cat_feature_names, index=live_df.index)

#         # Combine to create the final DataFrame for prediction
#         #final_df = pd.concat([live_num_features, live_cat_df], axis=1)
#         final_df = live_num_features

#         # Get the feature names the model was trained on
#         model_columns = model.feature_names_in_
        
#         # Check for discrepancies
#         model_cols_set = set(model_columns)
#         final_df_cols_set = set(final_df.columns)

#         missing_features = model_cols_set - final_df_cols_set
#         extra_features = final_df_cols_set - model_cols_set

#         if missing_features:
#             print(f"ERROR: The following features are MISSING from the data but required by the model: {missing_features}")
#             return 0 # Stop prediction if features are missing

#         if extra_features:
#             # This is not a fatal error, but good to know.
#             print(f"NOTE: The following features were generated but are not used by the model: {extra_features}")

#         # Reorder the DataFrame columns to match the model's expected order EXACTLY.
#         final_df = final_df[model_columns]

#         # Make prediction
#         prediction = model.predict(final_df)
#         print(f"Model prediction: {prediction[0]}")
#         return int(prediction[0])

#     except Exception as e:
#         print(f"Error during prediction pre-processing: {e}")
#         import traceback
#         traceback.print_exc()
#         return 0

def get_model_prediction(latest_features: pd.Series, model) -> int:
    """Prepares the latest data and returns the model's prediction."""
    if model is None:
        print("Model not loaded. Cannot make a prediction.")
        return 0

    model_columns = model.feature_names_in_
    live_df = pd.DataFrame(latest_features).T
    
    # Keep only the columns the model expects
    final_df = live_df[model_columns]

    prediction = model.predict(final_df)
    print(f"Model prediction: {prediction[0]}")
    return int(prediction[0])

def wait_for_next_complete_candle():
    """Waits for the next 5-minute candle boundary plus a small buffer."""
    now_utc = datetime.now(timezone.utc)
    next_interval_start = now_utc.replace(minute=(now_utc.minute // 5) * 5, second=0, microsecond=0) + timedelta(minutes=5)
    WAKE_BUFFER = timedelta(seconds=5)
    target_time = next_interval_start + WAKE_BUFFER
    sleep_duration = (target_time - now_utc).total_seconds()
    
    if sleep_duration > 0:
        print(f"Sleeping for {sleep_duration:.1f} seconds until next candle is ready...")
        return sleep_duration
    return 1 # If we're already past, check again shortly

def verify_latest_candle_complete(df):
    """
    Verify that the latest candle in our data is actually complete.
    A candle that opens at time T is complete after time T+5min.
    """
    if df.empty:
        return False
    
    now_utc = datetime.now(timezone.utc)
    latest_candle_open = df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
    
    # Calculate when this candle closed (candle open + 5 minutes)
    candle_close_time = latest_candle_open + timedelta(minutes=5)
    
    # Check if enough time has passed since the candle closed
    time_since_close = now_utc - candle_close_time
    
    # We need at least a small buffer after close time
    REQUIRED_BUFFER = timedelta(seconds=5)
    
    if time_since_close < REQUIRED_BUFFER:
        print(f"WARNING: Latest candle at {latest_candle_open} closed only {time_since_close.total_seconds():.1f}s ago - might be incomplete!")
        return False
    
    # Also verify it's on a proper 5-minute boundary
    if latest_candle_open.minute % 5 != 0 or latest_candle_open.second != 0:
        print(f"WARNING: Latest candle timestamp {latest_candle_open} is not on a 5-minute boundary!")
        return False
    
    print(f"Latest candle at {latest_candle_open} is complete (closed {time_since_close.total_seconds():.1f}s ago)")
    return True
            

# Replace the data loading and updating section in main.py

def main():
    print("--- Starting Robust Trading Bot ---")
    
    model = load_model()
    if model is None:
        print("--- Bot shutting down due to missing model/encoder. ---")
        return

    print(f"Mode: {'SIMULATED' if config.SIMULATED else 'LIVE - REAL MONEY'}")
    print(f"Symbol: {config.TRADING_SYMBOL}, Leverage: {config.LEVERAGE}x")

    if not config.SIMULATED:
        binance_client.client.set_leverage(config.TRADING_SYMBOL, config.LEVERAGE)

    # IMPORTANT: Use consistent file names
    # Option 1: Use the same file for everything
    MASTER_DATA_FILE = "btcusdt_5m_data.parquet"
    
    # Load master data
    master_data = pd.read_parquet(config.CANDLE_DATA_FILE)
    if master_data.empty:
        return

    # Ensure timezone awareness
    if master_data.index.tz is None:
        master_data.index = master_data.index.tz_localize('UTC')

    # --- NEW, ROBUST INITIALIZATION ---
    timeframes_to_calc = ['5min', '15min', '1h']
    obv_column_names = [f'OBV_{tf}' for tf in timeframes_to_calc]
    progressive_calculators = {tf: ProgressiveResamplingOBV(timeframe=tf) for tf in timeframes_to_calc}

    # Check if OBV columns already exist
    obv_exists = all(col in master_data.columns for col in obv_column_names)
    
    if not obv_exists:
        print("Initializing OBV columns...")
        # Your existing OBV initialization code here
        master_data.sort_index(inplace=True)
        resampling_rules = {'close': 'last', 'volume': 'sum'}
        
        for tf in timeframes_to_calc:
            resampled_data = master_data.resample(tf).agg(resampling_rules).dropna()
            if resampled_data.empty:
                print(f"Warning: No data to resample for timeframe {tf}. Skipping.")
                master_data[f'OBV_{tf}'] = np.nan
                continue

            signed_volume = resampled_data['close'].diff().fillna(0).apply(np.sign) * resampled_data['volume']
            obv_series = signed_volume.cumsum()
            obv_series.name = f'OBV_{tf}'
            master_data = master_data.merge(obv_series, left_index=True, right_index=True, how='left')

            calculator = progressive_calculators[tf]
            calculator.last_obv = obv_series.iloc[-1]
            calculator.last_close = resampled_data['close'].iloc[-1]
            calculator._last_processed_timestamp = obv_series.index[-1]
            calculator.buffer = master_data.iloc[-20:].copy()

        master_data[obv_column_names] = master_data[obv_column_names].ffill()
        print("OBV initialization complete.")
    else:
        print("OBV columns already exist. Setting up calculators from existing data...")
        # Initialize calculators from existing OBV data
        for tf in timeframes_to_calc:
            calculator = progressive_calculators[tf]
            obv_col = f'OBV_{tf}'
            
            # Get the last valid OBV value
            last_valid_idx = master_data[obv_col].last_valid_index()
            if last_valid_idx is not None:
                calculator.last_obv = master_data.loc[last_valid_idx, obv_col]
                calculator.last_close = master_data.loc[last_valid_idx, 'close']
                calculator._last_processed_timestamp = last_valid_idx
                calculator.buffer = master_data.iloc[-20:].copy()

    trade_manager = trade_manager_mod.TradeManager()
    first_iteration = True

    while True:
        try:
            print(f"\n{'='*60}")
            print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC] Starting analysis cycle")
            print(f"{'='*60}")
            
            # --- 1. FETCH NEW CANDLES ---
            last_timestamp = master_data.index[-1]
            print(f"Last timestamp in master_data: {last_timestamp}")
            
            # Get the latest data (this returns only OHLCV, no OBV)
            updated_ohlcv_data = data_handler.get_latest_data()
            
            if not updated_ohlcv_data.empty:
                # Ensure timezone awareness
                if updated_ohlcv_data.index.tz is None:
                    updated_ohlcv_data.index = updated_ohlcv_data.index.tz_localize('UTC')
                
                # Get only the new candles (those after our last timestamp)
                new_candles = updated_ohlcv_data[updated_ohlcv_data.index > last_timestamp]
                
                if not new_candles.empty:
                    print(f"Found {len(new_candles)} new candles. Updating features progressively.")
                    
                    # Only take OHLCV columns from new candles to avoid duplicates
                    new_candles_ohlcv = new_candles[['open', 'high', 'low', 'close', 'volume']].copy()
                    
                    # --- 2. CALCULATE OBV FOR NEW CANDLES ---
                    # Create temporary DataFrame to hold new OBV values
                    new_obv_values = pd.DataFrame(index=new_candles_ohlcv.index)
                    
                    for tf, calculator in progressive_calculators.items():
                        new_obv_series = calculator.update(new_candles_ohlcv)
                        obv_col = f'OBV_{tf}'
                        
                        # Add new OBV values to our temporary DataFrame
                        for timestamp, value in new_obv_series.items():
                            new_obv_values.loc[timestamp, obv_col] = value
                    
                    # --- 3. MERGE NEW DATA WITH MASTER ---
                    # Combine OHLCV and OBV data for new candles
                    new_complete_data = pd.concat([new_candles_ohlcv, new_obv_values], axis=1)

                    # ----------------- FIX STARTS HERE -----------------
                    # The resampling for longer timeframes (like 1h) can create future timestamps.
                    # We must truncate new_complete_data to ensure it doesn't extend beyond the
                    # last candle we actually fetched from the exchange.
                    if not updated_ohlcv_data.empty:
                        last_complete_timestamp = updated_ohlcv_data.index[-1]
                        new_complete_data = new_complete_data[new_complete_data.index <= last_complete_timestamp]
                    # ------------------ FIX ENDS HERE ------------------
                    
                    # Forward fill OBV values (in case some are missing)
                    for obv_col in obv_column_names:
                        if obv_col in new_complete_data.columns:
                            # Fill from the last known value in master_data
                            last_value = master_data[obv_col].iloc[-1]
                            new_complete_data[obv_col] = new_complete_data[obv_col].ffill().fillna(last_value)
                    
                    # Append to master data
                    master_data = pd.concat([master_data, new_complete_data])
                    master_data = master_data[~master_data.index.duplicated(keep='last')]
                    
                    # Save the updated data
                    master_data.to_parquet(config.CANDLE_DATA_FILE)
                    print(f"Data updated and saved. Total candles: {len(master_data)}")
                else:
                    print("No new candles found.")
            else:
                print("Data fetch returned empty DataFrame.")

            #print(f"Master dataset contains {len(master_data)} candles.")
            #print(f"Master data columns: {master_data.columns.tolist()}")

            # --- 4. CALCULATE OTHER FEATURES ON A RECENT SLICE ---
            slice_start_time = datetime.now(timezone.utc) - timedelta(days=5)
            
            # Ensure master_data index is timezone aware before slicing
            if master_data.index.tz is None:
                master_data.index = master_data.index.tz_localize('UTC')
                
            # Convert slice_start_time to match the timezone of master_data
            if master_data.index.tz is not None:
                slice_start_time = slice_start_time.replace(tzinfo=master_data.index.tz)
            
            data_slice = master_data.loc[master_data.index >= slice_start_time].copy()
            
            #print(f"Data slice contains {len(data_slice)} candles")
            #print(f"Data slice range: {data_slice.index[0]} to {data_slice.index[-1]}")

            # Calculate features
            featured_data = apply_feature_engineering_live(
                                data_slice,
                                timeframes=["5min","15min","1h"], # Check if this matches your function
                                base_tf="5min",
                                features=[ # List of features *excluding* 'obv'
                                    'day_high_low',
                                    'price_vs_open', 'candle', 'volume_features',
                                    'obv_session', 'obv_relative', 'wick_pct', 'atr'
                                ]
                            )
            
            # ----------------- FIX STARTS HERE -----------------
            # The feature engineering function renames the core OBV columns.
            # We will rename ONLY these three specific columns back to what the model expects.
            columns_to_rename = {
                'OBV_5min_5min': 'OBV_5min',
                'OBV_15min_5min': 'OBV_15min',
                'OBV_1h_5min': 'OBV_1h'
            }

            # Rename the columns that exist in the DataFrame
            featured_data.rename(columns=columns_to_rename, inplace=True)
            # ------------------ FIX ENDS HERE ------------------
            # print("Sending file to parquet")
            # featured_data.to_parquet("features_live.parquet")
            if featured_data.empty:
                print("Feature calculation resulted in an empty DataFrame. Waiting for more data.")
                sleep_duration = wait_for_next_complete_candle()
                time.sleep(sleep_duration)
                continue
                
            
            latest = featured_data.iloc[-1]
            
            print(f"\nAnalyzing candle: {latest.name}")
            print(f"Price: ${latest.close:.2f}")
            
            # Verify this candle is complete
            candle_time = latest.name.to_pydatetime().replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            candle_close_time = candle_time + timedelta(minutes=5)
            time_since_close = now_utc - candle_close_time
            
            if time_since_close < timedelta(seconds=5):
                print(f"WARNING: Analyzing potentially incomplete candle (closed only {time_since_close} ago)")
                print("Skipping this cycle for safety.")
                sleep_duration = wait_for_next_complete_candle()
                time.sleep(sleep_duration)
                continue

            # CHECK POSITION MANAGEMENT
            if trade_manager.in_trade:
                # First, check if continuous monitoring detected an executed order
                continuous_exit_reason, continuous_exit_price = trade_manager.check_for_executed_orders()
                if continuous_exit_reason:
                    print(f"🎯 Continuous monitor detected {continuous_exit_reason} execution!")
                    trade_manager.close_trade(continuous_exit_price, continuous_exit_reason)
                else:
                    # Second, sync trade state with actual position (this handles cancelled orders)
                    sync_result = trade_manager.sync_trade_state_with_position()
                    if sync_result[0]:  # If sync found an exit
                        exit_reason, exit_price = sync_result
                        trade_manager.close_trade(exit_price, exit_reason)
                    else:
                        # Third, normal exit condition checking (this includes manual order checks)
                        exit_reason, exit_price = trade_manager.check_exit_conditions(latest.high, latest.low)
                        if exit_reason:
                            if config.SIMULATED:
                                trade_manager.close_trade(exit_price, exit_reason)
                            else:
                                # For live trading, the order should already be executed
                                # We just need to confirm and close the trade record
                                trade_manager.close_trade(exit_price, exit_reason)
                        else:
                            print("In trade. No exit signal found.")
            
            # CHECK FOR NEW ENTRY SIGNALS (only if not in a trade)
            if not trade_manager.in_trade and not first_iteration:
                pred = get_model_prediction(latest, model)
                
                side = None
                if pred == 1: side = 'long'
                elif pred == 2: side = 'short'
                
                if side:
                    # Use only the 5min ATR for risk calculation
                    atr_for_risk = latest.get('ATR_14_5min', 0)
                    
                    if atr_for_risk <= 0:
                        print(f"Skipping trade: ATR_14_5min is {atr_for_risk}.")
                    else:
                        print(f"Using ATR_14_5min = {atr_for_risk:.6f} for risk calculation")
                        entry_price = latest.close
                        
                        if config.SIMULATED:
                            current_balance = trade_manager.simulated_balance
                        else:
                            current_balance = binance_client.client.get_futures_balance('USDT')
                            print(f"Current Balance {current_balance}")
                        
                        if current_balance > 0:
                            position_value = current_balance * config.RISK_FRACTION * config.LEVERAGE
                            position_size = position_value / entry_price
                            position_size = round(position_size, 3)
                            print(f"Position size: {position_size}")

                            if config.SIMULATED:
                                trade_manager.enter_trade(side, entry_price, position_size, atr_for_risk)
                            else:
                                order_side = 'BUY' if side == 'long' else 'SELL'
                                order = binance_client.client.create_market_order(config.TRADING_SYMBOL, order_side, position_size)
                                if order:
                                    # Portfolio Margin might return different field names
                                    actual_entry_price = None
                                    
                                    # Try different possible field names for the execution price
                                    if 'avgPrice' in order and order['avgPrice'] and float(order['avgPrice']) > 0:
                                        actual_entry_price = float(order['avgPrice'])
                                    elif 'price' in order and order['price'] and float(order['price']) > 0:
                                        actual_entry_price = float(order['price'])
                                    elif 'executedPrice' in order and order['executedPrice'] and float(order['executedPrice']) > 0:
                                        actual_entry_price = float(order['executedPrice'])
                                    
                                    # Check if order was filled
                                    if 'status' in order:
                                        order_status = order['status']
                                        if order_status == 'FILLED':
                                            print(f"Market order filled immediately at ${actual_entry_price}")
                                        elif order_status == 'NEW' or order_status == 'PARTIALLY_FILLED':
                                            print(f"Market order status: {order_status}")
                                            # For unfilled market orders, check execution after a brief wait
                                            if not actual_entry_price or actual_entry_price <= 0:
                                                print("Waiting for market order execution...")
                                                time.sleep(2)  # Wait for execution
                                                
                                                # Check order status again
                                                try:
                                                    updated_order = binance_client.client.get_order_status(config.TRADING_SYMBOL, order['orderId'], False)
                                                    if updated_order:
                                                        print(f"Updated order status: {updated_order.get('status', 'UNKNOWN')}")
                                                        if updated_order.get('avgPrice') and float(updated_order['avgPrice']) > 0:
                                                            actual_entry_price = float(updated_order['avgPrice'])
                                                            print(f"Order executed at ${actual_entry_price}")
                                                except Exception as e:
                                                    print(f"Could not check updated order status: {e}")
                                    
                                    # If we still don't have a price, use the latest close as estimate
                                    if not actual_entry_price or actual_entry_price <= 0:
                                        print(f"WARNING: Could not get execution price from order. Using latest close: {entry_price}")
                                        print(f"Order response: {order}")
                                        actual_entry_price = entry_price
                                    
                                    trade_manager.enter_trade(side, actual_entry_price, position_size, atr_for_risk)
                                else:
                                    print("CRITICAL: Failed to create entry order!")
                        else:
                            print("Cannot trade: balance is zero.")
                else:
                    print("No entry signal found.")

            # ROBUST SLEEP LOGIC
            first_iteration = False
            sleep_duration = wait_for_next_complete_candle()
            time.sleep(sleep_duration)

        except KeyboardInterrupt:
            print("\nBot stopped manually.")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            import traceback
            traceback.print_exc()
            print("Waiting 60 seconds before retrying...")
            time.sleep(60)

if __name__ == "__main__":
    main()