import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import traceback
import gspread
from gspread_dataframe import set_with_dataframe
import numpy as np

# --- Configuration ---
USE_SAMPLE_TICKERS = False  # SET TO FALSE FOR FULL NIFTY 500 RUN
SAMPLE_TICKERS = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'DMART', 'NONEXISTENTTICKER']

# DMA Config
SHIFT_PERIOD_10D = 4
SHIFT_PERIOD_20D = 8
SHIFT_PERIOD_50D = 20
SHIFT_PERIOD_100D = 40

# RSI Config
RSI_PERIOD = 14

# ATR Config
ATR_PERIOD_1 = 14 # For ATR 14
ATR_PERIOD_2 = 50 # For ATR 50

# Bollinger Bands Config
BB_LENGTH = 20
BB_STD_DEV = 2.0 # Standard deviation

GOOGLE_SHEET_NAME = "Nifty 500 Daily Analysis"
WORKSHEET_NAME = "PythonOutput"
GOOGLE_CREDENTIALS_FILE = "google_credentials.json"

if USE_SAMPLE_TICKERS:
    nifty_500_base_tickers = SAMPLE_TICKERS
    print(f"--- Using SAMPLE tickers: {nifty_500_base_tickers} ---")
else:
    try:
        csv_file_name = "ind_nifty500list.csv"
        symbol_column_name = "Symbol"
        nifty_500_df = pd.read_csv(csv_file_name)
        nifty_500_base_tickers = [
            str(ticker).strip()
            for ticker in nifty_500_df[symbol_column_name].dropna().unique().tolist()
            if str(ticker).strip()
        ]
        if not nifty_500_base_tickers:
            print(f"Ticker list is empty from CSV. Please check the file and column name.")
            exit()
        print(f"--- Loaded {len(nifty_500_base_tickers)} base tickers from {csv_file_name} ---")
    except Exception as e:
        print(f"Error loading ticker CSV: {e}")
        traceback.print_exc()
        exit()

nifty_500_yf_tickers = [ticker + ".NS" for ticker in nifty_500_base_tickers if ticker]

data_list = []
# Determine max lookback needed, considering all indicators
max_lookback_for_indicators = max(
    SHIFT_PERIOD_10D + 10, SHIFT_PERIOD_20D + 20,
    SHIFT_PERIOD_50D + 50, SHIFT_PERIOD_100D + 100,
    RSI_PERIOD * 2, ATR_PERIOD_1 + 5, ATR_PERIOD_2 + 5,
    BB_LENGTH + 5, # For Bollinger Bands
    14 # General minimum
)
days_to_fetch = max(350 + max_lookback_for_indicators, ATR_PERIOD_2 + 50, BB_LENGTH + 50)
start_date_fetch = (datetime.now() - timedelta(days=days_to_fetch)).strftime('%Y-%m-%d')
end_date_fetch = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

print(f"Calculated data fetch period: {days_to_fetch} days (from {start_date_fetch} to {end_date_fetch})")
print(f"DMA Config: 10D(S:{SHIFT_PERIOD_10D}), 20D(S:{SHIFT_PERIOD_20D}), 50D(S:{SHIFT_PERIOD_50D}), 100D(S:{SHIFT_PERIOD_100D})")
print(f"ATR Config: {ATR_PERIOD_1}-day, {ATR_PERIOD_2}-day")
print(f"Bollinger Bands Config: Length={BB_LENGTH}, StdDev={BB_STD_DEV}")


all_stock_data_batch = pd.DataFrame()
if nifty_500_yf_tickers:
    print(f"\nFetching EOD data from {start_date_fetch} to {end_date_fetch} for {len(nifty_500_yf_tickers)} tickers in a single batch...")
    try:
        all_stock_data_batch = yf.download(
            nifty_500_yf_tickers,
            start=start_date_fetch,
            end=end_date_fetch,
            progress=True,
            timeout=240,
            auto_adjust=False
        )
        if all_stock_data_batch.empty and nifty_500_yf_tickers:
             print("Warning: yf.download returned an empty DataFrame for the requested tickers.")
    except Exception as e:
        print(f"CRITICAL ERROR during batch download of stock data: {e}")
        traceback.print_exc()
else:
    print("No tickers to process. Exiting.")
    exit()

ticker_data_map = {}
if not all_stock_data_batch.empty:
    if isinstance(all_stock_data_batch.columns, pd.MultiIndex):
        unique_tickers_in_data = all_stock_data_batch.columns.get_level_values(1).unique()
        for yf_ticker_from_batch in unique_tickers_in_data:
            try:
                df_one_ticker = all_stock_data_batch.xs(yf_ticker_from_batch, level=1, axis=1)
                ticker_data_map[yf_ticker_from_batch] = df_one_ticker.copy()
            except Exception as e:
                print(f"Error pre-processing data for {yf_ticker_from_batch} from batch: {e}")
    else:
        print("Warning: Batch download did not result in MultiIndex columns as expected. Attempting to handle.")
        if len(nifty_500_yf_tickers) == 1:
            ticker_data_map[nifty_500_yf_tickers[0]] = all_stock_data_batch.copy()

print(f"\nProcessing {len(nifty_500_base_tickers)} configured tickers...")
for ticker_index, base_ticker in enumerate(nifty_500_base_tickers):
    yf_ticker = base_ticker + ".NS"
    print(f"Processing EOD for {yf_ticker} ({ticker_index + 1}/{len(nifty_500_base_tickers)})...")

    # Initialize all values
    eod_open_price, eod_close_price, prev_day_high_val = (np.nan,) * 3
    # fluctuation_pct removed
    dma10_val, dma20_val, dma50_val, dma100_val, rsi_val = (np.nan,) * 5
    atr14_val, atr50_val = (np.nan,) * 2
    # bb_bandwidth_val removed, only keeping PercentB and the bands themselves
    bb_lower_val, bb_middle_val, bb_upper_val, bb_percent_b_val = (np.nan,) * 4

    try:
        current_stock_df = ticker_data_map.get(yf_ticker)

        if current_stock_df is None or current_stock_df.empty:
            print(f"  No data for {yf_ticker} found in pre-processed batch. Values will remain NaN.")
        elif current_stock_df['Close'].isnull().all():
            print(f"  All 'Close' prices are NaN for {yf_ticker}. Values will remain NaN.")
        else:
            current_stock_df = current_stock_df.dropna(subset=['Close']).copy() # Work with rows having a Close price
            if current_stock_df.empty:
                 print(f"   Data became empty for {yf_ticker} after dropping NaN Close values.")
            else:
                last_valid_day_data = current_stock_df.iloc[-1]
                eod_open_price = last_valid_day_data.get('Open', np.nan)
                eod_close_price = last_valid_day_data.get('Close', np.nan)

                if len(current_stock_df) >= 2:
                    prev_day_data_row = current_stock_df.iloc[-2]
                    prev_day_high_val = prev_day_data_row.get('High', np.nan)

                # Determine close prices for TA calculations
                if 'Adj Close' in current_stock_df.columns and not current_stock_df['Adj Close'].isnull().all():
                    close_prices_for_ta = current_stock_df['Adj Close'].dropna()
                elif 'Close' in current_stock_df.columns and not current_stock_df['Close'].isnull().all():
                    close_prices_for_ta = current_stock_df['Close'].dropna()
                else:
                    print(f"  No valid 'Adj Close' or 'Close' data for TA for {yf_ticker}.")
                    close_prices_for_ta = pd.Series(dtype=float)

                # Fluctuation % calculation removed

                # DMAs and RSI
                if not close_prices_for_ta.empty:
                    close_prices_for_ta = pd.to_numeric(close_prices_for_ta, errors='coerce').dropna()
                    if not close_prices_for_ta.empty:
                        min_len_for_10d = 10 + SHIFT_PERIOD_10D; min_len_for_20d = 20 + SHIFT_PERIOD_20D
                        min_len_for_50d = 50 + SHIFT_PERIOD_50D; min_len_for_100d = 100 + SHIFT_PERIOD_100D
                        min_len_for_rsi = RSI_PERIOD * 2
                        if RSI_PERIOD < 10: min_len_for_rsi = RSI_PERIOD + 15

                        if len(close_prices_for_ta) >= min_len_for_10d:
                            s = close_prices_for_ta.rolling(window=10).mean().shift(SHIFT_PERIOD_10D)
                            if not s.empty and not s.isnull().all(): dma10_val = s.iloc[-1]
                        if len(close_prices_for_ta) >= min_len_for_20d:
                            s = close_prices_for_ta.rolling(window=20).mean().shift(SHIFT_PERIOD_20D)
                            if not s.empty and not s.isnull().all(): dma20_val = s.iloc[-1]
                        if len(close_prices_for_ta) >= min_len_for_50d:
                            s = close_prices_for_ta.rolling(window=50).mean().shift(SHIFT_PERIOD_50D)
                            if not s.empty and not s.isnull().all(): dma50_val = s.iloc[-1]
                        if len(close_prices_for_ta) >= min_len_for_100d:
                            s = close_prices_for_ta.rolling(window=100).mean().shift(SHIFT_PERIOD_100D)
                            if not s.empty and not s.isnull().all(): dma100_val = s.iloc[-1]
                        if len(close_prices_for_ta) >= min_len_for_rsi:
                            s = ta.rsi(close_prices_for_ta, length=RSI_PERIOD)
                            if s is not None and not s.empty and not s.isnull().all(): rsi_val = s.iloc[-1]
                    else:
                        print(f"  Close prices for TA became empty after numeric conversion for {yf_ticker}.")
                else:
                    print(f"  Close price series for TA is empty for {yf_ticker}.")

                # --- ATR Calculation ---
                required_cols_for_atr = ['High', 'Low', 'Close']
                cols_present_and_not_all_nan_atr = all(col in current_stock_df.columns for col in required_cols_for_atr) and \
                                           not current_stock_df[required_cols_for_atr].isnull().all().all()
                if cols_present_and_not_all_nan_atr:
                    if len(current_stock_df) >= ATR_PERIOD_1:
                        try:
                            current_stock_df.ta.atr(length=ATR_PERIOD_1, append=True)
                            atr14_col_name = f'ATRr_{ATR_PERIOD_1}'
                            if atr14_col_name in current_stock_df.columns and not pd.isna(current_stock_df[atr14_col_name].iloc[-1]):
                                atr14_val = current_stock_df[atr14_col_name].iloc[-1]
                        except Exception as e_atr14: print(f"    Error calculating ATR_{ATR_PERIOD_1} for {yf_ticker}: {e_atr14}")
                    if len(current_stock_df) >= ATR_PERIOD_2:
                        try:
                            current_stock_df.ta.atr(length=ATR_PERIOD_2, append=True)
                            atr50_col_name = f'ATRr_{ATR_PERIOD_2}'
                            if atr50_col_name in current_stock_df.columns and not pd.isna(current_stock_df[atr50_col_name].iloc[-1]):
                                atr50_val = current_stock_df[atr50_col_name].iloc[-1]
                        except Exception as e_atr50: print(f"    Error calculating ATR_{ATR_PERIOD_2} for {yf_ticker}: {e_atr50}")
                else: print(f"    Missing or all-NaN HLC data for ATR calculation for {yf_ticker}.")

                # --- Bollinger Bands Calculation ---
                # pandas_ta bbands returns: BBL (Lower), BBM (Middle), BBU (Upper),
                # BBB (Bandwidth Percent = (BBU-BBL)/BBM ), BBP (Percent B = (Price-BBL)/(BBU-BBL) )
                if 'Close' in current_stock_df.columns and not current_stock_df['Close'].isnull().all().any():
                    if len(current_stock_df) >= BB_LENGTH:
                        try:
                            current_stock_df.ta.bbands(length=BB_LENGTH, std=BB_STD_DEV, append=True)
                            bbl_col = f'BBL_{BB_LENGTH}_{BB_STD_DEV:.1f}' # pandas_ta appends std with .1f format
                            bbm_col = f'BBM_{BB_LENGTH}_{BB_STD_DEV:.1f}'
                            bbu_col = f'BBU_{BB_LENGTH}_{BB_STD_DEV:.1f}'
                            # bbb_col removed (Bollinger Bandwidth BBB_...)
                            bbp_col = f'BBP_{BB_LENGTH}_{BB_STD_DEV:.1f}' # Percent B

                            if bbl_col in current_stock_df.columns and not pd.isna(current_stock_df[bbl_col].iloc[-1]):
                                bb_lower_val = current_stock_df[bbl_col].iloc[-1]
                            if bbm_col in current_stock_df.columns and not pd.isna(current_stock_df[bbm_col].iloc[-1]):
                                bb_middle_val = current_stock_df[bbm_col].iloc[-1]
                            if bbu_col in current_stock_df.columns and not pd.isna(current_stock_df[bbu_col].iloc[-1]):
                                bb_upper_val = current_stock_df[bbu_col].iloc[-1]
                            # bb_bandwidth_val extraction removed
                            if bbp_col in current_stock_df.columns and not pd.isna(current_stock_df[bbp_col].iloc[-1]):
                                bb_percent_b_val = current_stock_df[bbp_col].iloc[-1]
                        except Exception as e_bb:
                            print(f"    Error calculating Bollinger Bands for {yf_ticker}: {e_bb}")
                    else:
                        print(f"    Not enough data rows for Bollinger Bands for {yf_ticker} (need >= {BB_LENGTH}, got {len(current_stock_df)}).")
                else:
                    print(f"    Missing or all-NaN Close data for Bollinger Bands calculation for {yf_ticker}.")

        data_list.append({
            'Stock Symbol': base_ticker,
            'EOD Open': eod_open_price, 'EOD Close': eod_close_price,
            'Prev Day High': prev_day_high_val,
            '10D DMA': dma10_val, '20D DMA': dma20_val,
            '50D DMA': dma50_val, '100D DMA': dma100_val,
            'RSI': rsi_val,
            f'ATR_{ATR_PERIOD_1}': atr14_val,
            f'ATR_{ATR_PERIOD_2}': atr50_val,
            f'BB_Lower_{BB_LENGTH}_{BB_STD_DEV:.1f}': bb_lower_val,
            f'BB_Middle_{BB_LENGTH}_{BB_STD_DEV:.1f}': bb_middle_val,
            f'BB_Upper_{BB_LENGTH}_{BB_STD_DEV:.1f}': bb_upper_val,
            # BB_Bandwidth column removed
            f'BB_PercentB_{BB_LENGTH}_{BB_STD_DEV:.1f}': bb_percent_b_val
            # Fluctuation % (EOD) column removed
        })

    except Exception as e:
        print(f"  GENERAL ERROR processing data for {yf_ticker}: {e}")
        traceback.print_exc()
        data_list.append({
            'Stock Symbol': base_ticker, 'EOD Open': 'ERROR', 'EOD Close': 'ERROR',
            'Prev Day High': 'ERROR', '10D DMA': 'ERROR', '20D DMA': 'ERROR',
            '50D DMA': 'ERROR', '100D DMA': 'ERROR', 'RSI': 'ERROR',
            f'ATR_{ATR_PERIOD_1}': 'ERROR', f'ATR_{ATR_PERIOD_2}': 'ERROR',
            f'BB_Lower_{BB_LENGTH}_{BB_STD_DEV:.1f}': 'ERROR',
            f'BB_Middle_{BB_LENGTH}_{BB_STD_DEV:.1f}': 'ERROR',
            f'BB_Upper_{BB_LENGTH}_{BB_STD_DEV:.1f}': 'ERROR',
            # BB_Bandwidth error entry removed
            f'BB_PercentB_{BB_LENGTH}_{BB_STD_DEV:.1f}': 'ERROR'
            # Fluctuation % (EOD) error entry removed
        })

# --- Create DataFrame & Export ---
print("\nConsolidating EOD data into report DataFrame...")
report_df = pd.DataFrame(data_list)
columns_in_order = [
    'Stock Symbol', 'EOD Open', 'EOD Close', 'Prev Day High',
    '10D DMA', '20D DMA', '50D DMA', '100D DMA', 'RSI',
    f'ATR_{ATR_PERIOD_1}', f'ATR_{ATR_PERIOD_2}',
    f'BB_Lower_{BB_LENGTH}_{BB_STD_DEV:.1f}',
    f'BB_Middle_{BB_LENGTH}_{BB_STD_DEV:.1f}',
    f'BB_Upper_{BB_LENGTH}_{BB_STD_DEV:.1f}',
    # BB_Bandwidth column removed from order
    f'BB_PercentB_{BB_LENGTH}_{BB_STD_DEV:.1f}'
    # Fluctuation % (EOD) column removed from order
]
for col in columns_in_order:
    if col not in report_df.columns:
        report_df[col] = np.nan
report_df = report_df.reindex(columns=columns_in_order)

cols_to_numeric_round = [
    'EOD Open', 'EOD Close', 'Prev Day High',
    '10D DMA', '20D DMA', '50D DMA', '100D DMA', 'RSI',
    f'ATR_{ATR_PERIOD_1}', f'ATR_{ATR_PERIOD_2}',
    f'BB_Lower_{BB_LENGTH}_{BB_STD_DEV:.1f}',
    f'BB_Middle_{BB_LENGTH}_{BB_STD_DEV:.1f}',
    f'BB_Upper_{BB_LENGTH}_{BB_STD_DEV:.1f}',
    # BB_Bandwidth column removed from rounding
    f'BB_PercentB_{BB_LENGTH}_{BB_STD_DEV:.1f}'
    # Fluctuation % (EOD) column removed from rounding
]
for col in cols_to_numeric_round:
    if col in report_df.columns:
        report_df[col] = pd.to_numeric(report_df[col], errors='coerce')
        if not report_df[col].isnull().all():
             # For PercentB, you might want more precision
             if f'BB_PercentB_{BB_LENGTH}_{BB_STD_DEV:.1f}' == col :
                 report_df[col] = report_df[col].round(4) # Example: 4 decimal places for PercentB
             else:
                 report_df[col] = report_df[col].round(2)


file_timestamp = datetime.now().strftime('%Y-%m-%d')
run_timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
csv_report_name = f"Nifty500_EOD_Report_{'SAMPLE_' if USE_SAMPLE_TICKERS else 'ALL_'}{file_timestamp}.csv"
excel_report_name = f"Nifty500_EOD_Report_{'SAMPLE_' if USE_SAMPLE_TICKERS else 'ALL_'}{run_timestamp}.xlsx"

try:
    report_df.to_csv(csv_report_name, index=False)
    print(f"\nEOD Report successfully generated as CSV: {csv_report_name}")
    report_df.to_excel(excel_report_name, index=False, engine='openpyxl')
    print(f"EOD Report successfully generated as Excel: {excel_report_name}")

    if not report_df.empty:
        print("\nFirst 5 rows of the EOD report:")
        print(report_df.head())
except Exception as e:
    print(f"\nError writing local report files: {e}")
    traceback.print_exc()

if not report_df.empty:
    print(f"\nAttempting to upload data to Google Sheet: '{GOOGLE_SHEET_NAME}', Worksheet: '{WORKSHEET_NAME}'")
    try:
        report_df_gsheet = report_df.fillna('')

        gc = gspread.service_account(filename=GOOGLE_CREDENTIALS_FILE)
        spreadsheet = gc.open(GOOGLE_SHEET_NAME)
        try:
            worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
        except gspread.exceptions.WorksheetNotFound:
            print(f"Worksheet '{WORKSHEET_NAME}' not found. Creating it.")
            worksheet = spreadsheet.add_worksheet(title=WORKSHEET_NAME, rows="1", cols=len(report_df_gsheet.columns) if not report_df_gsheet.empty else 1)

        worksheet.clear()
        if not report_df_gsheet.empty:
            set_with_dataframe(worksheet, report_df_gsheet, include_index=False, include_column_header=True, resize=True)
            print(f"Successfully updated Google Sheet '{GOOGLE_SHEET_NAME}' -> Worksheet '{WORKSHEET_NAME}'.")
            print(f"You can view it at: https://docs.google.com/spreadsheets/d/{spreadsheet.id}")
        else:
            print("Report DataFrame for Google Sheets is empty. Cleared worksheet but uploaded no data.")
    except Exception as e:
        print(f"An ERROR occurred while updating Google Sheets: {e}")
        traceback.print_exc()
else:
    print("\nReport DataFrame is empty. Skipping Google Sheets upload.")

print("\nPython EOD script finished.")
