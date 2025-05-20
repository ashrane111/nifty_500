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
# Added a non-existent ticker to sample to test robustness
SAMPLE_TICKERS = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'DMART', 'NONEXISTENTTICKER']

SHIFT_PERIOD_10D = 4
SHIFT_PERIOD_20D = 8
SHIFT_PERIOD_50D = 20
SHIFT_PERIOD_100D = 40
RSI_PERIOD = 14

GOOGLE_SHEET_NAME = "Nifty 500 Daily Analysis"  # Or your actual sheet name
WORKSHEET_NAME = "PythonOutput"
GOOGLE_CREDENTIALS_FILE = "google_credentials.json"  # Your credentials file

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

# Prepare list of Yahoo Finance tickers (with .NS suffix)
nifty_500_yf_tickers = [ticker + ".NS" for ticker in nifty_500_base_tickers if ticker]

data_list = []
max_shift_needed = max(SHIFT_PERIOD_10D, SHIFT_PERIOD_20D, SHIFT_PERIOD_50D, SHIFT_PERIOD_100D, RSI_PERIOD, 14)
days_to_fetch = max(350 + max_shift_needed, 2)
start_date_fetch = (datetime.now() - timedelta(days=days_to_fetch)).strftime('%Y-%m-%d')
end_date_fetch = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

print(f"Calculated data fetch period: {days_to_fetch} days (from {start_date_fetch} to {end_date_fetch})")
print(f"DMA Config: 10D(S:{SHIFT_PERIOD_10D}), 20D(S:{SHIFT_PERIOD_20D}), 50D(S:{SHIFT_PERIOD_50D}), 100D(S:{SHIFT_PERIOD_100D})")

all_stock_data_batch = pd.DataFrame()
if nifty_500_yf_tickers:
    print(f"\nFetching EOD data from {start_date_fetch} to {end_date_fetch} for {len(nifty_500_yf_tickers)} tickers in a single batch...")
    try:
        all_stock_data_batch = yf.download(
            nifty_500_yf_tickers,
            start=start_date_fetch,
            end=end_date_fetch,
            progress=True,
            timeout=180,  # Increased timeout for larger request
            auto_adjust=False # Returns 'Adj Close' and standard OHLC
        )
        if all_stock_data_batch.empty and nifty_500_yf_tickers:
             print("Warning: yf.download returned an empty DataFrame for the requested tickers.")
    except Exception as e:
        print(f"CRITICAL ERROR during batch download of stock data: {e}")
        traceback.print_exc()
else:
    print("No tickers to process. Exiting.")
    exit()

# Pre-process batch data into a dictionary of DataFrames, one for each ticker
ticker_data_map = {}
if not all_stock_data_batch.empty:
    if isinstance(all_stock_data_batch.columns, pd.MultiIndex):
        # Level 1 of columns index usually contains ticker symbols if yf.download gets multiple tickers
        unique_tickers_in_data = all_stock_data_batch.columns.get_level_values(1).unique()
        for yf_ticker_from_batch in unique_tickers_in_data:
            try:
                # Extract all price type columns for this specific ticker
                df_one_ticker = all_stock_data_batch.xs(yf_ticker_from_batch, level=1, axis=1)
                ticker_data_map[yf_ticker_from_batch] = df_one_ticker.copy() # Use .copy()
            except Exception as e:
                print(f"Error pre-processing data for {yf_ticker_from_batch} from batch: {e}")
    else:
        # This case is unlikely if yf.download receives a list of tickers,
        # as it should return a MultiIndex column DataFrame even for a single ticker in a list.
        # However, adding a fallback for robustness.
        print("Warning: Batch download did not result in MultiIndex columns as expected. Attempting to handle.")
        if len(nifty_500_yf_tickers) == 1: # If only one ticker was requested
            # Assume the DataFrame is for this single ticker with simple columns
            ticker_data_map[nifty_500_yf_tickers[0]] = all_stock_data_batch.copy()


print(f"\nProcessing {len(nifty_500_base_tickers)} configured tickers...")
for ticker_index, base_ticker in enumerate(nifty_500_base_tickers):
    yf_ticker = base_ticker + ".NS"
    print(f"Processing EOD for {yf_ticker} ({ticker_index + 1}/{len(nifty_500_base_tickers)})...")

    # Initialize values to NaN for the current ticker
    eod_open_price = np.nan
    eod_close_price = np.nan
    prev_day_high_val = np.nan
    fluctuation_pct = np.nan
    dma10_val, dma20_val, dma50_val, dma100_val, rsi_val = (np.nan,) * 5

    try:
        current_stock_df = ticker_data_map.get(yf_ticker)

        if current_stock_df is None or current_stock_df.empty:
            print(f"  No data for {yf_ticker} found in pre-processed batch. Values will remain NaN.")
        elif current_stock_df['Close'].isnull().all():
            print(f"  All 'Close' prices are NaN for {yf_ticker}. Values will remain NaN.")
        else:
            # Valid data found, proceed with calculations
            # Ensure working with rows that have a 'Close' price
            current_stock_df = current_stock_df.dropna(subset=['Close']).copy()
            if current_stock_df.empty:
                 print(f"   Data became empty for {yf_ticker} after dropping NaN Close values.")
            else:
                last_valid_day_data = current_stock_df.iloc[-1]

                eod_open_price = last_valid_day_data.get('Open', np.nan)
                eod_close_price = last_valid_day_data.get('Close', np.nan)

                if len(current_stock_df) >= 2:
                    prev_day_data_row = current_stock_df.iloc[-2]
                    prev_day_high_val = prev_day_data_row.get('High', np.nan)

                if 'Adj Close' in current_stock_df.columns and not current_stock_df['Adj Close'].isnull().all():
                    close_prices_for_ta = current_stock_df['Adj Close'].dropna()
                elif 'Close' in current_stock_df.columns and not current_stock_df['Close'].isnull().all():
                    close_prices_for_ta = current_stock_df['Close'].dropna()
                else:
                    print(f"  No valid 'Adj Close' or 'Close' data for TA for {yf_ticker}.")
                    close_prices_for_ta = pd.Series(dtype=float)

                day_close_price_num = pd.to_numeric(eod_close_price, errors='coerce')
                prev_day_high_val_num = pd.to_numeric(prev_day_high_val, errors='coerce')

                if not np.isnan(prev_day_high_val_num) and not np.isnan(day_close_price_num) and prev_day_high_val_num != 0:
                    fluctuation_pct = ((prev_day_high_val_num - day_close_price_num) / prev_day_high_val_num) * 100

                if not close_prices_for_ta.empty:
                    close_prices_for_ta = pd.to_numeric(close_prices_for_ta, errors='coerce').dropna()
                    if not close_prices_for_ta.empty: # Check again after numeric conversion
                        min_len_for_10d = 10 + SHIFT_PERIOD_10D
                        min_len_for_20d = 20 + SHIFT_PERIOD_20D
                        min_len_for_50d = 50 + SHIFT_PERIOD_50D
                        min_len_for_100d = 100 + SHIFT_PERIOD_100D
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

        data_list.append({
            'Stock Symbol': base_ticker,
            'EOD Open': eod_open_price, 'EOD Close': eod_close_price,
            'Prev Day High': prev_day_high_val,
            '10D DMA': dma10_val, '20D DMA': dma20_val,
            '50D DMA': dma50_val, '100D DMA': dma100_val,
            'RSI': rsi_val,
            'Fluctuation % (EOD)': fluctuation_pct
        })

    except Exception as e:
        print(f"  GENERAL ERROR processing data for {yf_ticker}: {e}")
        traceback.print_exc()
        data_list.append({
            'Stock Symbol': base_ticker, 'EOD Open': 'ERROR', 'EOD Close': 'ERROR',
            'Prev Day High': 'ERROR',
            '10D DMA': 'ERROR', '20D DMA': 'ERROR',
            '50D DMA': 'ERROR', '100D DMA': 'ERROR', 'RSI': 'ERROR',
            'Fluctuation % (EOD)': 'ERROR'
        })

# --- Create DataFrame & Export ---
print("\nConsolidating EOD data into report DataFrame...")
report_df = pd.DataFrame(data_list)
columns_in_order = [
    'Stock Symbol', 'EOD Open', 'EOD Close', 'Prev Day High',
    '10D DMA', '20D DMA', '50D DMA', '100D DMA', 'RSI',
    'Fluctuation % (EOD)'
]
for col in columns_in_order: # Ensure all columns exist even if report_df is empty or missing some
    if col not in report_df.columns:
        report_df[col] = np.nan
report_df = report_df.reindex(columns=columns_in_order)

for col in ['EOD Open', 'EOD Close', 'Prev Day High',
            '10D DMA', '20D DMA', '50D DMA', '100D DMA', 'RSI',
            'Fluctuation % (EOD)']:
    if col in report_df.columns:
        report_df[col] = pd.to_numeric(report_df[col], errors='coerce')
        if not report_df[col].isnull().all(): # Only round if there are non-NaN values
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
        # For Google Sheets, replace NaN with empty strings for better compatibility
        report_df_gsheet = report_df.fillna('')

        gc = gspread.service_account(filename=GOOGLE_CREDENTIALS_FILE)
        spreadsheet = gc.open(GOOGLE_SHEET_NAME)
        try:
            worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
        except gspread.exceptions.WorksheetNotFound:
            print(f"Worksheet '{WORKSHEET_NAME}' not found. Creating it.")
            worksheet = spreadsheet.add_worksheet(title=WORKSHEET_NAME, rows="1", cols=len(report_df_gsheet.columns) if not report_df_gsheet.empty else 1)

        worksheet.clear()
        if not report_df_gsheet.empty: # Check again before writing
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
print("\nNEXT STEPS for Live Data in your viewing Google Sheet (e.g., 'MyDashboard'):")
print("1. Ensure 'MyDashboard' pulls data from the 'PythonOutput' sheet (e.g., using ={'PythonOutput'!A:J} or IMPORTRANGE in A1).")
# ... (rest of the print statements are the same)
print("2. Add a column for 'Today's Live Open'. If stock symbols are in 'MyDashboard'!A2 downwards, in the new column (e.g., K2), use:")
print("   =IF(ISBLANK(A2), \"\", GOOGLEFINANCE(\"NSE:\"&A2, \"priceopen\"))")
print("3. Add a column for 'Live Price'. In another new column (e.g., L2), use:")
print("   =IF(ISBLANK(A2), \"\", GOOGLEFINANCE(\"NSE:\"&A2, \"price\"))")
print("4. Drag these formulas down. These columns will update periodically via Google Finance.")
print("5. If you want a 'Live Fluctuation %' based on these Google Sheet live values, add another column with a formula like (assuming Prev Day High from Python is in 'MyDashboard'!D2, Live Price in L2):")
print("   =IF(AND(ISNUMBER(D2), ISNUMBER(L2), D2<>0), ((D2 - L2) / D2) * 100, \"\")")