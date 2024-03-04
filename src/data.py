from pathlib import Path
import requests

from src.paths import RAW_DATA_DIR, PREPROCESSED_DATA_DIR

from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import tqdm
from datetime import datetime, timedelta


def calculate_rfm_for_client(
                             df: pd.DataFrame, 
                             user: str, 
                             amount: float, 
                             time_reference: pd.Timestamp, 
                             time_window: int
                             ) -> Tuple[float, int, float]:
    """Calculate RFM features based on a time window for a single client."""
    
    # Calculate the start and end of the time window
    time_window_start = time_reference - pd.DateOffset(days=time_window)
    time_window_end = time_reference
    
    # Filter the DataFrame to only include rows for the client and within the time window
    df_client = df[(df['user'] == user) & (df['date'] >= time_window_start) & (df['date'] <= time_window_end)]
    
    # Calculate Recency
    sorted_dates = df_client['date'].sort_values(ascending=False)
    last_trx_date = time_window_start if len(sorted_dates) == 1 else sorted_dates.iloc[1]
    delta_date = (time_window_end -  last_trx_date).days
    gamma_recency = 0.05  # Decay factor for recency based on Baesens Hoppner and Verdonck (2021)
    recency = np.exp(-1*gamma_recency*delta_date) 
    
    # Calculate frequency feature based on the number of transactions
    frequency = df_client.shape[0]
    
    # Calculate monetary feature based on the ratio of each trx and the median of trx amounts 
    monetary = amount / df_client['amount'].median()
    
    return recency, frequency, monetary

def calculate_rfm(
                  df: pd.DataFrame, 
                  time_window: int
                  ) -> pd.DataFrame:
    """Calculate RFM features based on a time window for each instance."""
    
    # Initialize lists to store the results
    recency = []
    frequency = []
    monetary = []
    
    # Calculate the RFM features for each instance
    for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Calculating RFM features'):
        r, f, m = calculate_rfm_for_client(df, row['user'], row['amount'], row['date'], time_window)

        frequency.append(f)
        monetary.append(m)
        recency.append(r)

    # Create a new DataFrame for the RFM features
    rfm = pd.DataFrame({
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary
    })
    
    # Concatenate the original DataFrame with the new DataFrame
    df = pd.concat([df.reset_index(drop=True), rfm], axis=1)
    
    return df

def preprocess_data(
                    df: pd.DataFrame,
                    year: int,  # Year of the dataset for modeling
                    time_delta: int,  # Lag time for RFM features 
                    drop_cols: Optional[List[str]]
                    ) -> pd.DataFrame:
    """
    Preprocess the credit card transactions dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The input data frame to be preprocessed. It should contain credit card transaction data.
    year : int
        The year of the dataset for modeling. This is used to filter the data and calculate features.
    time_delta : int
        The lag time for RFM (Recency, Frequency, Monetary) features. This is used in the calculation of RFM features.
    drop_cols : Optional[List[str]]
        A list of column names to be dropped from the data frame. If None, no columns will be dropped.

    Returns
    -------
    df : pd.DataFrame
        The preprocessed data frame. It has renamed columns, reformatted feature values, filtered transactions, 
        a single datetime column, and calculated RFM features.

    Notes
    -----
    The function performs the following steps:
    1. Renames the columns using the `rename_columns` function.
    2. Reformats the feature values using the `reformat_feature_values` function.
    3. Filters out transactions with non-positive amounts and drops specified columns.
    4. Converts the date and time columns to a single datetime column.
    5. Filters the data based on the specified year and a 90-day period before the first day of that year.
    6. Calculates the RFM features using the `calculate_rfm` function with the specified time window.
    7. Creates other features using the `create_other_features` function.
    8. Filters the data based on the specified year.
    """
     
    # Rename the columns
    df = df.copy()
    cols = df.columns
    processed_columns = [col.lower().replace(' ', '_') for col in cols]
    processed_columns = [col.lower().replace('?', '') for col in processed_columns]
    df.rename(columns=dict(zip(df.columns, processed_columns)), inplace=True)
    df.rename(columns={'is_fraud': 'fraud'}, inplace=True)
    
    # Reformat the feature values
    df['fraud'] = df['fraud'].map({'Yes': True, 'No': False})
    df['use_chip'] = df['use_chip'].str.replace(' ','_').str.lower()
    df['errors'] = df['errors'].fillna('no_error').str.replace(' ','_').str.lower()
    df['merchant_state'] = df['merchant_state'].fillna('online').str.replace(' ','_').str.lower()
    df['amount'] = df['amount'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    
    # Create a features based on the amount and merchant_state
    df = df[df['amount']>0]
    df['amount_log'] = np.log(df['amount'] + 1)
    df['foreign_transaction'] = df['merchant_state'].apply(lambda x: True if x != 'online' and len(str(x)) > 2 else False)
    
    # Create a new features based on the date
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['day_week'] = df['date'].dt.dayofweek
    df['hour'] = df['time'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['time'].apply(lambda x: int(x.split(':')[1]))
    df['morning'] = df['hour'].apply(lambda x: True if (x >= 9) & (x < 12) else False)
    df['afternoon'] = df['hour'].apply(lambda x: True if (x >= 12) & (x < 18) else False)
    df['full_date'] = pd.to_datetime(df[['year', 'month', 'day','hour','minute']])
    
    # Filter out transactions with non-positive amounts and drop columns

    df.drop(drop_cols, axis=1, inplace=True)
    
    # Convert the date and time columns to a single datetime column
    
    
    # Create a datetime object for the first day of the year
    first_day_year = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    
    # Calculate 90 days before the first day of the year
    start_date = first_day_year - timedelta(days=90)

    
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df = df.loc[mask]
    
    # Calculate the RFM features
    df = calculate_rfm(df, time_window=time_delta)
    
    df = df[df['year'] == year]
    
    return df


def download_dataset() -> Path:
    """Download the IBM fraud dataset from Dropbox and save it to the local directory."""
    
    URL = 'https://www.dropbox.com/scl/fi/mn09ew3r0bbydw8kmnvy6/creditcard_altman.csv?rlkey=dpj4s0bmkubu5uqifdpfy3bow&dl=1'
    response = requests.get(URL)
    if response.status_code == 200:
        path = Path(RAW_DATA_DIR / 'ibm_fraud_cc.csv')
        open(path, 'wb').write(response.content)   
        return path
    else:
        raise Exception("Failed to download CSV file. Status code:", response.status_code)

def download_parquet_dataset() -> pd.DataFrame:
    """Download a Parquet dataset from a URL and save it to the local directory."""
    
    URL = 'https://www.dropbox.com/scl/fi/65p2ox1uafn2838hbjk9z/full_data_2000_2005.parquet?rlkey=k6pzj2b1eb0fs7ml7zr5e1sz1&dl=1'
    response = requests.get(URL)
    if response.status_code == 200:
        path = Path(PREPROCESSED_DATA_DIR / 'full_data_2000_2005.parquet')
        open(path, 'wb').write(response.content)
        return path
    else:
        raise Exception("Failed to download Parquet file. Status code:", response.status_code)

def fetch_batch_data(
                     from_date: datetime, 
                     to_date: datetime
                     ) -> pd.DataFrame:
    """
    Simulate production data by sampling historical data from 23 years ago.
    """
    local_file = PREPROCESSED_DATA_DIR / 'full_data_2000_2005.parquet'
    if not local_file.exists():
        try:
            download_parquet_dataset()
        except:
            raise Exception('Failed to download preprocessed data (Parquet file)')
    else:
        print('File on preprocessed data already exists')

    df_filtered = pd.read_parquet(local_file)
    
    years_shift = 23  # how many years to go back in time

    fetch_data_to_ = to_date.replace(year=to_date.year - years_shift)
    fetch_data_from_ = from_date.replace(year=from_date.year - years_shift)
    print(f'{fetch_data_from_=}, {fetch_data_to_=}')
    
    
    df_filtered = df_filtered[(df_filtered['full_date'] >= fetch_data_from_) & (df_filtered['full_date'] < fetch_data_to_)]

    return df_filtered