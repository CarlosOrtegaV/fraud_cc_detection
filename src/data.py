from pathlib import Path
import requests

from src.paths import RAW_DATA_DIR

from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import tqdm
from datetime import datetime, timedelta

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns of a DataFrame to a consistent format."""

    cols = df.columns
    processed_columns = [col.lower().replace(' ', '_') for col in cols]
    processed_columns = [col.lower().replace('?', '') for col in processed_columns]
    df.rename(columns=dict(zip(df.columns, processed_columns)), inplace=True)
    df.rename(columns={'is_fraud': 'fraud'}, inplace=True)
    return df

def reformat_feature_values(df: pd.DataFrame) -> pd.DataFrame:
    """Reformat the columns of a DataFrame to the appropriate data types."""
    
    df['fraud'] = df['fraud'].map({'Yes': True, 'No': False})
    df['use_chip'] = df['use_chip'].str.replace(' ','_').str.lower()
    df['errors'] = df['errors'].fillna('no_error').str.replace(' ','_').str.lower()
    df['merchant_state'] = df['merchant_state'].fillna('online').str.replace(' ','_').str.lower()
    df['amount'] = df['amount'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    
    return df


def create_other_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features based on the original columns of the DataFrame."""
    
    # Create a new feature based on the amount
    df['amount_log'] = np.log(df['amount'] + 1)
    
    # Create a new feature based on the date
    df['day_week'] = df['date'].dt.dayofweek
    df['hour'] = df['time'].str[:2].astype(int)
    df['foreign_transaction'] = df['merchant_state'].apply(lambda x: True if x != 'online' and len(str(x)) > 2 else False)
    df['morning'] = df['hour'].apply(lambda x: True if (x >= 9) & (x < 12) else False)
    df['afternoon'] = df['hour'].apply(lambda x: True if (x >= 12) & (x < 18) else False)

    return df

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

def preprocess_data(df: pd.DataFrame,
                    year: int,  # Year of the dataset for modeling
                    time_delta: int,  # Lag time for RFM features 
                    drop_cols: Optional[List[str]]) -> pd.DataFrame:
    """Preprocess the credit card transactions dataset."""
       
    # Rename the columns
    df = rename_columns(df)
    
    # Reformat the feature values
    df = reformat_feature_values(df)
    
    # Filter out transactions with non-positive amounts and drop columns
    df = df[df['amount']>0]
    df.drop(drop_cols, axis=1, inplace=True)
    
    # Convert the date and time columns to a single datetime column
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    # Create a datetime object for the first day of the year
    first_day_year = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    
    # Calculate 90 days before the first day of the year
    start_date = first_day_year - timedelta(days=90)

    
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df = df.loc[mask]
    
    # Calculate the RFM features
    df = calculate_rfm(df, time_window=time_delta)
    df = create_other_features(df)
    
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