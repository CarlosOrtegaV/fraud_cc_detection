import pandas as pd

from src.paths import RAW_DATA_DIR
from src.data import download_raw_dataset, preprocess_data
from src.config import FEATURE_GROUP_METADATA
from src.feature_store_api import get_or_create_feature_group
from src.logger import get_logger

logger = get_logger()

def run():

    logger.info('Downloading data from the data warehouse')
    download_raw_dataset()
    df = pd.read_parquet(RAW_DATA_DIR / 'ibm_fraud_cc.parquet')

    # Preprocess the data
    logger.info('Preprocessing raw data')    
    drop_cols = ['card','merchant_city', 'zip']
    df_trans = preprocess_data(df, year=2000, time_delta=60, drop_cols=drop_cols)

    # get a pointer to the feature group we wanna write to
    feature_group = get_or_create_feature_group(FEATURE_GROUP_METADATA)

    # start a job to insert the data into the feature group
    logger.info('Inserting data into feature group...')
    feature_group.insert(df_trans, write_options={"wait_for_job": False})

if __name__ == '__main__':
    run()