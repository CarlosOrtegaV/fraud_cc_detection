from datetime import datetime, timedelta
from argparse import ArgumentParser
from pdb import set_trace as stop

import pandas as pd

from src import config
from src.data import fetch_batch_data
from src.feature_store_api import get_feature_group, get_or_create_feature_group
from src.logger import get_logger

logger = get_logger()

def run(date: datetime):
    """_summary_

    Args:
        date (datetime): _description_

    Returns:
        _type_: _description_
    """
    logger.info('Fetching raw data from data warehouse')
    
    # we fetch raw data for the last 1 week, to add redundancy to our data pipeline, 
    # if the pipeline fails for some reason,
    # we can still re-write data for that missing hour in a later run.
    batch_df_ = fetch_batch_data(
        from_date=(date - timedelta(weeks=1)),
        to_date=date
    )

    # get a pointer to the feature group we wanna write to
    logger.info('Getting pointer to the feature group we wanna save data to')
    feature_group = get_or_create_feature_group(config.FEATURE_GROUP_METADATA)

    # Insert data into the feature group
    logger.info('Starting job to insert data into feature group...')
    feature_group.insert(batch_df_, write_options={"wait_for_job": False})
    
    logger.info('Finished job to insert data into feature group')

if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--datetime',
                        type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
                        help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS')
    args = parser.parse_args()

    # if args.datetime was provided, use it as the current_date, otherwise
    # use the current datetime in UTC
    if args.datetime:
        current_date = pd.to_datetime(args.datetime)
    else:
        current_date = pd.to_datetime('2001-01-08')
    
    logger.info(f'Running feature pipeline for {current_date=}')
    run(current_date)