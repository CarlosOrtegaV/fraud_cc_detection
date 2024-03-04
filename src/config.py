import os
from dotenv import load_dotenv
from src.feature_store_api import FeatureGroupConfig
from src.paths import PARENT_DIR

load_dotenv(PARENT_DIR / '.env')

HOPSWORKS_PROJECT_NAME = 'fraud_detection'
try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception('Create a .env file on project root with HOPSWORKS_API_KEY.')

FEATURE_GROUP_NAME = 'transactions_with_rfm_feature_group'
FEATURE_GROUP_VERSION = 2

FEATURE_GROUP_METADATA = FeatureGroupConfig(
    name='time_series_hourly_feature_group',
    version=FEATURE_GROUP_VERSION,
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION,
    description="Transaction data with RFM features at hourly frequency",
    primary_key = ['full_date_unix','user','amount'],
    event_time='full_date_unix',
    online_enabled=True,
)