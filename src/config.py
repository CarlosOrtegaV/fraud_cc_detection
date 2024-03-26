import os
from dotenv import load_dotenv
from src.feature_store_api import FeatureGroupConfig, FeatureViewConfig
from src.paths import PARENT_DIR

load_dotenv(PARENT_DIR / '.env')

HOPSWORKS_PROJECT_NAME = 'fraud_detection'
try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception('Create a .env file on project root with HOPSWORKS_API_KEY.')


# Feature Store Backfilling
FEATURE_GROUP_NAME = 'transactions_with_rfm_feature_group'
FEATURE_GROUP_VERSION = 1
FEATURE_GROUP_METADATA = FeatureGroupConfig(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION,
    description="Transaction data with RFM features at hourly frequency",
    primary_key = ['full_date_unix','user','id_amount'],
    event_time='full_date_unix',
    online_enabled=True,
)

FEATURE_VIEW_NAME = 'transactions_with_rfm_feature_feature_view'
FEATURE_VIEW_VERSION = 2
FEATURE_VIEW_METADATA = FeatureViewConfig(
    name=FEATURE_VIEW_NAME,
    version=FEATURE_GROUP_VERSION,
    feature_group=FEATURE_GROUP_METADATA,
)

MODEL_NAME = "fraud_detector"

# Hyperparameter search
N_HYPERPARAMETER_SEARCH_TRIALS = 10

# Minimum recall for the model based on the assumed fraud investigation efficiency P(y=1|l=1) = 1/5
# P(l=1)/P(y=1) ~~ 1, and the bayes' theorem P(l=1|y=1) ~~ P(y=1|l=1)
MIN_RECALL = 0.20