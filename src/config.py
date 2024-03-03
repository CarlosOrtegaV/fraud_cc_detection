import os
from dotenv import load_dotenv

from src.paths import PARENT_DIR

load_dotenv(PARENT_DIR / '.env')

HOPSWORKS_PROJECT_NAME = 'fraud_detection'
try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception('Create a .env file on project root with HOPSWORKS_API_KEY.')

FEATURE_GROUP_NAME = 'transactions_with_rfm_feature_group'
FEATURE_GROUP_VERSION = 2