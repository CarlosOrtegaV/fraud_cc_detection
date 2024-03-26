import os
from typing import Optional
from pathlib import Path

from comet_ml import Experiment
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score
from xgboost import XGBClassifier
import optuna

from src import config
from src.paths import PARENT_DIR, DATA_CACHE_DIR
from src.config import FEATURE_VIEW_METADATA, N_HYPERPARAMETER_SEARCH_TRIALS
from src.data_split import create_masks_data_split_per_months
from src.feature_store_api import get_or_create_feature_view
from src.model_registry_api import push_model_to_registry
from src.model import get_pipeline, fbeta_score_under_constraint
from src.discord import send_message_to_channel
from src.logger import get_logger

logger = get_logger()

# load variables from .env file as environment variables
load_dotenv(PARENT_DIR / '.env')

def fetch_data_from_store(
    from_date: pd.Timestamp,
    to_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetches time-series data from the store, transforms it into features and
    targets and returns it as a pandas DataFrame.
    """
    # get pointer to featurew view
    logger.info('Getting pointer to feature view...')
    feature_view = get_or_create_feature_view(FEATURE_VIEW_METADATA)
    
    # generate training data from the feature view
    logger.info('Generating training data')
    data, _ = feature_view.training_data(
        description='Training batch of transactions with RFM features',
    )

    # filter data based on the from_date and to_date expressed\
    # as Unix milliseconds
    from_df = int(from_date.timestamp() * 1000)
    to_df = int(to_date.timestamp() * 1000)
    data = data[data['full_date_unix'].between(from_df, to_df)]
    
    return data

def find_best_hyperparameters(
    train_mask_arrays: np.array, 
    val_mask_arrays: np.array,
    df: pd.DataFrame,
    n_trials: Optional[int] = 5,    
) -> dict:
    
    def objective(trial: optuna.trial.Trial) -> float:
        """
        Given a set of hyper-parameters, it trains a model and computes an average
        validation error based on a TimeSeriesSplit
        """
        # pick hyper-parameters
        hyperparams = {
            "n_estimators": trial.suggest_int("n_estimators", 2, 50),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.10, 1.0),
        }
        scores = []
        splits = train_mask_arrays.shape[0]
        for i in range(splits):
            
            # split data for training and validation
            X_train = df.drop(columns=['fraud']).loc[train_mask_arrays[i]]
            y_train = df['fraud'].loc[train_mask_arrays[i]]
            
            X_val = df.drop(columns=['fraud']).loc[val_mask_arrays[i]]
            y_val = np.asarray(df['fraud'].loc[val_mask_arrays[i]]).astype(int)

            # fit the pipeline
            cat_vars = ['errors','use_chip','foreign_transaction', 'morning', 'afternoon','day_week']
            high_cat_vars = ['mcc']
            num_vars = ['amount','amount_log','recency','frequency','monetary']
            m0_xgb = get_pipeline(cat_vars, high_cat_vars, num_vars, **hyperparams)
            m0_xgb.fit(X_train, y_train)
            
            # evaluate the model
            y_proba = m0_xgb.predict_proba(X_val)[:, 1]
            ap = average_precision_score(y_val, y_proba)
            scores.append(ap)
   
        # Return the mean score
        return np.array(scores).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=123)
                                )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    logger.info(f'{best_params=}')

    return best_params

def load_data(
    local_path_data: Optional[Path] = None,
) -> pd.DataFrame:
    
    if local_path_data:
        logger.info('Loading data from local file')
        df = pd.read_parquet(local_path_data)
    else:
        logger.info('Fetching data from the feature store')
        # Set the desired year to 2000
        year = 2000

        # Create datetime objects for the start and end dates of the year 2000
        from_date = pd.to_datetime(f'{year}-01-01')
        to_date = pd.to_datetime(f'{year}-12-31')

        df = fetch_data_from_store(from_date, to_date)

        # save data to local file
        try:
            local_file = DATA_CACHE_DIR / 'data.parquet'
            df.to_parquet(local_file)
            logger.info(f'Saved data to local file at {local_file}')
        except:
            logger.info('Could not save data to local file')
            pass

    return df


def train(
    local_path_data: Optional[Path] = None,
) -> None:
    """
    Trains model and pushes it to the model registry if it meets the minimum
    performance threshold.
    """
    logger.info('Start model training...')

    # start Comet ML experiment run
    logger.info('Creating Comet ML experiment')
    experiment = Experiment(
        api_key=os.environ["COMET_ML_API_KEY"],
        workspace=os.environ["COMET_ML_WORKSPACE"],
        project_name = os.environ['COMET_ML_PROJECT_NAME'],
    )

    # load features and targets
    df = load_data(local_path_data)
    experiment.log_dataset_hash(df)

    # split the data into training and validation sets
    n_splits = 3
    offset_trainval = 5
    offset_test = 1
    topk = 0.001
    train_masks, val_masks, test_masks = create_masks_data_split_per_months(df,
                                                                   n_splits=n_splits,
                                                                   offset_trainval=offset_trainval,
                                                                   offset_test=offset_test
                                                                   )
    logger.info(f'Splitting data into training and test sets')

    experiment.log_parameters({
        'n_splits': n_splits,
        'offset_trainval': offset_trainval,
        'offset_test': offset_test,
        'topk': topk
    })
    
    # find the best hyperparameters using time-based cross-validation
    logger.info('Finding best hyperparameters...')
    best_hyperparameters = find_best_hyperparameters(train_masks,
                                                     val_masks, 
                                                     df,
                                                     n_trials=N_HYPERPARAMETER_SEARCH_TRIALS)
    experiment.log_parameters(best_hyperparameters)
    experiment.log_parameter('N_HYPERPARAMETER_SEARCH_TRIALS', N_HYPERPARAMETER_SEARCH_TRIALS)

    # train the model using the best hyperparameters
    logger.info('Training model using the best hyperparameters...')  
    
    number_split = train_masks.shape[0]
    # fit the pipeline
    cat_vars = ['errors','use_chip','foreign_transaction', 'morning', 'afternoon','day_week']
    high_cat_vars = ['mcc']
    num_vars = ['amount','amount_log','recency','frequency','monetary']
    m0_train = get_pipeline(cat_vars, high_cat_vars, num_vars, **best_hyperparameters)

    list_fbeta_score = []
    list_fbeta_max_score = []
    list_precision = []
    list_recall = []
    list_recall_money = []
    list_optimal_threshold = []
    beta_ = 2
    
    for split in range(number_split):

        X_train = df.drop(columns=['fraud']).loc[train_masks[split]]
        y_train = df['fraud'].loc[train_masks[split]]
        X_val = df.drop(columns=['fraud']).loc[val_masks[split]]
        y_val = np.asarray(df['fraud'].loc[val_masks[split]]).astype(int)
        

        amount_val = df['amount'].loc[val_masks[split]].values.reshape(-1,)
        
        m0_train.fit(X_train, y_train)

        pred_val_m0 = m0_train.predict_proba(X_val)[:, 1]
        
        n_topkperc = np.round((len(pred_val_m0[np.argsort(pred_val_m0)[::-1]])*topk),0).astype(int)
        ix_sorted_top_val = np.argsort(pred_val_m0)[::-1]
        
        for i in range(1,n_topkperc+1):

            pseudo_labels = np.ones_like(pred_val_m0[ix_sorted_top_val[:i]]).astype(int)
            
            precision_under_constraint = precision_score(y_val[ix_sorted_top_val[:i]], pseudo_labels) 
            list_precision.append(precision_under_constraint)
            
            recall_under_constraint = np.sum(y_val[ix_sorted_top_val[:i]])/np.sum(y_val) 
            list_recall.append(recall_under_constraint)
            
            recall_monetary_under_constraint = np.sum(y_val[ix_sorted_top_val[:i]]* \
                                                    amount_val[ix_sorted_top_val[:i]]) / \
                                                    np.sum(y_val*amount_val)
            list_recall_money.append(recall_monetary_under_constraint)
            
            fbeta_score_ = fbeta_score_under_constraint(recall_under_constraint, precision_under_constraint, beta_)
            list_fbeta_score.append(fbeta_score_)
        
        my_dict = dict(zip(np.sort(pred_val_m0)[::-1], list_fbeta_score))
        max_key = max(my_dict, key=my_dict.get)
        optimal_threshold = max_key
        
        list_optimal_threshold.append(optimal_threshold)
        list_fbeta_max_score.append(max(my_dict.values()))
        
    optimal_treshold_test = np.median(list_optimal_threshold)
    
    list_recall_test_ = []
    list_precision_test_ = []
    list_proportion_investigations_ = []

    for split in range(number_split):

        trainval_masks = train_masks[split] + val_masks[split]

        X_trainval = df.drop(columns=['fraud']).loc[trainval_masks]
        y_trainval = df['fraud'].loc[trainval_masks]
        
        X_test = df.drop(columns=['fraud']).loc[test_masks[split]]
        y_test = np.asarray(df['fraud'].loc[test_masks[split]]).astype(int)
        

        amount_test = df['amount'].loc[test_masks[split]].values.reshape(-1,)
        
        m0_trainval = get_pipeline(cat_vars, high_cat_vars, num_vars, **best_hyperparameters)
        m0_trainval.fit(X_trainval, y_trainval)

        pred_test_m0 = m0_trainval.predict_proba(X_test)[:, 1]
        
        label_pred_test = (pred_test_m0 > optimal_treshold_test).astype(int)
        
        recall_monetary_test_set = np.sum(label_pred_test*amount_test*y_test)/np.sum(y_test*amount_test)

        recall_score_ = recall_score(y_test, label_pred_test)
        precision_score_ = precision_score(y_test, label_pred_test)
        proportion_investigations = np.sum((pred_test_m0 > optimal_threshold).astype(int))/y_test.shape[0]

        list_recall_test_.append(recall_score_)
        list_precision_test_.append(precision_score_)
        list_proportion_investigations_.append(proportion_investigations)
    
        mean_recall = np.mean(list_recall_test_)
        mean_precision = np.mean(list_precision_test_)
        mean_recall_money = np.mean(recall_monetary_test_set)
        mean_prop_invs = np.mean(list_proportion_investigations_)
        
    logger.info(f'mean_recall={100*mean_recall:.2f}%')
    logger.info(f'mean_precision={100*mean_precision:.2f}%')
    logger.info(f'mean_recall_money={100*mean_recall_money:.2f}%')
    logger.info(f'mean_prop_invs={100*mean_prop_invs:.2f}% out of the whole test set, and out of {100*(mean_prop_invs/topk):.2f}% investigations permitted')
    
    experiment.log_metric('test_recall', mean_recall)
    
    # push the model to the Hopsworks model registry if it meets the minimum performance threshold
    experiment.log_parameter('MIN_RECALL', config.MIN_RECALL)
    if mean_recall > config.MIN_RECALL:
        logger.info('Pushing model to the model registry...')
        model_version = push_model_to_registry(
            model=m0_trainval,
            model_name=config.MODEL_NAME,
        )
        logger.info(f'Model version {model_version} pushed to the model registry.')

        # add model version to the experiment in CometML
        experiment.log_parameter('model_version', model_version)

        # send notification on Discord
        send_message_to_channel(f'New model pushed to the model registry. mean_recall={100*mean_recall:.2f}%, {model_version=}')
        
    else:
        logger.info('Model did not meet the minimum performance threshold. Skip pushing to the model registry.')


if __name__ == '__main__':

    from fire import Fire
    Fire(train)