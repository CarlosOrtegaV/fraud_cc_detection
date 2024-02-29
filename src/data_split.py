import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import calendar

from tomlkit import date

def create_masks_data_split_per_months(
                     df: pd.DataFrame, 
                     n_splits: int=10,
                     offset_trainval: int=0,
                     offset_test: int=1,
                     validation_ratio: float=0.2,
                     expanding_window: bool=True,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a DataFrame into training and validation sets using an expanding window per months.
    
    Args:
        df {pd.DataFrame}: The DataFrame to be split.
        n_splits {int, optional}: The number of splits to be performed. Defaults to 10.
        offset_trainval {int, optional}: The number of months to be used for the training and validation sets. Defaults to 0.
        validation_ratio {float, optional}: The ratio of validation data to be used. Defaults to 0.2.
        expanding_window {bool, optional}: Whether to use an expanding window or not. Defaults to True.
        
    Returns:
        {Tuple[np.ndarray, np.ndarray]}: A tuple containing the training and validation sets as numpy arrays.
        
    Raises:
        AssertionError: If the number of splits is not between 2 and 10.
        AssertionError: If the data is not from a single year.
    """
    assert offset_test > 0, 'The offset for the test set must be greater than 0.'
    assert (n_splits > 1) & (n_splits <= 10), 'Number of splits must be between 2 and 10.' 
    assert np.unique(df['date'].dt.year).shape[0] == 1, 'The data must be from a single year.'
    
    year = df['date'].dt.year.min()  # Get the year of the data
    
    #  We ensure the training data consists of at least two months
    starting_month = 1  
    list_val_set_first_day = []
    
    if expanding_window == True:
        list_train_set_first_day = [datetime(year, starting_month, 1) for i in range(1, n_splits+1)]
        
    else:
        list_train_set_first_day = [datetime(year, starting_month + i - 1, 1) for i in range(1, n_splits+1)]
    
    trainval_set_last_month = starting_month + offset_trainval
    test_set_last_month = starting_month + offset_trainval
    
    list_trainval_set_last_day = [datetime(year, 
                                           trainval_set_last_month + (1 + offset_test)*(i - 1), 
                                           calendar.monthrange(year, trainval_set_last_month + (1 + offset_test)*(i - 1))[1]
                                           ) for i in range(1, n_splits+1)]
    list_test_set_last_day = [datetime(year, 
                                       trainval_set_last_month + (1 + offset_test)*i, 
                                       calendar.monthrange(year, trainval_set_last_month + (1 + offset_test)*i)[1]
                                       ) for i in range(1, n_splits+1)]
    
    # Create the validation set based on the validation_ratio
    for i in range(n_splits):
        time_gap = list_trainval_set_last_day[i] - list_train_set_first_day[i]
        
        val_days = int(np.round(time_gap.days*validation_ratio))
        val_set_first_day = list_trainval_set_last_day[i] - timedelta(days=val_days)
        list_val_set_first_day.append(val_set_first_day)
    
    
            
    # Create a list to store the training and validation sets
    list_mask_train = []
    list_mask_val = []
    list_mask_test = []
    
    for i in range(n_splits):
        mask_train = (df['date'] >= list_train_set_first_day[i]) & (df['date'] < list_val_set_first_day[i])
        mask_val = (df['date'] >= list_val_set_first_day[i]) & (df['date'] < list_trainval_set_last_day[i])
        mask_test = (df['date'] > list_trainval_set_last_day[i]) & (df['date'] <= list_test_set_last_day[i])
        
        list_mask_train.append(mask_train)
        list_mask_val.append(mask_val)
        list_mask_test.append(mask_test)
        
    # Create np.arrays to store the training and validation sets
    masks_train = np.asarray(list_mask_train)
    masks_val = np.asarray(list_mask_val)
    masks_test = np.asarray(list_mask_test)
    
    return masks_train, masks_val, masks_test
