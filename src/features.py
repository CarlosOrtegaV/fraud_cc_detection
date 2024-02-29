import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing._target_encoder import TargetEncoder
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

import xgboost as xgb

class IsolationForestAnomalyScore(BaseEstimator, TransformerMixin):
  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)
    
  def fit(self, X, y=None):
    n_estimators = getattr(self, 'n_estimators', 100)
    
    self.iforest = IsolationForest(n_estimators=n_estimators)
    self.iforest.fit(X)
    return self

  def transform(self, X):
    anomaly_scores = self.iforest.decision_function(X)
    return anomaly_scores
      
      
class CatNarrow(BaseEstimator, TransformerMixin):
  "Narrows the categories of a categorical variable based on a threshold."

  def __init__(self, threshold: float=0.10):
    self.threshold = threshold
    
  def fit(self, X, y=None):
    X_ = np.array(X).astype(str)
    self.dc_filtered  = []
    
    # For each column, check the frequency of each category and filter based on threshold
    for c in range(X_.shape[1]):
      X_aux = X_[:, c]
      unique, counts = np.unique(X_aux, return_counts=True)
      counts_normalized = counts / sum(counts)
      dc = dict(zip(unique, counts_normalized))
      dc_filtered_ = {k: v for k, v in dc.items() if v >= self.threshold}  # Only keep categories with frequency >= threshold
      dc_filtered_ = np.array(list(dc_filtered_.keys())).reshape(-1,1)
      self.dc_filtered.append(dc_filtered_)


  def transform(self, X):
    """Replace categories with frequency < threshold by 'undefined'.
    
    Args:
      X (np.ndarray): The input array containing categories.
      
    Returns:
      X_trans (np.ndarray): The transformed array with replaced categories.
    """
       
    X_ = np.array(X).astype(str)
    list_aux = []
    
    for c in range(X_.shape[1]):  
      X_aux = np.array(X_[:, c]).reshape(-1,1)
      dc_filtered_ = self.dc_filtered[c]
      np.place(X_aux, np.isin(X_aux, dc_filtered_, invert = True), ['undefined']) # Replace categories with frequency < threshold by 'undefined'
      list_aux.append(X_aux)
    
    X_trans = np.concatenate(list_aux, axis = 1)
    return X_trans


# numeric_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='median')),
#                                         ('scaler', StandardScaler())
#                                         ]
#                                )

# categorical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='constant',
#                                                                       fill_value='missing_value')),
#                                             ('catnarrow', CatNarrow(threshold = 0.10)),
#                                             ('targe_enc', TargetEncoder()),
#                                             ('scaler', StandardScaler())
#                                             ]
#                                    )
                                   
# processing = ColumnTransformer(transformers=[
#                                             ('cat', categorical_transformer, cat_vars),
#                                             ('num', numeric_transformer, num_vars)
#                                             ],
#                                remainder='drop'
#                               )

# pipe = Pipeline(steps=[
#                       ('processing', processing),
#                       ('isolation_forest', IsolationForestAnomalyScore(n_estimators=100)),
#                       ('classifier', xgb.XGBClassifier())
#                        ]
#               )



# pipe_e = pipe.fit(X.iloc[ix_tr_all], y_e[ix_tr_all])

