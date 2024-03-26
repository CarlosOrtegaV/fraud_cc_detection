from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from xgboost import XGBClassifier

import numpy as np
from typing import List


def fbeta_score_under_constraint(recall: float, precision: float, beta: float) -> float:
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

class TopCategoriesEncoder(BaseEstimator, TransformerMixin):
  def __init__(self, n=5):
    self.n = n
    self.top_categories = []

  def fit(self, X, y=None):
    # Calculate the proportion of the positive class for each category
    
    proportions = self.group_mean(X, y)

    # Sort the categories by the proportion of the positive class
    sorted_categories = dict(sorted(proportions.items(), key=lambda item: item[1], reverse=True))

    # Store the top n categories
    self.top_categories = list(sorted_categories.keys())[:self.n]

    return self
    
  def transform(self, X):
    # For each top category, add a new binary feature to X
    X = X.flatten()
    X_trans = np.zeros((X.shape[0], len(self.top_categories)))
    for i, category in enumerate(self.top_categories):
    
      X_trans[:, i] = (X == category)

    return X_trans

  @staticmethod
  def group_mean(X, y):
    # Get unique categories
    categories = np.unique(X)
    X = X.flatten()

    # Compute mean for each category
    means = {category: np.mean(y[X == category]) for category in categories}
  
    return means
    
class AddIsolationForestAnomalyScore(BaseEstimator, TransformerMixin):
  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)
    
  def fit(self, X, y=None):
    n_estimators = getattr(self, 'n_estimators', 100)
    
    self.iforest = IsolationForest(n_estimators=n_estimators)
    self.iforest.fit(X)
    return self

  def transform(self, X):
    anomaly_scores = self.iforest.decision_function(X).reshape(-1, 1)
    return np.hstack([X, anomaly_scores])
      
      
class CatNarrow(BaseEstimator, TransformerMixin):
  "Narrows the categories of a categorical variable based on a threshold."

  def __init__(self, threshold: float=0.05):
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
    
    return self
  
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
  

def get_pipeline(cat_vars: List[str], 
                 high_cat_vars: List[str], 
                 num_vars: List[str],
                 **hyperparams,
    )->Pipeline:

    num_transformer = Pipeline(steps = [
                                        ('imputer', SimpleImputer(strategy='median')),
                                        ('scaler', StandardScaler())
                                        ]
                            )

    cat_transformer = Pipeline(steps = [
                                        ('imputer', SimpleImputer(
                                                                strategy='constant',
                                                                fill_value=-9999
                                                                )
                                        ),
                                        ('catnarrow', CatNarrow(threshold = 0.05)),
                                        ('target_enc', TargetEncoder()),
                                        ('scaler', StandardScaler())
                                                ]
                                    )

    high_cat_transformer = Pipeline(steps = [
                                        ('imputer', SimpleImputer(
                                                                strategy='constant',
                                                                fill_value=-9999
                                                                )
                                        ),
                                        ('topcat_enc', TopCategoriesEncoder(n=3)),
                                        ('target_enc', TargetEncoder()),
                                        ('scaler', StandardScaler())
                                                ]
                                    )
                                    

    processing = ColumnTransformer(transformers=[
                                                ('cat', cat_transformer, cat_vars),
                                                ('high_cat', high_cat_transformer, high_cat_vars),
                                                ('num', num_transformer, num_vars)
                                                ],
                                remainder='drop'
                                )
    
    pipe = Pipeline(steps=[
                        ('processing', processing),
                        ('isolation_forest', AddIsolationForestAnomalyScore(n_estimators=100)),
                        ('xgboost', XGBClassifier(**hyperparams,
                                                  random_state=123,
                                                  n_jobs=-1))
                        ]
                    )   
    return pipe