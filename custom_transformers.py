import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    A transformer that performs Target Encoding on specified column in a Pandas DataFrame.

        Parameters:
        -----------
        col : string
            The column name of the categorical variable you want to target encode
        target : string
            The column name of the target variable
        method : string, default='mean'
            If 'mean' categories are encoded by getting mean target
            If 'median' categories are encoded by getting median target
        random_state : int, default=42
            Seed for the random number generator used for imputing missing values.
        Returns:
        --------
        A new Pandas DataFrame with the specified columns winsorized.

    """
    def __init__(self, cols, target = None, method = ['mean'], random_state = 42):
        self.random_state = random_state
        self.cols = cols
        self.target = target
        self.method = method
        self.cats_names = dict()
        self.cats_encoded = dict()
        self.encoding = dict()
        
    def fit(self, X, y = None):
        self.feature_names = X.columns
        X_temp = X.copy()
        X_temp[self.target] = y
        for i, col in enumerate(self.cols):
            cats_mapped = X_temp.groupby(col)[self.target].mean()
            if self.method[i] == 'mean':
                cats_mapped = X_temp.groupby(col)[self.target].mean()
            elif self.method[i] == 'median':
                cats_mapped = X_temp.groupby(col)[self.target].median()
            self.cats_names[col] = cats_mapped.index
            self.cats_encoded[col] = cats_mapped.values
            self.encoding[col] = {self.cats_names[col][j] : self.cats_encoded[col][j] for j in range(len(cats_mapped))}        
        return self
    
    def transform(self, X):
        temp = X.copy()
        for col in self.cols:
            temp[col] = temp[col].map(self.encoding[col])
        return temp
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names_out(self, names = None):
        return self.feature_names


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """
    A transformer that performs Multi-Hot Encoding on specified column in a Pandas DataFrame.

    Input should be Pandas DataFrame with columns of lists of categories

    """
    def __init__(self, col):
        self.col = col
        
    def fit(self, X, y = None):
        classes__ = set()
        X_temp = X.copy()
        for row in X[self.col]:
            for cat in row:
                classes__.add(cat)
        classes__ = list(classes__)
        self.nclasses__ = len(classes__)
        self.classes__ = classes__
        return self
    
    def transform(self, X):
        results = np.zeros((X.shape[0], self.nclasses__))
        for i in range(self.nclasses__):
            results[:, i] = X[self.col].apply(lambda x: self.classes__[i] in x)
        return results
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names_out(self, names = None):
        return self.classes__


class NthRootTransformer(BaseEstimator, TransformerMixin):
    """

    A transformer that takes n-th root transformation on specified columns in a Pandas DataFrame.

        Parameters:
        -----------
        n : float, default=0.5
            Takes the n-th root
        random_state : int, default=42
            Seed for the random number generator.

        Returns:
        --------
        A new Pandas DataFrame with the specified columns transformed.

    """
    def __init__(self, n = 1/2, random_state = 42):
        self.random_state = random_state
        self.n = n
        
    def fit(self, X, y = None):
        self.feature_names = X.columns
        return self
    
    def transform(self, X):
        return np.power(X, self.n)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names_out(self, names = None):
        return self.feature_names
    

class LogTransformer(BaseEstimator, TransformerMixin):
    """

    A transformer that takes log transformation + x of specified columns in a Pandas DataFrame.

        Parameters:
        -----------
        x : float, default=0
            Takes the log of (col+x)
        random_state : int, default=42
            Seed for the random number generator.

        Returns:
        --------
        A new Pandas DataFrame with the specified columns transformed.

    """
    def __init__(self, x = 0, random_state = 42):
        self.random_state = random_state
        self.x = x
        
    def fit(self, X, y = None):
        self.feature_names = X.columns
        return self
    
    def transform(self, X):
        return np.log(X+x)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names_out(self, names = None):
        return self.feature_names
  

class WinsorizationImpute(BaseEstimator, TransformerMixin):
    """

    A transformer that performs winsorization imputation on specified columns in a Pandas DataFrame.

        Parameters:
        -----------
        p : float, default=0.05
            The percentile value representing the lower bound for winsorization.
        q : float, default=0.95
            The percentile value representing the upper bound for winsorization.
        random_state : int, default=42
            Seed for the random number generator used for imputing missing values.
        cols : list, default=None
            The list of names of columns to be winsorized.

        Returns:
        --------
        A new Pandas DataFrame with the specified columns winsorized.

    """


    def __init__(self, p=0.05, q=0.95, random_state=42, cols=None):
        self.p = p
        self.q = q
        self.random_state = random_state
        self.cols = cols
        
    def fit(self, X, y=None):
        self.feature_names = X.columns
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for col in self.cols:
            lower_bound = X[col].quantile(self.p)
            upper_bound = X[col].quantile(self.q)
            self.lower_bounds_[col] = lower_bound
            self.upper_bounds_[col] = upper_bound
        return self
    
    def transform(self, X):
        X_winsorized = X.copy()
        for col in self.cols:
            lower_bound = self.lower_bounds_[col]
            upper_bound = self.upper_bounds_[col]
            outliers_mask = (X_winsorized[col] < lower_bound) | (X_winsorized[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            if outliers_count > 0:
                random_values = np.random.normal(loc=X_winsorized[col].mean(),
                                                 scale=X_winsorized[col].std(),
                                                 size=outliers_count)
                random_values = np.clip(random_values, lower_bound, upper_bound)
                X_winsorized.loc[outliers_mask, col] = random_values
        return X_winsorized

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names_out(self, names = None):
        return self.feature_names
 