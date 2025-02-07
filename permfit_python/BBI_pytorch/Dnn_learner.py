import numpy as np
from sklearn.base import BaseEstimator

class DNN_learner(BaseEstimator):
    """ToDo
    Parameters
    ----------
    Attributes
    ----------
    ToDO
    """

    def __init__(
        self,
        estimator_default,
    ):
        self.list_estimators = []
        self.estimator_default = estimator_default
        self.dim_repeat = 1

    def fit(self, X, y=None):
        """Build the DNN learner with the training set (X, y)
        Parameters
        ----------
        X : {pandas dataframe}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        dim_repeat = y.shape[-1]
        X_ = self.check_X_dim(X, dim_repeat)
        
        self.list_estimators = [None] * dim_repeat
        self.X_test = [None] * dim_repeat

        for y_col in range(dim_repeat):
            self.list_estimators[y_col] = self.estimator_default.clone()
            self.list_estimators[y_col].fit(X_[y_col, ...], y[:, [y_col]])

        return self
    
    def _predict(self, X, scale, proba=False):
        if isinstance(X, list):
            X_ = [self.check_X_dim(el, self.dim_repeat) for el in X]
        else:
            X_ = self.check_X_dim(X, self.dim_repeat)
        list_res = []
        for estimator_ind, estimator in enumerate(self.list_estimators):
            if isinstance(X_, list):
                curr_X = [el[estimator_ind, ...] for el in X_]
            else:
                curr_X = X_[estimator_ind, ...]
            if proba:
                list_res.append(estimator.predict_proba(curr_X, scale))
            else:
                list_res.append(estimator.predict(curr_X, scale))
            self.X_test[estimator_ind] = estimator.X_test.copy()
        return np.array(list_res)
        

    def predict(self, X, scale=True):
        """Predict regression target for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        scale: bool, default=True
            The continuous features will be standard scaled or not.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        return self._predict(X, scale, proba=False)


    def predict_proba(self, X, scale=True):
        """Predict class probabilities for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        scale: bool, default=True
            The continuous features will be standard scaled or not.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        return self._predict(X, scale, proba=True)


    def set_params(self, **kwargs):
        """Set the parameters of this estimator."""
        self.estimator_default.set_params(**kwargs)
        for key, value in kwargs.items():
            for estimator in self.list_estimators:
                setattr(estimator, key, value)

    def check_X_dim(self, X, dim_repeat):
        if (len(X.shape) != 3) or (X.shape[0] != dim_repeat):
            X_ = np.squeeze(X)
            X_ = np.array([X_ for i in range(dim_repeat)])
            self.dim_repeat = dim_repeat
        else:
            X_ = X
        return X_

    def encode_outcome(self, y, train=True):
        y_enc = []
        for y_col in range(y.shape[-1]):
            y_enc.append(self.list_estimators[y_col].encode_outcome(y[:, [y_col]], train=train))
        return np.concatenate(y_enc)