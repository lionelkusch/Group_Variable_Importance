import itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from scipy.special import expit as sigmoid
from scipy.special import softmax

from .utils import (
    DNN,
    create_X_y,
    OrdinalEncode,
    hyper_tuning
)


class BaggingDNN(BaseEstimator):
    """
    Parameters
    ----------
    encode: bool, default=False
        Encoding the categorical outcome.
    do_hyper: bool, default=True
        Tuning the hyperparameters of the provided estimator.
    dict_hyper: dict, default=None
        The dictionary of hyperparameters to tune.
    n_ensemble: int, default=10
        The number of sub-DNN models to fit to the data
    min_keep: int, default=10
        The minimal number of DNNs to be kept
    batch_size: int, default=32
        The number of samples per batch for training
    batch_size_val: int, default=128
        The number of samples per batch for validation
    n_epoch: int, default=200
        The number of epochs for the DNN learner(s)
    verbose: int, default=0
        If verbose > 0, the fitted iterations will be printed
    bootstrap: bool, default=True
        Application of bootstrap sampling for the training set
    split_perc: float, default=0.8
        The training/validation cut for the provided data
    prob_type: str, default='regression'
        A classification or a regression problem
    list_grps: list of lists, default=None
        A list collecting the indices of the groups' variables
        while applying the stacking method.
    list_cont: list, default=None
        The list of continuous variables
    beta1: float, default=0.9
        The exponential decay rate for the first moment estimates.
    beta2: float, default=0.999
        The exponential decay rate for the second moment estimates.
    lr: float, default=1e-3
        The learning rate
    epsilon: float, default=1e-8
        A small constant added to the denominator to prevent division by zero.
    l1_weight: float, default=1e-2
        The L1-regularization paramter for weight decay.
    l2_weight: float, default=0
        The L2-regularization paramter for weight decay.
    n_jobs: int, default=1
        The number of workers for parallel processing.
    group_stacking: bool, default=False
        Apply the stacking-based method for the provided groups.
    inp_dim: list, default=None
        The cumsum of inputs after the linear sub-layers.
    random_state: int, default=2023
        Fixing the seeds of the random generator
    """

    def __init__(
        self,
        encode=False,
        do_hyper=False,
        dict_hyper=None,
        n_ensemble=10,
        min_keep=10,
        estimator_default=None,
        verbose=0,
        bootstrap=True,
        split_perc=0.8,
        prob_type="regression",
        list_cont=None,
        list_grps=None,
        n_jobs=1,
        group_stacking=False,
        inp_dim=None,
        random_state=2023,
    ):
        self.encode = encode
        self.do_hyper = do_hyper        # Initializing the dictionary for tuning the hyperparameters

        self.n_ensemble = n_ensemble
        self.min_keep = min_keep
        self.verbose = verbose
        self.bootstrap = bootstrap
        self.split_perc = split_perc
        self.prob_type = prob_type
        if prob_type in ("regression", "ordinal"):
            self.loss_func = lambda y,pred: np.std(y) ** 2 - mean_squared_error(y, pred)
        else:
            self.loss_func = lambda y,pred: log_loss(y, np.ones(y.shape) * np.mean(y, axis=0)) - log_loss(y, pred)
        self.list_grps = list_grps
        self.list_cont = list_cont
        self.n_jobs = n_jobs
        self.group_stacking = group_stacking
        self.inp_dim = inp_dim
        self.random_state = random_state
        self.enc_y = []
        self.link_func = {
            "classification": softmax,
            "ordinal": sigmoid,
            "binary": sigmoid,
        }
        self.is_encoded = False
        if estimator_default is None:
            self.estimator_default = DNN(
                inp_dim,
                1,
                prob_type=prob_type,
                list_grps=list_grps,
                group_stacking=group_stacking,
                random_state=random_state,
                verbose=verbose)
        else:
            self.estimator_default = estimator_default
        if dict_hyper is None and type(self.estimator_default) is DNN:
            self.dict_hyper = {
                "lr": [1e-2, 1e-3, 1e-4],
                "l1_weight": [0, 1e-2, 1e-4],
                "l2_weight": [0, 1e-2, 1e-4],
            }
        else:
            self.dict_hyper = dict_hyper

    def fit(self, X, y):
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
        if self.n_ensemble == 1:
            raise ValueError('Not taking in consideration value of n_ensemble < 1')
        # Disabling the encoding parameter with the regression case
        if self.prob_type == "regression":
            if len(y.shape) != 2:
                y = y.reshape(-1, 1)
            self.encode = False

        if self.encode:
            y = self.encode_outcome(y)
            self.is_encoded = True
            y = np.squeeze(y, axis=0)

        # Switch to the special binary case
        if (self.prob_type == "classification") and (y.shape[-1] < 3):
            self.prob_type = "binary"
        n, p = X.shape
        self.min_keep = max(min(self.min_keep, self.n_ensemble), 1)
        rng = np.random.RandomState(self.random_state)
        list_seeds = rng.randint(1e5, size=(self.n_ensemble))

        # Initialize the list of continuous variables
        if self.list_cont is None:
            self.list_cont = list(np.arange(p))

        # Initialize the list of groups
        if self.list_grps is None:
            if not self.group_stacking:
                self.list_grps = []

        # Convert the matrix of predictors to numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Hyperparameter tuning
        if self.do_hyper:
            list_hyper = list(itertools.product(*list(self.dict_hyper.values())))
            name_param = list(self.dict_hyper.keys())
            best_hyper = hyper_tuning(self.estimator_default, X, y, name_param, list_hyper,
                                    encode_outcome=self.encode_outcome)
            self.estimator_default.set_params(**best_hyper)

        parallel = Parallel(
            n_jobs=min(self.n_jobs, self.n_ensemble), verbose=self.verbose
        )
        res_ens = list(
            zip(
                *parallel(
                    delayed(joblib_ensemble_dnnet)(
                        X,
                        y,
                        self.estimator_default,
                        self.loss_func,
                        prob_type=self.prob_type,
                        link_func=self.link_func,
                        list_cont=self.list_cont,
                        list_grps=self.list_grps,
                        bootstrap=self.bootstrap,
                        split_perc=self.split_perc,
                        group_stacking=self.group_stacking,
                        inp_dim=self.inp_dim,
                        random_state=list_seeds[i],
                    )
                    for i in range(self.n_ensemble)
                )
            )
        )
        pred_m = np.array(res_ens[3])
        loss = np.array(res_ens[4])

        # Keeping the optimal subset of DNNs
        sorted_loss = loss.copy()
        sorted_loss.sort()
        new_loss = np.empty(self.n_ensemble - 1)
        for i in range(self.n_ensemble - 1):
            current_pred = np.mean(pred_m[loss >= sorted_loss[i], :], axis=0)
            if self.prob_type == "regression":
                new_loss[i] = mean_squared_error(y, current_pred)
            else:
                new_loss[i] = log_loss(y, current_pred)
        keep_dnn = (
            loss
            >= sorted_loss[np.argmin(new_loss[: (self.n_ensemble - self.min_keep + 1)])]
        )

        self.optimal_list = [
            (res_ens[0][i], (res_ens[1][i], res_ens[2][i]))
            for i in range(self.n_ensemble)
            if keep_dnn[i]
        ]
        self.is_fitted = True
        return self

    def encode_outcome(self, y, train=False):
        list_y = []
        if len(y.shape) != 2:
            y = y.reshape(-1, 1)
        if self.prob_type == "regression":
            list_y.append(y)

        for col in range(y.shape[1]):
            if train:
                # Encoding the target with the classification case
                if self.prob_type in ("classification", "binary", "ordinal"):
                    if self.prob_type == "ordinal":
                        self.enc_y.append(OrdinalEncode())
                    else:
                        self.enc_y.append(OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                    curr_y = self.enc_y[col].fit_transform(y[:, [col]])
                    list_y.append(curr_y)
            else:
                # Encoding the target with the classification case 
                if self.prob_type in ("classification", "binary", "ordinal"):
                    curr_y = self.enc_y[col].transform(y[:, [col]])
                    list_y.append(curr_y)
                
        return np.array(list_y)

    def _predict(self, X, scale=True):
        # Process the common prediction part
        pred_list = self.__pred_common(X, scale=scale)

        res_pred = np.zeros((pred_list[0].shape))
        total_n_elements = 0
        for ind_mod, pred in enumerate(pred_list):
            if self.prob_type == "regression":
                res_pred += (
                    pred * self.optimal_list[ind_mod][1][1].scale_
                    + self.optimal_list[ind_mod][1][1].mean_
                )
            else:
                res_pred += self.link_func[self.prob_type](pred)
            total_n_elements += 1
        res_pred = res_pred.copy() / total_n_elements
        if self.prob_type == "binary":
            res_pred = np.array([[1-res_pred[i][0], res_pred[i][0]] for i in range(res_pred.shape[0])])
        
        return res_pred
    
    def score(self, X, y, scale=True):
        res_pred = self._predict(X, scale=scale)
        loss = self.loss_func(y, res_pred)
        return loss

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
        if self.prob_type != "regression":
            raise Exception("Use the predict_proba function for classification")
        return self._predict(X, scale=scale)


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
        if self.prob_type == "regression":
            raise Exception("Use the predict function for classification")
        return self._predict(X, scale=scale)


    def __scale_test(self, X):
        """This function prepares the input for the DNN estimator either in the default
        case or after applying the stacking method
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        """
        # Check is fit had been called
        check_is_fitted(self, ["is_fitted"])

        if isinstance(X, pd.DataFrame):
            X_ = np.array(X)
        else:
            X_ = X

        # The input will be either the original input or the result
        # of the provided sub-linear layers in a stacking way for the different groups
        # In the stacking method, each sub-linear layer will have a corresponding output
        if self.group_stacking:
            X_test_n = [None] * len(self.optimal_list)
            for mod in range(len(self.optimal_list)):
                X_test_scaled = X_.copy()
                if len(self.list_cont) > 0:
                    X_test_scaled[:, self.list_cont] = self.optimal_list[mod][1][
                        0
                    ].transform(X_test_scaled[:, self.list_cont])
                X_test_n[mod] = self.optimal_list[mod][0].get_prediction_group(X_test_scaled, self.inp_dim, self.list_grps)
        else:
            X_test_n = [X_.copy()]
        self.X_test = X_test_n.copy()
        return X_test_n

    def __pred_common(self, X, scale=True):
        """
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        """
        # Prepare the test set for the prediction
        if scale:
            X_ = self.__scale_test(X)
        else:
            X_ = X
        if not self.group_stacking:
            X_ = [X_[0].copy() for i in range(self.n_ensemble)]

        pred = [None] * len(self.optimal_list)
        for ind_mod, mod in enumerate(self.optimal_list):
            X_test_scaled = X_[ind_mod].copy()
            pred[ind_mod] = self.optimal_list[ind_mod][0].predict(X_test_scaled)
        return pred

    def set_params(self, **kwargs):
        """Set the parameters of this estimator."""
        # TODO considere also the case where is already fitted
        for key, value in kwargs.items():
            assert hasattr(self, key) or hasattr(self.estimator_default, key)
            if hasattr(self, key):
                setattr(self, key, value)
            if hasattr(self.estimator_default, key):
                setattr(self.estimator_default, key, value)

    def clone(self):
        dict_param = self.get_params()
        for key in np.copy(list(dict_param.keys())):
            if 'estimator_default' in key:
                del dict_param[key]
        dict_param['estimator_default'] = self.estimator_default.clone()
        return type(self)(** dict_param)


def joblib_ensemble_dnnet(
    X,
    y,
    estimator,
    loss_fun,
    prob_type="regression",
    link_func=None,
    list_cont=None,
    list_grps=None,
    bootstrap=False,
    split_perc=0.8,
    group_stacking=False,
    inp_dim=None,
    random_state=None,
):
    """
    Parameters
    ----------
    X : {array-like, sparse-matrix}, shape (n_samples, n_features)
        The input samples.
    y : {array-like}, shape (n_samples,)
        The output samples.
    random_state: int, default=None
        Fixing the seeds of the random generator
    """

    pred_v = np.empty(X.shape[0])
    # Sampling and Train/Validate splitting
    (
        X_train_scaled,
        y_train_scaled,
        X_valid_scaled,
        y_valid_scaled,
        X_scaled,
        y_valid,
        scaler_x,
        scaler_y,
        valid_ind,
    ) = create_X_y(
        X,
        y,
        bootstrap=bootstrap,
        split_perc=split_perc,
        prob_type=prob_type,
        list_cont=list_cont,
        random_state=random_state,
    )

    param_estimator = estimator.get_params()
    param_estimator.update({'prob_type':prob_type,
                            'list_grps':list_grps,
                            'group_stacking':group_stacking,
                            'random_state':random_state,
    })
    current_model = type(estimator)(**param_estimator)
    current_model.fit_validate(X_train_scaled, y_train_scaled,
                               X_valid_scaled, y_valid_scaled
                               )
    if not group_stacking:
        X_scaled_n = X_scaled.copy()
    else:
        X_scaled_n = current_model.get_prediction_group(X_scaled, inp_dim, list_grps)

    pred = current_model.predict(X_scaled_n)

    if prob_type == "regression":
        pred_v = pred * scaler_y.scale_ + scaler_y.mean_
    else: # "classification", "binary",  "ordinal"
        pred_v = link_func[prob_type](pred)
    
    loss = loss_fun(y_valid, pred_v[valid_ind]) 
    return (current_model, scaler_x, scaler_y, pred_v, loss)
    