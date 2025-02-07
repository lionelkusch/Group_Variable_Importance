import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torchmetrics import Accuracy
from joblib import Parallel, delayed


def create_X_y(
    X,
    y,
    bootstrap=True,
    split_perc=0.8,
    prob_type="regression",
    list_cont=None,
    random_state=None,
):
    """Create train/valid split of input data X and target variable y.
    Parameters
    ----------
    X: {array-like, sparse matrix}, shape (n_samples, n_features)
        The input samples before the splitting process.
    y: ndarray, shape (n_samples, )
        The output samples before the splitting process.
    bootstrap: bool, default=True
        Application of bootstrap sampling for the training set.
    split_perc: float, default=0.8
        The training/validation cut for the provided data.
    prob_type: str, default='regression'
        A classification or a regression problem.
    list_cont: list, default=[]
        The list of continuous variables.
    random_state: int, default=2023
        Fixing the seeds of the random generator.

    Returns
    -------
    X_train_scaled: {array-like, sparse matrix}, shape (n_train_samples, n_features)
        The bootstrapped training input samples with scaled continuous variables.
    y_train_scaled: {array-like}, shape (n_train_samples, )
        The bootstrapped training output samples scaled if continous.
    X_valid_scaled: {array-like, sparse matrix}, shape (n_valid_samples, n_features)
        The validation input samples with scaled continuous variables.
    y_valid_scaled: {array-like}, shape (n_valid_samples, )
        The validation output samples scaled if continous.
    X_scaled: {array-like, sparse matrix}, shape (n_samples, n_features)
        The original input samples with scaled continuous variables.
    y_valid: {array-like}, shape (n_samples, )
        The original output samples with validation indices.
    scaler_x: scikit-learn StandardScaler
        The standard scaler encoder for the continuous variables of the input.
    scaler_y: scikit-learn StandardScaler
        The standard scaler encoder for the output if continuous.
    valid_ind: list
        The list of indices of the validation set.
    """
    rng = np.random.RandomState(random_state)
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    n = X.shape[0]

    if bootstrap:
        train_ind = rng.choice(n, n, replace=True)
    else:
        train_ind = rng.choice(n, size=int(np.floor(split_perc * n)), replace=False)
    valid_ind = np.array([ind for ind in range(n) if ind not in train_ind])

    X_train, X_valid = X[train_ind], X[valid_ind]
    y_train, y_valid = y[train_ind], y[valid_ind]

    # Scaling X and y
    X_train_scaled = X_train.copy()
    X_valid_scaled = X_valid.copy()
    X_scaled = X.copy()

    if len(list_cont) > 0:
        X_train_scaled[:, list_cont] = scaler_x.fit_transform(X_train[:, list_cont])
        X_valid_scaled[:, list_cont] = scaler_x.transform(X_valid[:, list_cont])
        X_scaled[:, list_cont] = scaler_x.transform(X[:, list_cont])
    if prob_type == "regression":
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_valid_scaled = scaler_y.transform(y_valid)
    else:
        y_train_scaled = y_train.copy()
        y_valid_scaled = y_valid.copy()

    return (
        X_train_scaled,
        y_train_scaled,
        X_valid_scaled,
        y_valid_scaled,
        X_scaled,
        y_valid,
        scaler_x,
        scaler_y,
        valid_ind,
    )

def relu(x):
    """The function applies the relu function element-wise to the input array x.
    https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
    """
    return np.maximum(x, 0, x)
    # return (abs(x) + x) / 2

def convert_predict_proba(list_probs):
    """If the classification is done using a one-hot encoded variable, the list of
    probabilites will be a list of lists for the probabilities of each of the categories.
    This function takes the probabilities of having each category (=1 with binary) and stack
    them into one ndarray.
    """
    if len(list_probs.shape) == 3:
        list_probs = np.array(list_probs)[..., 1].T
    return list_probs

class OrdinalEncode:
    """This function encodes the ordinal variable with a special gradual encoding storing also
    the natural order information.
    """
    def __init__(self):
        self.mapping_dict = None
        
    def fit(self, y):
        # Retrieve the unique values
        unique_vals = np.unique(y)
        # Mapping each unique value to its corresponding index
        mapping_dict = {}
        for i, val in enumerate(unique_vals):
            mapping_dict[val] = i + 1
        self.mapping_dict = mapping_dict
        return self

    def transform(self, y):
        # create a zero-filled array for the ordinal encoding
        y_ordinal = np.zeros((len(y), len(set(y))))
        # set the appropriate indices to 1 for each ordinal value and all lower ordinal values
        for ind_el, el in enumerate(y):
            y_ordinal[ind_el, np.arange(self.mapping_dict[el])] = 1
        return y_ordinal[:, 1:]
    
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)


def sample_predictions(predictions, random_state=None):
    """This function samples from the same leaf node of the input sample
    in both the regression and the classification case
    """
    rng = np.random.RandomState(random_state)
    # print(predictions[..., rng.randint(predictions.shape[2]), :])
    # print(predictions.shape)
    # exit(0)
    return predictions[..., rng.randint(predictions.shape[2]), :]


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        layer.weight.data = (layer.weight.data.uniform_() - 0.5) * 0.2
        layer.bias.data = (layer.bias.data.uniform_() - 0.5) * 0.1


def Dataset_Loader(X, y, shuffle=False, batch_size=50):
    if y.shape[-1] == 2:
        y = y[:, [1]]
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(y).float()
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    return loader


class DNN(nn.Module):
    """Feedfoward neural network with 4 hidden layers"""

    def __init__(self,
                 input_dim,
                 n_classes,
                 prob_type="regression",
                 n_epoch=20,
                 batch_size=32,
                 batch_size_val=128,
                 beta1=0.9,
                 beta2=0.999,
                 lr=1e-3,
                 l1_weight=1e-2,
                 l2_weight=1e-2,
                 epsilon=1e-3,
                 list_grps=None,
                 group_stacking=False,
                 random_state=2023,
                 verbose=0
                 ):
        super().__init__()        # Set the seed for PyTorch's random number generator
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon = epsilon
        if prob_type == "regression":
            self.loss_fun = F.mse_loss
            self.accuracy = None
        elif prob_type == "classification" or "ordinal":
            self.loss_fun = F.cross_entropy # Calculate loss
            self.accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        elif prob_type == "binary":
            self.loss_fun = F.binary_cross_entropy_with_logits
            self.accuracy = Accuracy(task="binary")
        else:
            raise ValueError('bad problem type')
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.n_epoch = n_epoch
        self.verbose = verbose
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.random_state = random_state

        # Specify whether to use GPU or CPU
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.random_state = random_state
        torch.manual_seed(self.random_state)

        # Set the seed for PyTorch's CUDA random number generator(s), if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
        self.list_grps = list_grps
        self.group_stacking = group_stacking
        if group_stacking:
            self.layers_stacking = nn.ModuleList(
                [
                    nn.Linear(
                        in_features=len(grp),
                        out_features=input_dim[grp_ind + 1] - input_dim[grp_ind],
                    )
                    for grp_ind, grp in enumerate(list_grps)
                ]
            )
            input_dim_single = input_dim[-1]
        else:
            input_dim_single = input_dim
        self.layers = nn.Sequential(
            # hidden layers
            nn.Linear(input_dim_single, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            # output layer
            nn.Linear(20, n_classes),
        )
        self.to(self.device)
        # Initializing weights/bias
        self.apply(init_weights)
        
        # save input for cloning 
        # this feature shouldn't be used #TODO fix it
        self.prob_type = prob_type
        self.input_dim = input_dim
        self.n_classes = n_classes
        self._name_attribute = ["lr", "beta1", "beta2", "epsilon", "prob_type", "l1_weight",
                                "l2_weight", "n_epoch", "verbose", "batch_size",
                                "batch_size_val", "random_state", "list_grps", "group_stacking",
                                "n_classes", "input_dim", 
                                ]
    
    def set_params(self, **kwargs):
        """Set the parameters of this estimator."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get_params(self):
        """Get the parameters of this estimator."""
        out = dict()
        for key in self._name_attribute:
            value = getattr(self, key)
            out[key] = value
        return out

    def clone(self):
        return type(self)(**self.get_params())

    def loss(self, X_valid, y_valid):
        validate_loader = Dataset_Loader(X_valid, y_valid, batch_size=self.batch_size_val)
        # Validation Phase
        batch_losses = [self.get_loss(batch) for batch in validate_loader]
        batch_sizes = [len(batch[0]) for batch in validate_loader]
        mean_loss = torch.stack(batch_losses).sum().item() / np.sum(batch_sizes)
        return mean_loss
        
    
    def fit_validate(self, X_train, y_train, X_valid, y_valid):
        """
        train_loader: DataLoader for Train data
        val_loader: DataLoader for Validation data
        original_loader: DataLoader for Original_data
        p: Number of variables
        n_epochs: The number of epochs
        lr: learning rate
        beta1: Beta1 parameter for Adam optimizer
        beta2: Beta2 parameter for Adam optimizer
        epsilon: Epsilon parameter for Adam optimizer
        l1_weight: L1 regalurization weight
        l2_weight: L2 regularization weight
        verbose: If > 2, the metrics will be printed
        prob_type: A classification or regression problem
        """
        # Creating DataLoaders
        train_loader = Dataset_Loader(
            X_train,
            y_train,
            shuffle=True,
            batch_size=self.batch_size,
        )
        validate_loader = Dataset_Loader(X_valid, y_valid, batch_size=self.batch_size_val)

        # Adam Optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=self.epsilon
        )
        best_loss = 1e100
        for epoch in range(self.n_epoch):
            # Training Phase
            self.train()
            for batch in train_loader:
                optimizer.zero_grad()
                loss = self.get_loss(batch)

                loss.backward()
                optimizer.step()
                for name, param in self.named_parameters():
                    if "bias" not in name:
                        param.data -= (
                            self.l1_weight * torch.sign(param.data) + self.l2_weight * param.data
                        )
            # Validation Phase
            self.eval()
            result = [self.validation_step(batch) for batch in validate_loader]
            loss_valid = 0
            total = 0
            for x in result:
                loss_valid += x["val_loss"] * x["batch_size"]
                total += x["batch_size"]
            loss_valid /= total # Combine losses
            if loss_valid < best_loss:
                best_loss = loss_valid
                best_state = copy.deepcopy(self.state_dict())
            if self.verbose >= 2:
                self.eval()
                outputs = [self.validation_step(batch) for batch in train_loader]
                self.epoch_end(epoch, self.validation_epoch_end(outputs))
        self.load_state_dict(best_state)
        return best_loss

    def forward(self, x):
        if self.group_stacking:
            list_stacking = [None] * len(self.layers_stacking)
            for ind_layer, layer in enumerate(self.layers_stacking):
                list_stacking[ind_layer] = layer(x[:, self.list_grps[ind_layer]])
            x = torch.cat(list_stacking, dim=1)
        return self.layers(x)

    def get_loss(self, batch, accuracy=False):
        X, y = batch[0].to(self.device), batch[1].to(self.device)
        y_pred = self(X)  # Generate predictions
        if accuracy:
            return self.loss_fun(y_pred, y), self.accuracy(y_pred, y.int())
        else: 
            return self.loss_fun(y_pred, y)

    def validation_step(self, batch):
        if self.accuracy is not None:
            loss_acc = self.get_loss(batch, accuracy=True)
            result = {
                "val_loss": float(loss_acc[0]),
                "val_acc": float(loss_acc[1]),
                "batch_size": len(batch[0]),
            }
        else:
            result = {
                "val_loss": float(self.get_loss(batch)),
                "batch_size": len(batch[0]),
            }
        return result

    def validation_epoch_end(self, outputs):
        total_element = np.sum([x["batch_size"] for x in outputs])
        batch_losses = np.sum([x["val_loss"] * x["batch_size"] for x in outputs])/ total_element
        result = {"val_loss": batch_losses}  # Combine losses
        if self.accuracy is not None:
            batch_accs = np.sum([x["val_acc"] * x["batch_size"] for x in outputs])/total_element
            result['val_acc'] = batch_accs
        return result

    def epoch_end(self, epoch, result):
        if self.accuracy is None:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch + 1, result["val_loss"]))
        else:
            print(
                "Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
                    epoch + 1, result["val_loss"], result["val_acc"]
                )
            )
    
    def score(self, X, y):
        self.eval()
        return self.get_loss([X, y])
        
    def _get_bais_weight(self):
        weight = []
        bias = []
        for name, param in self.state_dict().items():
            if name.split(".")[0] == "layers":
                if name.split(".")[-1] == "weight":
                    weight.append(param.numpy().T)
                if name.split(".")[-1] == "bias":
                    bias.append(param.numpy()[np.newaxis, :])
        return [weight, bias]    
    

    def _get_bais_weight_stack(self, list_grps):
        weight_stack = [[].copy() for _ in range(len(list_grps))]
        bias_stack = [[].copy() for _ in range(len(list_grps))]
        for name, param in self.state_dict().items():
            if name.split(".")[0] == "layers_stacking":
                curr_ind = int(name.split(".")[1])
                if name.split(".")[-1] == "weight":
                    weight_stack[curr_ind].append(param.numpy().T)
                if name.split(".")[-1] == "bias":
                    bias_stack[curr_ind].append(param.numpy()[np.newaxis, :])
        return [weight_stack, bias_stack]


    def get_prediction_group(self, X_scaled, inp_dim, list_grps):
        X_scaled_n = np.zeros((X_scaled.shape[0], inp_dim[-1]))
        weight_stack, bias_stack = self._get_bais_weight_stack(list_grps)
        for grp_ind in range(len(list_grps)):
            n_layer_stacking = len(weight_stack[grp_ind]) - 1
            curr_pred = X_scaled[:, list_grps[grp_ind]].copy()
            for ind_w_b in range(n_layer_stacking):
                if ind_w_b == 0:
                    curr_pred = relu(
                        X_scaled[:, list_grps[grp_ind]].dot(
                            weight_stack[grp_ind][ind_w_b]
                        )
                        + bias_stack[grp_ind][ind_w_b]
                    )
                else:
                    curr_pred = relu(
                        curr_pred.dot(weight_stack[grp_ind][ind_w_b])
                        + bias_stack[grp_ind][ind_w_b]
                    )
            X_scaled_n[
                :,
                list(np.arange(inp_dim[grp_ind], inp_dim[grp_ind + 1])),
            ] = (
                curr_pred.dot(weight_stack[grp_ind][n_layer_stacking])
                + bias_stack[grp_ind][n_layer_stacking]
            )
        return X_scaled_n
        
        
    def predict(self, X_scaled_n):
        #Todo improve it by using torch function
        weight, bias = self._get_bais_weight()
        n_layer = len(weight) - 1
        for j in range(n_layer):
            if j == 0:
                pred = relu(X_scaled_n.dot(weight[j]) + bias[j])
            else:
                pred = relu(pred.dot(weight[j]) + bias[j])

        pred = pred.dot(weight[n_layer]) + bias[n_layer]
        return pred


def compute_imp_std(pred_scores):
    weights = np.array([el.shape[-2] for el in pred_scores])
    # Compute the mean of each fold over the number of observations
    pred_mean = np.array(
        [np.mean(el.copy(), axis=-2) for el in pred_scores]
    )

    # Weighted average
    imp = np.average(
        pred_mean, axis=0, weights=weights
    )

    # Compute the standard deviation of each fold
    # over the number of observations
    pred_std = np.array(
        [
            np.mean(
                (el - imp[..., np.newaxis]) ** 2,
                axis=-2,
            )
            for el in pred_scores
        ]
    )
    std = np.sqrt(
        np.average(pred_std, axis=0, weights=weights)
        / (np.sum(weights) - 1)
    )
    return (imp, std)


def hyper_tuning(
    estimator,
    X,
    y,
    name_param,
    list_hyper,
    encode_outcome=lambda x: x,
    bootstrap=True,
    split_perc=0.8,
    prob_type = "regression",
    list_cont=None,
    random_state=None,
    n_jobs=None,
    n_ensemble=10,
    verbose=0,
):
    (
        X_train_scaled,
        y_train_scaled,
        X_valid_scaled,
        y_valid_scaled,
        X_scaled,
        __,
        scaler_x,
        scaler_y,
        ___,
    ) = create_X_y(
        X,
        y,
        bootstrap=bootstrap,
        split_perc=split_perc,
        prob_type=prob_type,
        list_cont=list_cont,
        random_state=random_state,
    )
    parallel = Parallel(
        n_jobs=min(n_jobs, n_ensemble), verbose=verbose
    )
    y_train_scaled = encode_outcome(y_train_scaled, train=True)
    y_valid_scaled = encode_outcome(y_valid_scaled)
    def fit_validate_estimator(X_train, y_train, X_test, y_test, value_param):
        estimator_x = estimator.clone()
        dict_param = {}
        for index, name in enumerate(name_param):
            dict_param[name] = value_param[index]
        estimator_x.set_params(**dict_param)
        return estimator_x.fit_validate(X_train, y_train, X_test, y_test)
    list_loss = [
        parallel(
                delayed(fit_validate_estimator)(
                    X_train_scaled,
                    y_train_scaled[i, ...],
                    X_valid_scaled,
                    y_valid_scaled[i, ...],
                    value_param = el
                )
                for el in list_hyper
            )
        for i in range(y_train_scaled.shape[0])
    ]
    ind_min = np.argmin(list_loss)
    best_hyper = list_hyper[ind_min]
    return best_hyper