# Standard library imports
import os
import random
import traceback
from pdb import set_trace
import sys
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Callable
# Third-party library imports
from util.timer import Timer
from util.data import split_data, dataframe_to_array, binarize_classes
from util.metrics import accuracy
from util.metrics import mse
from sklearn.metrics import confusion_matrix
from util.metrics import nll, sse
from util.data import AddBias, Standardization, ImageNormalization, OneHotEncoding
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from scipy.stats import norm
from datasets.MNISTDataset import MNISTDataset
from datasets.HousingDataset import HousingDataset
from activations import *
from train import *
from data_processing import *


class Layer():
    """ Class which stores all variables required for a layer in a neural network
            W: array of weights 
            b: array of biases 
            g: Activation function for neurons in layer
            name: Name of the layer
            neurons: Number of neurons in the layer
            inputs: Number of inputs in the layer  
            Z: Linear combination of weights and inputs for all neurons.    
            A: Activation output for all neurons. Initialized to an empty array.
    """

    def __init__(
        self,
        W: np.array,
        b: np.array,
        g: object,
        name: str = ""
    ):
        self.W = W
        self.b = b
        self.g = g
        self.name = name
        self.neurons = len(W)
        self.inputs = W.shape[1]
        self.Z = np.array([])
        self.A = np.array([])

    def print_info(self) -> None:
        """ Prints info for all class attributes"""
        print(f"{self.name}")
        print(f"\tNeurons: {self.neurons}")
        print(f"\tInputs: {self.inputs}")
        print(f"\tWeight shape: {self.W.shape}")
        print(f"\tBias shape: {self.b.shape}")
        print(f"\tActivation function: {self.g.__name__}")
        print(f"\tZ shape: {self.Z.shape}")
        print(f"\tA shape: {self.A.shape}")


def get_mini_batches(data_len: int, batch_size: int = 32) -> List[np.ndarray]:
    X_idx = np.arange(data_len)
    np.random.shuffle(X_idx)
    batches = [X_idx[i:i+batch_size] for i in range(0, data_len, batch_size)]

    return batches


# Neural Network


class NeuralNetwork(BaseEstimator):
    def __init__(
        self,
        neurons_per_layer: List[int],
        learning_curve_loss: Callable,
        delta_loss_func: Callable,
        g_hidden: object,
        g_output: object,
        alpha: float = 0.01,
        epochs: int = 1,
        batch_size: int = 32,
        seed: int = None,
        verbose: bool = True,
        decay: bool = False,
    ):
        if seed is not None:
            np.random.seed(seed)
        self.neurons_per_layer = neurons_per_layer
        self.learning_curve_loss = learning_curve_loss
        self.delta_loss_func = delta_loss_func
        self.g_hidden = g_hidden
        self.g_output = g_output
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.decay = decay

        self.nn = []
        self.avg_trn_loss_tracker = []
        self.avg_vld_loss_tracker = []

    def init_neural_network(self, n_input_features: int):
        """Initializes the neural network's layers."""
        self.nn = []
        inputs = n_input_features
        for l, neurons in enumerate(self.neurons_per_layer):
            g = self.g_output if l == len(
                self.neurons_per_layer) - 1 else self.g_hidden
            name = f"Layer {l + 1}" + (" (Output)" if l ==
                                       len(self.neurons_per_layer) - 1 else " (Hidden)")
            W = self.init_weights(neurons, inputs)
            b = np.zeros((neurons, 1))
            self.nn.append(Layer(W, b, g, name))
            inputs = neurons  # Update inputs for the next layer

    def init_weights(self, neurons: int, inputs: int) -> np.ndarray:
        return np.random.randn(neurons, inputs) * np.sqrt(2. / inputs)

    def fit(self, X: np.ndarray, y: np.ndarray, X_vld: np.ndarray = None, y_vld: np.ndarray = None):
        """Fit the model to the training data."""
        self.init_neural_network(X.shape[1])
        for epoch in range(self.epochs):
            batch_loss_sum = 0
            batches = get_mini_batches(len(X), self.batch_size)
            for batch_idxs in batches:
                X_batch, y_batch = X[batch_idxs], y[batch_idxs]
                y_hat = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_hat)
                batch_loss_sum += self.learning_curve_loss(y_batch, y_hat)
            avg_trn_loss = batch_loss_sum / len(X)
            self.avg_trn_loss_tracker.append(avg_trn_loss)

            if self.verbose:
                print(
                    f"Epoch {epoch + 1}: Training Loss = {avg_trn_loss}", end='')

            if X_vld is not None and y_vld is not None:
                y_hat_vld = self.forward(X_vld)
                vld_loss = self.learning_curve_loss(y_vld, y_hat_vld)
                avg_vld_loss = vld_loss / len(y_vld)
                self.avg_vld_loss_tracker.append(avg_vld_loss)
                if self.verbose:
                    print(f", Validation Loss = {avg_vld_loss}")
            else:
                if self.verbose:
                    print()  # Newline for neatness if no validation loss is printed

    def forward(self, X: np.ndarray) -> np.ndarray:
        A = X.T  # Transpose input to match weight matrix orientation
        for layer in self.nn:
            # Calculate pre-activation (weighted input)
            layer.Z = np.dot(layer.W, A) + layer.b
            layer.A = layer.g.activation(layer.Z)  # Apply activation function
            A = layer.A  # Input for the next layer is the activation of the current layer
        return A.T  # Transpose output to match the original input orientation

    def backward(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray, lambda_: float = 0.01) -> None:
        m = X.shape[0]  # Number of examples

        # Start with the gradient of the loss function with respect to the output
        dZ = self.delta_loss_func(y.T, y_hat.T)

        for l in reversed(range(len(self.nn))):  # Iterate backwards through layers
            layer = self.nn[l]
            # Previous layer's activations
            A_prev = X.T if l == 0 else self.nn[l-1].A

            # Compute gradients
            dW = np.dot(dZ, A_prev.T) / m
            dB = np.sum(dZ, axis=1, keepdims=True) / m

            # Regularization term for weights (L2 regularization)
            if l > 0:  # Exclude regularization for input layer
                dW += (lambda_ / m) * layer.W

            # Update weights and biases
            layer.W -= self.alpha * dW
            layer.b -= self.alpha * dB

            if l > 0:  # No need to calculate these for the first layer (input)
                # Gradient of loss w.r.t. previous layer's activation
                dA_prev = np.dot(layer.W.T, dZ)
                # Apply derivative of activation function
                dZ = dA_prev * self.nn[l-1].g.derivative(self.nn[l-1].Z)


class NeuralNetworkRegressor(NeuralNetwork):
    def __init__(self, **kwargs):
        if 'neurons_per_layer' in kwargs:
            kwargs['neurons_per_layer'][-1] = 1

        super().__init__(**kwargs)

        self.__dict__.update(kwargs)

        # Maintain a list of parameter names for get_params and set_params methods
        self._param_names = list(kwargs.keys())

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        # Returns a dictionary of parameters, ensuring compatibility with scikit-learn's utilities.
        return {param: getattr(self, param) for param in self._param_names}

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        # Ensure the neural network reflects any updated parameters.
        self.init_neural_network(self.nn[0].inputs if self.nn else 0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the neural network model."""
        # Forward pass through the network to generate predictions
        y_hat = self.forward(X)
        # Ensure output shape is consistent with scikit-learn's expectations for regressors
        return y_hat.reshape(-1, 1)


#  Neural Network Classifier


class NeuralNetworkClassifier(NeuralNetwork):
    def __init__(self, **kwargs):
        if 'neurons_per_layer' in kwargs:
            kwargs['neurons_per_layer'][-1] = kwargs.get('output_neurons', 10)

        super().__init__(**kwargs)
        self.__dict__.update(kwargs)
        self._param_names = list(kwargs.keys())

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        # Retrieves a dictionary of all parameters set on this instance
        return {param: getattr(self, param) for param in self._param_names}

    def set_params(self, **parameters):
        """Set parameters for this estimator."""
        # Iteratively sets each provided parameter
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        # Reflect any updates in the initialized neural network
        self.init_neural_network(self.nn[0].inputs if self.nn else 0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for the test data X."""
        # Forward pass through the network to obtain class probabilities
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform classification on samples in X."""
        # Obtain class probabilities
        probabilities = self.predict_proba(X)
        # Determine the class with the highest probability for each sample
        y_hat = np.argmax(probabilities, axis=1)
        return y_hat.reshape(-1, 1)


# Define Hyperparameters


class HyperParametersAndTransforms():

    @staticmethod
    def get_params(name):
        model = getattr(HyperParametersAndTransforms, name)
        params = {}
        for key, value in model.__dict__.items():
            if not key.startswith('__') and not callable(key):
                if not callable(value) and not isinstance(value, staticmethod):
                    params[key] = value
        return params

    class NeuralNetworkRegressor():
        model_kwargs = dict(
            neurons_per_layer=[64, 1],
            learning_curve_loss=sse,
            delta_loss_func=delta_mse,
            g_hidden=ReLU,
            g_output=Linear,
            alpha=0.0035,
            epochs=250,
            batch_size=8,
            verbose=True,
            seed=42,
            decay=False,
        )
        data_prep_kwargs = dict(
            target_pipe=None,
            feature_pipe=Pipeline([('scaler', MinMaxScaler())]),
            use_features=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                          'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
        )

    class NeuralNetworkClassifier():
        model_kwargs = dict(
            neurons_per_layer=[10],
            learning_curve_loss=nll,
            delta_loss_func=delta_softmax_nll,
            g_hidden=ReLU,
            g_output=Softmax,
            alpha=0.0000225,
            epochs=45,
            batch_size=8,
            verbose=True,
            seed=42,
            decay=True,
        )
        data_prep_kwargs = dict(
            target_pipe=Pipeline([('OneHot', OneHotEncoder())]),
            feature_pipe=Pipeline([('scaler', MinMaxScaler())]),
        )
