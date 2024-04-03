import numpy as np


def sse(y_true, y_pred):
    """Sum of Squared Errors loss function."""
    return 0.5 * np.sum((y_true - y_pred) ** 2)


def delta_mse(y, y_hat):
    return y_hat - y


def delta_softmax_nll(y, y_hat):
    return y_hat - y


class Linear():
    @staticmethod
    def activation(z):
        return z

    @staticmethod
    def derivative(z):
        return np.ones_like(z)


class Sigmoid():
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def derivative(z):
        s = Sigmoid.activation(z)
        return s * (1 - s)


class Tanh():
    @staticmethod
    def activation(z):
        return np.tanh(z)

    @staticmethod
    def derivative(z):
        t = Tanh.activation(z)
        return 1 - t**2


class ReLU():
    @staticmethod
    def activation(z):
        return np.maximum(0, z)

    @staticmethod
    def derivative(z):
        return (z > 0).astype(float)


class Softmax():
    @staticmethod
    def activation(z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    @staticmethod
    def derivative(z):
        s = Softmax.activation(z)
        return s * (1 - s)
