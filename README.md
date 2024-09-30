# Custom Neural Network Implementation

A flexible and customizable neural network implementation in Python, built from scratch using NumPy and compatible with scikit-learn's estimator API. This project supports both regression and classification tasks, providing an excellent platform for learning about neural networks and integrating them into machine learning pipelines.

## Features

- Custom layer architecture with configurable weights, biases, and activation functions
- Support for both regression and classification tasks
- Compatible with scikit-learn's `BaseEstimator`, allowing seamless integration into pipelines
- Flexible activation functions for hidden and output layers
- Mini-batch gradient descent for efficient training
- L2 regularization support
- Data preprocessing pipelines using scikit-learn's `Pipeline`
- Support for MNIST and Housing datasets

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/custom-neural-network.git
   cd custom-neural-network
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install numpy scikit-learn scipy pandas wget
   ```

## Usage

### Neural Network Regressor

python
```
from Neural_Networks_Code import NeuralNetworkRegressor, HyperParametersAndTransforms
from sklearn.pipeline import Pipeline
Prepare your data
X_train, y_train, X_valid, y_valid = ... # Your data here
Get predefined hyperparameters
params = HyperParametersAndTransforms.NeuralNetworkRegressor.model_kwargs
Create and train the regressor
regressor = NeuralNetworkRegressor(params)
pipeline = Pipeline([
('scaler', HyperParametersAndTransforms.NeuralNetworkRegressor.data_prep_kwargs['feature_pipe']),
('regressor', regressor)
])
pipeline.fit(X_train, y_train, X_vld=X_valid, y_vld=y_valid)
Make predictions
predictions = pipeline.predict(X_test)
```

### Neural Network Classifier

```
from Neural_Networks_Code import NeuralNetworkClassifier, HyperParametersAndTransforms
from sklearn.pipeline import Pipeline
Prepare your data
X_train, y_train, X_valid, y_valid = ... # Your data here
Get predefined hyperparameters
params = HyperParametersAndTransforms.NeuralNetworkClassifier.model_kwargs
Create and train the classifier
classifier = NeuralNetworkClassifier(params)
pipeline = Pipeline([
('scaler', HyperParametersAndTransforms.NeuralNetworkClassifier.data_prep_kwargs['feature_pipe']),
('onehot', HyperParametersAndTransforms.NeuralNetworkClassifier.data_prep_kwargs['target_pipe']),
('classifier', classifier)
])
pipeline.fit(X_train, y_train, X_vld=X_valid, y_vld=y_valid)
Make predictions
predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)
```


### Project Structure

The main components of the project are:

1. `Neural_Networks_Code.py`: Contains the core neural network implementation, including the base `NeuralNetwork` class and its subclasses `NeuralNetworkRegressor` and `NeuralNetworkClassifier`.

2. `data_processing.py`: Includes classes for data preparation, specifically for the Housing and MNIST datasets.

3. `datasets/`: Directory containing dataset-specific classes for loading and preprocessing data.

4. `util/`: Directory with utility functions for metrics, data manipulation, and evaluation.

5. `activations.py`: Defines various activation functions used in the neural network.

6. `train.py`: Contains the `RunModel` class for training and evaluating models.

## Customization

You can customize the neural network architecture and hyperparameters by modifying the `HyperParametersAndTransforms` class in the `Neural_Networks_Code.py` file. This class provides predefined configurations for both regressor and classifier models.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
