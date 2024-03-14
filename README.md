# Two Hidden Layer Neural Network for Human Activity Recognition

## 1. Introduction
In this project, we focus on developing a neural network with two hidden layers to perform recognition of various human activities: downstairs motion, jogging motion, sitting motion, standing motion, upstairs motion, and walking motion. The dataset contains 7352 training and 2947 test samples. Rather than utilizing the time series features of these signals, statistical features such as mean, standard deviation, variance, skewness, and kurtosis were extracted and represented as a 561-dimensional vector per sample in the dataset.

## 2. Methodology
The methodology involves constructing and training a two-hidden-layer neural network to recognize human activity, aiming to minimize the error between the predicted and true labels.

### a. Preprocessing
During preprocessing, the original training dataset was restructured into a new training set and a validation set, with the latter formed using stratified sampling to ensure a representative distribution of all classes, comprising 10% of each class from the original training dataset.

### b. Neural Network Architecture
The neural network consists of an input layer with 561 neurons, two hidden layers (first with 300 neurons and second with either 100 or 200 neurons), and an output layer with 6 neurons. Each neuron in the output layer represents one of the six human activity classes. This fully connected network employs ReLU activation functions for the hidden layers and a Softmax function for the output layer.

### c. Activation Functions
- **ReLU (Rectified Linear Unit):** Used for the first and second hidden layers.
- **Softmax:** Employed at the output layer.

### d. Loss Function
The Cross-Entropy function, calculated as the mean of Cross-Entropy for N samples, serves as the loss function.

### e. Weight Initialization
Weights and biases are initialized using a random uniform distribution within a small interval around zero ([-0.01, 0.01]).

## 3. Forward Propagation
In forward propagation, input data passes through the network to calculate the output via two hidden layers and one output layer, using ReLU for hidden layers and Softmax for the output.

### For the first hidden layer:
- **Linear transformation:** `self.v1 = np.dot(X, self.weights1.T) + self.bias1`
- **Activation with ReLU:**  `self.h1 = RELU(self.v1)`

### For the second hidden layer:
- **Linear transformation:** `self.v2 = np.dot(self.h1, self.weights2.T) + self.bias2`
- **If dropout_rate > 0:** Each dropped neuron `j` is set to zero: `self.v2[:, dropped_neurons] = 0`, then scaled: `self.h2 = (1/(1 - dropout_rate)) * RELU(self.v2)`
- **If dropout_rate = 0:** `self.h2 = RELU(self.v2)`

### For the output layer:
- **Linear transformation:** `self.v3 = np.dot(self.h2, self.weights3.T) + self.bias3`
- **Activation with Softmax:** `self.y = softmax(self.v3)`

## 4. Backward Propagation
Backward propagation involves computing gradients of the loss function with respect to the weights and biases, and updating them accordingly.

### a. Gradient of Error with Respect to Output Layer
- `self.dE_dv3 = forward_output - y`
 
### b. Gradients of Error with Respect to Weights and Biases of the Output Layer
- `self.dE_dw3 = np.dot(self.dE_dv3.T, self.h2)`
- `self.dE_db3 = np.sum(self.dE_dv3, axis=0)`
 
### c. Gradients for the Second Hidden Layer
- `self.dE_dh2 = np.dot(self.dE_dv3, self.weights3)`
- `self.dE_dw2 = np.dot((self.dE_dh2 * derivative_RELU(self.v2)).T, self.h1)`
- `self.dE_db2 = np.sum(self.dE_dh2 * derivative_RELU(self.v2), axis=0)`
 
### d. Gradients for the First Hidden Layer
- `self.dE_dh1 = np.dot(self.dE_dh2 * derivative_RELU(self.v2), self.weights2)`
- `self.dE_dw1 = np.dot((self.dE_dh1 * derivative_RELU(self.v1)).T, X)`
- `self.dE_db1 = np.sum(self.dE_dh1 * derivative_RELU(self.v1), axis=0)`

### e. Weights and Biases Update
- Updates for weights and biases involve the learning rate and momentum:
- `self.weights1 -= self.learning_rate * self.dE_dw1 + alpha_momentum * self.momentum_weights1`
- `self.weights2 -= self.learning_rate * self.dE_dw2 + alpha_momentum * self.momentum_weights2`
- `self.weights3 -= self.learning_rate * self.dE_dw3 + alpha_momentum * self.momentum_weights3`

- `self.bias1 -= self.learning_rate * self.dE_db1 + alpha_momentum * self.momentum_bias1`
- `self.bias2 -= self.learning_rate * self.dE_db2 + alpha_momentum * self.momentum_bias2`
- `self.bias3 -= self.learning_rate * self.dE_db3 + alpha_momentum * self.momentum_bias3`

- Momentum is updated based on the derivatives from the previous iteration.
