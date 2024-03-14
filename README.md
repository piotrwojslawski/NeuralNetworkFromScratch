# NeuralNetworkFromScratch
Two Hidden Layer Neural Network for Human Activity Recognition

1.	Introduction
In this project, we focus on developing 2 hidden layer Neural Network to perform recognition of human activity (downstairs motion, jogging motion, sitting motion, standing motion, upstairs motion, walking motion). The dataset contains 7352 training and 2947 test samples. In this project we won’t utilize the time series features of these signals. Instead, statistical features such as mean, standard deviation, variance, skewness, and kurtosis were extracted and represented as a 561-dimensional vector per sample in the dataset.
2.	Methodology
The methodology employed in this project involves constructing and training 2 hidden layer Neural Network to recognize human activity. The goal of this network is minimizing the error between the predicted and true labels.
a.	Preprocessing
During the preprocessing phase, the original training dataset was restructured into a new training set and a validation set. The validation set was formed using stratified sampling, ensuring it comprised 10% of each class from the original training dataset. This approach guarantees a representative distribution of all classes in the validation subset.
b.	Neural Network Architecture
This Neural Network consists of an input layer, two hidden layers, and an output layer. The input layer consists of 561 neurons, which represent statistical features such as mean, standard deviation, variance, skewness, and kurtosis based on time series data from sensors. The first hidden layer contains 300 neurons, while the second hidden layer contains 100 or 200 neuros depend on the configuration. The output layer consists of 6 neurons, each representing one of the six possible human activity classes. This neural network is fully connected, which means that the input is connected with first hidden layer, first hidden layer is connected with second hidden layer and second hidden layer is connected with output.
c.	Activation Functions
We employ two types of activation functions in our experiments:
ReLU (Rectified Linear Unit): For the first and second hidden layers, the ReLU function is used.
Softmax: The output layer uses the sigmoid function.
d.	Loss Function
In this model Cross-Entropy function is our loss function. For simplicity is was calculated as mean of Cross-Entropy for N samples.
e.	Weight Initialization
Weights and biases are initialized using a random uniform distribution within a small interval around zero [-0.01, 0.01]. 
3.	Forward propagation
In forward propagation, the input data is passed through the network to calculate the output. For this neural network, this process involves two hidden layers, each using the ReLU activation function and output layer with Softmax activation.

For the first hidden layer:
Linear transformation: self.v1 = np.dot(X, self.weights1.T) + self.bias1
Activation with ReLU:  self.h1 = RELU(self.v1)

For the second hidden layer:
Linear transformation: self.v2 = np.dot(self.h1, self.weights2.T) + self.bias2
If dropout_rate > 0 each dropped neuron j is settled to zero: 
self.v2[:, dropped_neurons] = 0

Then the entire expression of H2 is scaled to adjust for the dropout rate:
self.h2 = (1/(1 - dropout_rate)) * RELU(self.v2)

If dropout_rate = 0 the expression of H2 is just equal to:
Activation with ReLU:  self.h2 = RELU(self.v2)


For the output layer:
Linear transformation: self.v3 = np.dot(self.h2, self.weights3.T) + self.bias3
Activation with Softmax: self.y = softmax(self.v3)


4.	Backward propagation
Backward propagation involves computing gradients of the loss function with respect to the weights and biases, and then using these gradients to update the weights and biases. This is crucial for learning.
Base on the cross-entropy loss function and Softmax as activation function, we can simplify calculation as follows: 

a.	Gradient of Error with Respect to Output Layer
self.dE_dv3 = forward_output - y
 
b.	Gradients of Error with Respect to Weights and Biases of the Output Layer
self.dE_dw3 = np.dot(self.dE_dv3.T, self.h2)
self.dE_db3 = np.sum(self.dE_dv3, axis=0)
 
c.	Gradients for the Second Hidden Layer:
self.dE_dh2 = np.dot(self.dE_dv3, self.weights3)
 
self.dE_dw2 = np.dot((self.dE_dh2 * derivative_RELU(self.v2)).T, self.h1)
 
self.dE_db2 = np.sum(self.dE_dh2 * derivative_RELU(self.v2), axis=0)
 
d.	Gradients for the First Hidden Layer
self.dE_dh1 = np.dot(self.dE_dh2 * derivative_RELU(self.v2), self.weights2)
 
self.dE_dw1 = np.dot((self.dE_dh1 * derivative_RELU(self.v1)).T, X)
 
self.dE_db1 = np.sum(self.dE_dh1 * derivative_RELU(self.v1), axis=0)

e.	Weights and Biases Update:
self.weights1 -= self.learning_rate * self.dE_dw1 + alpha_momentum * self.momentum_weights1
self.weights2 -= self.learning_rate * self.dE_dw2 + alpha_momentum * self.momentum_weights2
self.weights3 -= self.learning_rate * self.dE_dw3 + alpha_momentum * self.momentum_weights3
 
self.bias1 -= self.learning_rate * self.dE_db1 + alpha_momentum * self.momentum_bias1
self.bias2 -= self.learning_rate * self.dE_db2 + alpha_momentum * self.momentum_bias2
self.bias3 -= self.learning_rate * self.dE_db3 + alpha_momentum * self.momentum_bias3
 
Momentum are updated based on the derivatives from the previous iteration.
