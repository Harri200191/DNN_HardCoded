# Deep Neural Network for MNIST Binary Classification

This project aims to build a deep neural network from scratch for binary classification of images from the MNIST dataset. The MNIST dataset consists of grayscale images of handwritten digits, where each image represents a digit from 0 or 1. In this project, we will focus specifically on classifying the digits 1 and 0.

## Dataset

The MNIST dataset is a widely used benchmark dataset in the field of machine learning. It contains 60,000 training images and 10,000 test images, each of size 28x28 pixels. For this project, we will pre-process the dataset to extract the images representing the digits 1 and 0 only.

## Architecture

The deep neural network implemented in this project is built from scratch using Python and the NumPy library. It consists of the following components:

1. **Input Layer**: The input layer has 784 units, corresponding to the flattened size of each image (28x28 pixels).

2. **Hidden Layers**: The network can have multiple hidden layers, each consisting of a variable number of units. The number of hidden layers and units per layer can be customized based on the requirements.

3. **Activation Function**: A ReLU (Rectified Linear Unit) activation function is used for all hidden layers, except for the output layer.

4. **Output Layer**: The output layer consists of a single unit, representing the binary classification decision (0 or 1). The activation function used in the output layer is the sigmoid function, which maps the output to a probability between 0 and 1.

5. **Loss Function**: The binary cross-entropy loss function is used to measure the difference between predicted and actual labels.

6. **Optimization Algorithm**: Stochastic Gradient Descent (SGD) is utilized to optimize the weights and biases of the neural network.

## Implementation

The implementation of the deep neural network is provided in the `neural_network.py` file. The code is structured into classes and functions to ensure modularity and reusability.

To use the neural network for binary classification on the MNIST dataset, follow these steps:

1. Ensure you have Python 3.x installed on your system.

2. Clone this repository and navigate to the project directory.

   ```bash
   git clone https://github.com/Haris200191/DNN_HardCoded.git
   cd DNN_HardCoded
   ```

3. Install the required dependencies 

4. Once training is complete, the model will be evaluated on the test set, and the accuracy and loss will be displayed.

Feel free to explore the code and experiment with different configurations to improve the model's performance.

## Results

The performance of the model trained on the MNIST dataset for binary classification can vary depending on the network architecture and hyperparameters chosen. It is recommended to experiment with different configurations to achieve the desired accuracy.

## Conclusion

This project demonstrates the process of building a deep neural network from scratch for binary classification of images from the MNIST dataset. By following the steps outlined above, you can train and evaluate the model on the digits 1 and 0. This project provides a solid foundation for understanding the inner workings of deep neural networks and can serve as a starting point for more complex image classification tasks.
