from myloads import *


class Layer:
    def __init__(self):
        pass

    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        num_units = input.shape[1]

        d_layer_d_input = np.eye(num_units)

        return np.dot(grad_output, d_layer_d_input)  # chain rule


class Dense(Layer):
    def __init__(self, input_units, output_units,  reg_param=0., **optim_method):
        # Optimization parameters depending on the method used
        if optim_method["name"] == "SGD":
            self.optim_method_name = "SGD"
            self.learning_rate = optim_method["learning_rate"]
        elif optim_method["name"] == "Adagrad":
            self.optim_method_name = "Adagrad"
            self.learning_rate = optim_method["learning_rate"]
            self.squared_grad_weights_accumulation = np.ones((input_units, output_units))
            self.squared_grad_biases_accumulation = np.ones((output_units))
            self.epsilon = 1e-8
        elif optim_method["name"] == "RMSProp":
            self.optim_method_name = "RMSProp"
            self.learning_rate = optim_method["learning_rate"]
            self.gamma = optim_method["gamma"]
            self.squared_grad_weights_accumulation = np.ones((input_units, output_units))
            self.squared_grad_biases_accumulation = np.ones((output_units))
            self.epsilon = 1e-8
        elif optim_method["name"] == "ADAM":
            self.optim_method_name = "ADAM"
            self.learning_rate = optim_method["learning_rate"]
            self.gamma_adaptative = optim_method["gamma_adaptative"]
            self.gamma_momentum = optim_method["gamma_momentum"]
            self.squared_grad_weights_accumulation = np.zeros((input_units, output_units))
            self.squared_grad_biases_accumulation = np.zeros((output_units))
            self.grad_weights_accumulation = np.zeros((input_units, output_units))
            self.grad_biases_accumulation = np.zeros((output_units))
            self.epsilon = 1e-8

        # Regularization parameter
        self.reg_param = reg_param

        # initialize biases at zeros
        self.biases = np.zeros(output_units)

        # initialize weights
        self.weights = np.random.randn(input_units, output_units) * 0.01

    def forward(self, input):
        return input.dot(self.weights) + self.biases

    def backward(self, input, grad_output):
        grad_input = grad_output.dot(self.weights.T)
        grad_reg = 2. * self.reg_param * self.weights;
        if self.optim_method_name == "SGD":
            grad_weights = input.T.dot(grad_output) + grad_reg
            grad_biases = np.sum(grad_output, axis=0)
            assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

            # Here perform a stochastic gradient descent step.
            self.weights = self.weights - self.learning_rate * grad_weights
            self.biases = self.biases - self.learning_rate * grad_biases

        elif self.optim_method_name == "Adagrad":
            # compute gradients for weights and biases parameters
            grad_weights = input.T.dot(grad_output) + grad_reg
            grad_biases = np.sum(grad_output, axis=0)

            # compute adaptative learning rates using gradients accumulation
            weights_learning_rate = self.learning_rate * 1. / np.sqrt(
                self.squared_grad_weights_accumulation + self.epsilon)
            biases_learning_rate = self.learning_rate * 1. / np.sqrt(
                self.squared_grad_biases_accumulation + self.epsilon)

            # perform a gradient descent step using adaptative learning rates
            self.weights = self.weights - np.multiply(weights_learning_rate, grad_weights)
            self.biases = self.biases - np.multiply(biases_learning_rate, grad_biases)

            # update of accumulations of squarred gradients
            self.squared_grad_weights_accumulation = self.squared_grad_weights_accumulation + grad_weights ** 2
            self.squared_grad_biases_accumulation = self.squared_grad_biases_accumulation + grad_biases ** 2

        elif self.optim_method_name == "RMSProp":
            # compute gradients for weights and biases parameters
            grad_weights = input.T.dot(grad_output) + grad_reg
            grad_biases = np.sum(grad_output, axis=0)

            # compute adaptative learning rates using gradients accumulation
            weights_learning_rate = self.learning_rate * 1. / np.sqrt(
                self.squared_grad_weights_accumulation + self.epsilon)
            biases_learning_rate = self.learning_rate * 1. / np.sqrt(
                self.squared_grad_biases_accumulation + self.epsilon)

            # perform a gradient descent step using adaptative learning rates
            self.weights = self.weights - np.multiply(weights_learning_rate, grad_weights)
            self.biases = self.biases - np.multiply(biases_learning_rate, grad_biases)

            # update of accumulations of squarred gradients using gamma (memory size parameter)
            self.squared_grad_weights_accumulation = self.gamma * self.squared_grad_weights_accumulation + (
                        1. - self.gamma) * grad_weights ** 2
            self.squared_grad_biases_accumulation = self.gamma * self.squared_grad_biases_accumulation + (
                        1. - self.gamma) * grad_biases ** 2

        elif self.optim_method_name == "ADAM":
            # compute gradients for weights and biases parameters
            grad_weights = input.T.dot(grad_output) + grad_reg
            grad_biases = np.sum(grad_output, axis=0)

            # compute unbiased moments of order 1 and 2
            weights_moment_1 = self.grad_weights_accumulation / (1. - self.gamma_momentum)
            weights_moment_2 = self.squared_grad_weights_accumulation / (1. - self.gamma_adaptative)
            biases_moment_1 = self.grad_biases_accumulation / (1. - self.gamma_momentum)
            biases_moment_2 = self.squared_grad_biases_accumulation / (1. - self.gamma_adaptative)

            # compute adaptative learning rates using gradients accumulation
            weights_learning_rate = self.learning_rate * 1. / (np.sqrt(weights_moment_2) + self.epsilon)
            biases_learning_rate = self.learning_rate * 1. / (np.sqrt(biases_moment_2) + self.epsilon)

            # perform a gradient descent step using adaptative learning rates and momentum
            self.weights = self.weights - np.multiply(weights_learning_rate, weights_moment_1)
            self.biases = self.biases - np.multiply(biases_learning_rate, biases_moment_1)

            # update of accumulations of gradients and squarred gradients using gammas (memory size parameters)
            self.grad_weights_accumulation = self.gamma_momentum * self.grad_weights_accumulation + (
                        1. - self.gamma_momentum) * grad_weights
            self.grad_biases_accumulation = self.gamma_momentum * self.grad_biases_accumulation + (
                        1. - self.gamma_momentum) * grad_biases
            self.squared_grad_weights_accumulation = self.gamma_adaptative * self.squared_grad_weights_accumulation + (
                        1. - self.gamma_adaptative) * grad_weights ** 2
            self.squared_grad_biases_accumulation = self.gamma_adaptative * self.squared_grad_biases_accumulation + (
                        1. - self.gamma_adaptative) * grad_biases ** 2

        return grad_input

class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        return np.maximum(input, 0)

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad
