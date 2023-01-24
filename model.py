import numpy as np
import time


class DeepNeuralNetwork():
    def __init__(self, sizes, activation='sigmoid'):
        # Class atributes
        self.sizes = sizes
        # Choose activation function
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            raise ValueError(
                "Activation function is currently not support, please use 'relu' or 'sigmoid' instead.")

        # Save all weights
        self.params = self.initialize()
        # Save all intermediate values, i.e. activations
        self.cache = {}

    def initialize(self):
        # Initialize and store neural network parameters

        # number of nodes in each layer
        input_layer_size = self.sizes[0]
        hidden_layer_size = self.sizes[1]
        output_layer_size = self.sizes[2]

        params = {
            "W1": np.random.randn(hidden_layer_size, input_layer_size) * np.sqrt(1./input_layer_size),
            "b1": np.zeros((hidden_layer_size, 1)) * np.sqrt(1./input_layer_size),
            "W2": np.random.randn(output_layer_size, hidden_layer_size) * np.sqrt(1./hidden_layer_size),
            "b2": np.zeros((output_layer_size, 1)) * np.sqrt(1./hidden_layer_size)
        }
        return params

    def optimizer_init(self):
        momemtum_opt = {
            "W1": np.zeros(self.params["W1"].shape),
            "b1": np.zeros(self.params["b1"].shape),
            "W2": np.zeros(self.params["W2"].shape),
            "b2": np.zeros(self.params["b2"].shape),
        }
        return momemtum_opt

    def feed_forward(self, x):
        '''
            y = σ(wX + b)
        '''
        self.cache["X"] = x

        # TODO: ako budeš implementirao multi layer neuralne mreže.
        # Ovdje trebaš da ubaciš petlju, koja će ići layer po layer i updateovati vrijednosti.

        # Input layer
        # Z1 = W1 x X.T + b1
        self.cache["Z1"] = np.matmul(
            self.params["W1"], self.cache["X"].T) + self.params["b1"]
        # A1 = activation_fn(Z1)
        self.cache["A1"] = self.activation(self.cache["Z1"])

        # Hidden layer
        # Z2 = W2 x A1 + b2
        self.cache["Z2"] = np.matmul(
            self.params["W2"], self.cache["A1"]) + self.params["b2"]
        # A2 = activation_fn(Z2)
        self.cache["A2"] = self.softmax(self.cache["Z2"])
        # Return values from the last layer, i.e, output layer
        return self.cache["A2"]

    def back_propagate(self, y, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is 
                  caused  by the dot and multiply operations on the huge arrays.

                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        current_batch_size = y.shape[0]

        dZ2 = output - y.T
        dW2 = (1./current_batch_size) * np.matmul(dZ2, self.cache["A1"].T)
        db2 = (1./current_batch_size) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(self.params["W2"].T, dZ2)
        dZ1 = dA1 * self.activation(self.cache["Z1"], derivative=True)
        dW1 = (1./current_batch_size) * np.matmul(dZ1, self.cache["X"])
        db1 = (1./current_batch_size) * np.sum(dZ1, axis=1, keepdims=True)

        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return self.grads

    def cross_entropy_loss(self, y, output):
        '''
            L(y, ŷ) = −∑ylog(ŷ).
        '''
        l_sum = np.sum(np.multiply(y.T, np.log(output)))
        m = y.shape[0]
        l = -(1./m) * l_sum
        return l

    def optimize(self, l_rate=0.1, beta=.9):
        '''
            Stochatic Gradient Descent (SGD):
            θ^(t+1) <- θ^t - η∇L(y, ŷ)

            Momentum:
            v^(t+1) <- βv^t + (1-β)∇L(y, ŷ)^t
            θ^(t+1) <- θ^t - ηv^(t+1)
        '''
        if self.optimizer == None or (self.optimizer != "sgd" and self.optimizer != "momentum"):
            raise ValueError("Optimizer not defined or not supported!") 
        

        if self.optimizer == "sgd":
            for key in self.params:
                self.params[key] = self.params[key] - l_rate * self.grads[key]
        
        elif self.optimizer == "momentum":
            for key in self.params:
                self.momemtum_opt[key] = (beta * self.momemtum_opt[key] + (1. - beta) * self.grads[key])
                self.params[key] = self.params[key] - l_rate * self.momemtum_opt[key]



    def train(self, x_train, y_train, x_test, y_test, epochs=10,
              batch_size=64, optimizer='momentum', l_rate=0.1, beta=.9):
        
        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        num_batches = x_train.shape[0] // self.batch_size

        # Initialize optimizer
        self.optimizer = optimizer
        if self.optimizer == 'momentum':
            self.momemtum_opt = self.optimizer_init()

        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}, test loss={:.2f}"
        # Train
        for i in range(self.epochs):
            # Shuffle
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for j in range(num_batches):
                # Batch
                start = j * self.batch_size
                end = min(start + self.batch_size, x_train.shape[0]-1)
                x = x_train_shuffled[start:end]
                y = y_train_shuffled[start:end]

                # Forward
                output = self.feed_forward(x)
                # Backprop
                _ = self.back_propagate(y, output)
                # Optimize
                self.optimize(l_rate=l_rate, beta=beta)

            # Evaluate performance
            # Training data
            output = self.feed_forward(x_train)
            train_acc = self.accuracy(y_train, output)
            train_loss = self.cross_entropy_loss(y_train, output)
            # Test data
            output = self.feed_forward(x_test)
            test_acc = self.accuracy(y_test, output)
            test_loss = self.cross_entropy_loss(y_test, output)
            print(template.format(i+1, time.time()-start_time,
                  train_acc, train_loss, test_acc, test_loss))

    def accuracy(self, y, output):
        '''
            What exactly does this function do
        '''
        return np.mean(np.argmax(y, axis=-1) == np.argmax(output.T, axis=-1))
    
    ''' -------------- ACTIVATION FUNCTIONS -------------- '''
    def relu(self, x, derivative=False):
        '''
            Derivative of ReLU is a bit more complicated since it is not differentiable at x = 0

            Forward path:
            relu(x) = max(0, x)
            In other word,
            relu(x) = 0, if x < 0
                    = x, if x >= 0

            Backward path:
            ∇relu(x) = 0, if x < 0
                     = 1, if x >=0
        '''
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        return np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        '''
            Forward path:
            σ(x) = 1 / 1+exp(-z)

            Backward path:
            ∇σ(x) = exp(-z) / (1+exp(-z))^2
        '''
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        '''
            softmax(x) = exp(x) / ∑exp(x)
        '''
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)
