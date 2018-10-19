
import numpy as np
import gameConst as gc
import json
import random
import sys

class CrossEntropy(object):
    @staticmethod
    def cost_func(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
    
    @staticmethod
    def delta(z, a, y):
        return a - y

class Quadratic(object):
    @staticmethod
    def cost_func(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)

# TODO Implement hyperparameter initialization and other optimization
class Network(object):

    """ initializes network with array of layers, where no_layers is number of layers """
    def __init__(self, sizes):
        self.sizes = sizes
        self.no_layers = len(sizes)
        self.weight_init()
        self.decision_made = []
        self.input = []
        self.cost = Quadratic()

    """ Initializes weights and biases. Biases are normally distributed with mean 0 and standard
        deviation 1. Weights are normally distributed with mean 0 and standard deviation 1 / sqrt(n)
        where n is the number of inputs. """
    def weight_init(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                         for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    """ Calculates the activation value for the input x and returns the output of the network. """
    def feedforward(self, x, gamma):
        
        if len(self.input) > 15000:
            del self.input[0]
        self.input.append(x)
        
        if np.random.uniform() < gamma:
            if np.random.uniform() < 0.3:
                vec_x = 1
            else:
                vec_x = 0
        else:
            for b, w in list(zip(self.biases, self.weights)):
                x = sigmoid(np.dot(w, x) + b)
            vec_x = np.mean(x)
            vec_x = np.rint(vec_x)
        
        if len(self.decision_made) > 15000:
            del self.decision_made[0]
        self.decision_made.append(vec_x)

        return vec_x

    def feedforward2(self, x):
        for b, w in list(zip(self.biases, self.weights)):
            x = sigmoid(np.dot(w, x) + b)
        result = np.rint(np.mean(x))
        return result

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbd = 0.0):
        """ Splits the data to training data and validation data such the SGD
            trains on 90% of data and tests on 10% of data """
        data_split = (int)(0.1 * len(training_data))
        validation_data = training_data[-data_split:]
        training_data = training_data[:-data_split]

        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
            for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_batch(mini_batch, eta, lmbd, len(training_data))

        """ Calculates and prints accuracy on training, validation data and cost of NN """
        accuracy = self.accuracy(training_data)
        val_accuracy = self.accuracy(validation_data)
        cost = self.total_cost(training_data, lmbd)
        print ("Accuracy on training data: {} / {}".format(accuracy, n))
        print ("Accuracy on validation data: {} / {}".format(val_accuracy, len(validation_data)))
        print ("Cost of the function: {}".format(cost))

    
    def backprog(self, inp, expec_out):
        diff_b = [np.zeros(b.shape) for b in self.biases]
        diff_w = [np.zeros(w.shape) for w in self.weights]
        activation = inp
        acitvations = [inp]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            acitvations.append(activation)
        # Calculate the error of our neural activation
        delta_L = (self.cost).delta(zs[-1], acitvations[-1], expec_out)
        diff_b[-1] = delta_L
        diff_w[-1] = np.dot(delta_L, acitvations[-2].transpose())
        for l in range(2, self.no_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta_L = np.dot(self.weights[-l+1].transpose(), delta_L) * sp
            diff_b[-l] = delta_L
            diff_w[-l] = np.dot(delta_L, acitvations[-l-1].transpose())
        return (diff_b, diff_w)
    

    def update_batch(self, mini_batch, eta, lmbd, n):
        diff_b = [np.zeros(b.shape) for b in self.biases]
        diff_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_diff_b, delta_diff_w = self.backprog(x, y)

            diff_b = (db + ddb for db, ddb in list(zip(diff_b, delta_diff_b)))
            diff_w = (dw + ddw for dw, ddw in list(zip(diff_w, delta_diff_w)))

        self.weights = [(1 - eta * lmbd / n) * w - (eta / len(mini_batch)) * dw
                            for w, dw in list(zip(self.weights, diff_w))]
        self.biases = [b - (eta / len(mini_batch)) * db 
                            for b, db in list(zip(self.biases, diff_b))]
        

    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


    def save_data(self, training_data):
        try:
            if len(training_data) > 15000:
                req_space = len(self.input)
                del training_data[:req_space]

            new_training_data = {"input": [t[0].tolist() for t in training_data],
                                 "output": [t[1] for t in training_data]}
            
            for i in self.input:
                new_training_data["input"].append(i.tolist())
            for o in self.decision_made:
                new_training_data["output"].append(o)
        except:
            new_training_data = {"input": training_data["input"],
                                 "output": training_data["output"]}

        f = open("trainingData.txt", "w")
        json.dump(new_training_data, f)
        f.close()


    def accuracy(self, data):
        results = [(self.feedforward2(x), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)


    def total_cost(self, data, lmda):
        cost = 0.0
        for x, y in data:
            a = self.feedforward2(x)
            cost += self.cost.cost_func(a, y) / len(data)
        cost += 0.5*(lmda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost    


""" Sigmoid function which makes our activation input continues and ensures that 
    it's smooth at the limits. """
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


""" Derivative of a sigmoid function. Instead of calculating it by hand,
    we use given identity """
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def load_data():
    f = open("trainingData.txt", "r")
    data = json.load(f)
    f.close()
    training_inputs = [np.reshape(x, (5, 1)) for x in data["input"]]
    training_results = data["output"]
    training_data = list(zip(training_inputs, training_results))
    return training_data


