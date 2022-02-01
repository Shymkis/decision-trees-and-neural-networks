from itertools import *
import math
import matplotlib.pyplot as plt
import numpy as np
import time

class Node:
    def __init__(self, node_id, layer_index):
        self.id = node_id
        self.layer_index = layer_index
        self.input = 0
        self.output = 0
        self.delta = 0


class Network:
    def __init__(self, layers, std):
        self.layers = []
        for layer_index, num_nodes in enumerate(layers):
            self.layers.append([])
            for node_id in range(num_nodes):
                node = Node(node_id, layer_index)
                self.layers[layer_index].append(node)
        self.weights = []
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.normal(0, std, (len(self.layers[i + 1]), len(self.layers[i]))))
        self.g = sigmoid
        self.g_prime = sigmoid_derivative

    def learn(self, examples, epochs=1, alpha=1):
        for _ in range(epochs):
            for example in examples:
                # Propagate the inputs forward to compute the outputs
                for node in self.layers[0]:
                    node.output = example["x"][node.id]
                k = 1
                for layer in self.layers[k:]:
                    for node_j in layer:
                        j = node_j.id
                        node_j.input = 0
                        for node_i in self.layers[k - 1]:
                            i = node_i.id
                            node_j.input += self.weights[k - 1][j][i]*node_i.output
                        node_j.output = self.g(node_j.input)
                    k += 1
                # Propagate deltas backward from output layer to input layer
                for node in self.layers[-1]:
                    node.delta = self.g_prime(node.input)*(example["y"][node.id] - node.output)
                k = len(self.layers) - 2
                for layer in self.layers[-2::-1]:
                    for node_i in layer:
                        i = node_i.id
                        sum_ = 0
                        for node_j in self.layers[k + 1]:
                            j = node_j.id
                            sum_ += self.weights[k][j][i]*node_j.delta
                        node_i.delta = self.g_prime(node_i.input)*sum_
                    k -= 1
                # Update every weight in network using deltas
                k = 0
                for layer in self.layers[:-1]:
                    for node_i in layer:
                        i = node_i.id
                        for node_j in self.layers[k + 1]:
                            j = node_j.id
                            self.weights[k][j][i] += alpha*node_i.output*node_j.delta
                    k += 1

    def feedforward(self, a):
        for w in self.weights:
            a = self.g(np.dot(w, a))
        return a

    def accuracy(self, examples):
        correct = 0
        for example in examples:
            pred_vec = self.feedforward(np.vstack(example["x"]))
            if np.argmax(pred_vec) == np.argmax(example["y"]):
                correct += 1
        return correct / len(examples) * 100

    def __str__(self):
        return str(self.weights)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1 - sigmoid(x))

def n_folds(N, examples):
    np.random.shuffle(examples)
    folds = []
    for x in range(N):
        train_exs = []
        test_exs = []
        for i, example in enumerate(examples):
            if i % N != x:
                train_exs.append(example)
            else:
                test_exs.append(example)
        folds.append({"train": train_exs, "test": test_exs})
    return folds

def main(title, examples, hidden_layers, num_folds=5, epochs=1, std=1, alpha=1):
    print(title, len(examples), hidden_layers, num_folds, epochs, std, alpha)

    start = time.time()

    layers = [len(examples[0]["x"])] + hidden_layers + [len(examples[0]["y"])]

    folds = n_folds(num_folds, examples)
    train_accuracies = test_accuracies = 0
    for fold in folds:
        network = Network(layers, std)
        network.learn(fold["train"], epochs, alpha)
        train_accuracies += network.accuracy(fold["train"])
        test_accuracies += network.accuracy(fold["test"])

    end = time.time()
    avg_time = round((end - start) / num_folds, 2)

    avg_train = round(train_accuracies / num_folds, 2)
    avg_test = round(test_accuracies / num_folds, 2)
    print("Average training set accuracy: " + str(avg_train) + "%")
    print("Average testing set accuracy: " + str(avg_test) + "%")

    print("Average time elapsed: " + str(avg_time))
    return avg_train, avg_test, avg_time

if __name__ == "__main__":
    proj1_examples = []
    train_file = open("train-file", "r")
    lines = train_file.readlines()
    for line in lines:
        line = line[:-1]
        values = line.replace("\"", "").replace(",", " ").split(" ")
        y = np.zeros(2)
        y[int(values[0])] = 1
        proj1_examples.append({
            "x": np.array([float(values[1]), float(values[2])]),
            "y": y
        })

    zoo_examples = []
    train_file = open("zoo.data", "r")
    lines = train_file.readlines()
    for line in lines:
        line = line[:-1]
        values = line.replace("\"", "").replace(",", " ").split(" ")
        x = []
        for i in range(1, len(values) - 1):
            if i == 13:
                x.append(float(values[i])/8)
            else:
                x.append(float(values[i]))
        y = np.zeros(7)
        y[int(values[-1]) - 1] = 1
        zoo_examples.append({
            "x": np.array(x),
            "y": y
        })

    letter_examples = []
    train_file = open("letter-recognition.data", "r")
    lines = train_file.readlines()
    for line in lines:
        line = line[:-1]
        values = line.replace("\"", "").replace(",", " ").split(" ")
        x = []
        for i in range(1, len(values)):
            x.append(float(values[i])/15)
        y = np.zeros(26)
        y[ord(values[0]) - 65] = 1
        letter_examples.append({
            "x": np.array(x),
            "y": y
        })

    # main("Proj1", proj1_examples, [20, 20], epochs=50)
    # main("Zoo", zoo_examples, [20, 20], epochs=50)
    # main("Letters", letter_examples, [20, 20], epochs=5)
    # main("Letters", letter_examples, [19], std=.5, alpha=1.75)

    # y1 = []
    # y2 = []
    # y3 = []
    # x = []
    # for i in range(5, 26, 5):
    #     avg_train, avg_test, avg_time = main("Proj1", proj1_examples, [i, i], epochs=20)
    #     y1.append(avg_train)
    #     y2.append(avg_test)
    #     y3.append(avg_time)
    #     x.append(i)
    # plt.plot(x, y1, label="Average Train Accuracy")
    # plt.plot(x, y2, label="Average Test Accuracy")
    # # plt.plot(x, y3, label="Average Time Elapsed")
    # plt.title("Accuracy vs. Number of Nodes on Project 1 Data, Two Layers")
    # plt.xlabel("Number of Nodes in Both Layers")
    # plt.ylabel("Accuracy (%)")
    # # plt.ylabel("Time Elapsed (s)")
    # plt.legend()
    # plt.show()
