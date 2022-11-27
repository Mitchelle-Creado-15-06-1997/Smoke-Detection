import numpy as np
from random import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import VarianceThreshold

class MLP(object):
    """A Multilayer Perceptron class.
    """

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=14):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs

        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """
        print("HIDDEN LAYERS: {} \n".format(hidden_layers))
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]
        print("LAYERS: {} \n".format(layers))

        # create random connection weights for the layers & save derivatives per layer
        weights = []
        derivatives = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
            
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
            
        self.weights = weights
        self.derivatives = derivatives
        
        print("WEIGHTS : {} \n".format(weights))
        print("DERIVATES : {} \n".format(derivatives))


        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
        print("ACTIVATIONS : {} \n".format(activations))

    def roc_auc_f1(self, mlp_clf):
        """
        roc_auc
        """
        
        mlp_clf.fit(Xtrain, ytrain)
        
        # calculate roc curve
        fpr_rf, tpr_rf, thresholds_rf = roc_curve(ytest, mlp_clf.predict_proba(Xtest)[:,1])
        print("FPR RF : {} \n".format(fpr_rf))
        print("TPR RF : {} \n".format(tpr_rf))
        print("THRESHOLD RF : {} \n".format(thresholds_rf))

        """
        Plot ROC 
        """

        skplt.metrics.plot_roc_curve(ytest, mlp_clf.predict_proba(Xtest))
        plt.show()

        """
        AUC
        """
        auc = roc_auc_score(ytest, mlp_clf.predict_proba(Xtest)[:,1])
        print('AUC: %.3f \n' % auc)

        """
        F1 score
        """

        #define array of actual classes
        actual = np.repeat([1, 0], repeats=[160, 240])

        #define array of predicted classes
        pred = np.repeat([1, 0, 1, 0], repeats=[120, 40, 70, 170])

        #calculate F1 score
        f1_score_val = f1_score(actual, pred)
        print("F1 SCORE : {} \n".format(f1_score_val))

        #calculate F1 score
        f1_score_val = f1_score(actual, pred)
        print("ACCURACY SCORE : {} \n".format(f1_score_val))

    def train_test_curve(self, mlp_clf):
        """
        Train test curve
        """

        train_sizes, train_scores, test_scores = learning_curve(estimator = mlp_clf,X = X, y = y_label, train_sizes = [1, 14, 17], cv = 5)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        #
        # Plot the learning curve
        #
        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        plt.title('Learning Curve')
        plt.xlabel('Training Data Size')
        plt.ylabel('Model accuracy')
        plt.grid()
        plt.legend(loc='lower right')
        plt.show()

    def model_ann(self, hidden):
        mlp_clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter = 1000,activation = 'logistic', solver = 'adam')
        return mlp_clf

    def bagging(self, mlp_clf):
        """
        bagging
        """
        bag_clf = BaggingClassifier(
        mlp_clf,
        n_estimators=13,
        max_samples=1.0,
        max_features=13,
        random_state=None,
        n_jobs=-1
        )
        bag_clf.fit(Xtrain,ytrain)
        y_prediction = bag_clf.predict(Xtest)
        
        return y_prediction

    def compute_error(self, y_true, y_pred):
        error_count = 0
        total_count = len(y_true)
        
        for i in range(total_count):
            if y_true[i] != y_pred[i]:
                error_count+=1
    
        return float(error_count)/float(total_count)

    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.

        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropogation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # save the activations for backpropogation
            self.activations[i + 1] = activations

        # return output layer activation
        return activations

    def back_propagate(self, error):
        """Backpropogates an error signal.
        Args:
            error (ndarray): The error to backprop.
        Returns:
            error (ndarray): The final error of the input
        """

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layer
            activations = self.activations[i+1]

            # apply sigmoid derivative function
            delta = error * self._sigmoid_derivative(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0],-1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)


    def train(self, inputs, targets, epochs, learning_rate):
        """Trains model running forward prop and backprop
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0

            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network!
                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                # now perform gradient descent on the derivatives
                # (this will update the weights
                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(Xtrain), i+1))

        print("=====***====Training complete!=====***====")


    def gradient_descent(self, learningRate=1):
        """Learns by descending the gradient
        Args:
            learningRate (float): How fast to learn.
        """
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate


    def _sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """

        y = 1.0 / (1 + np.exp(-x))
        return y


    def _sigmoid_derivative(self, x):
        """Sigmoid derivative function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """
        return x * (1.0 - x)


    def _mse(self, target, output):
        """Mean Squared Error loss function
        Args:
            target (ndarray): The ground trut
            output (ndarray): The predicted values
        Returns:
            (float): Output
        """
        return np.average((target - output) ** 2)


if __name__ == "__main__":
    smoke = pd.read_csv("./smoke.csv")
    smoke.head()
    smoke.drop(smoke.columns[[0, 1]], axis=1, inplace=True)

    smoke_copy = smoke.copy()
    X = smoke_copy.drop(columns=['Fire Alarm']).values
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    new_feature = sel.fit_transform(X)
    y_label = smoke_copy['Fire Alarm'].values 
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.2, random_state=2)

    sc = StandardScaler()
    sc.fit(Xtrain)
    Xtrain = sc.transform(Xtrain)
    Xtest = sc.transform(Xtest)

    print(f"Shape of train set is {Xtrain.shape} \n")
    print(f"Shape of test set is {Xtest.shape} \n")
    print(f"Shape of train label is {ytrain.shape} \n")
    print(f"Shape of test labels is {ytest.shape} \n")

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.90, random_state=2, shuffle=True)
    num_rows, num_cols = Xtrain.shape
    # create a Multilayer Perceptron with one hidden layer
    # mlp = ANN(num_rows, [2,2] , 1)

    post_score = 0
        
    find_hiddle_layers = [[20], [30], [40]]
    hiddle_layer = find_hiddle_layers[0]
    for i in range(len(find_hiddle_layers)):
        clf = MLPClassifier(hidden_layer_sizes=find_hiddle_layers[i], random_state=1, max_iter=300).fit(Xtrain, ytrain)
        score = clf.score(Xtrain, ytrain)
        if (post_score > score) :
            hiddle_layer = find_hiddle_layers[i]
        post_score = score
    
    mlp = MLP(num_cols, hiddle_layer, 1)
    
    # model
    model = mlp.model_ann(hiddle_layer)

    mlp.roc_auc_f1(model)

    mlp.train_test_curve(model)

    y_prediction = mlp.bagging(model)

    # Error after bagging
   
    testing_error = mlp.compute_error(ytest, y_prediction)
    print("Testing error after bagging {}".format(testing_error))

    output = mlp.forward_propagate(Xtest)

    print()
    print("=========Testing with inputs {} and output got is {}=========".format(Xtest, output))
    
