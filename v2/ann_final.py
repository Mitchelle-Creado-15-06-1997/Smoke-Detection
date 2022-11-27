import numpy as np
from random import random, randrange
import random
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
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

class ANN(object):
    """
    Artificial Neural Networks
    """

    def __init__(self, number_of_inputs=4, hidden_layer=[4, 4], number_of_outputs=14):
        """
            We construct the ANN by takes Inputs, Hidden layers (variable number) and output

        Args:
            number_of_inputs (int): Number of inputs
            hidden_layer (list): A list of ints for the hidden layers
            number_of_outputs (int): Number of outputs
        """
        print("HIDDEN LAYERS: {} \n".format(hidden_layer))
        self.number_of_inputs = number_of_inputs
        self.hidden_layer = hidden_layer
        self.number_of_outputs = number_of_outputs

        # Calculate the sum of all the layers
        cum_layers = [number_of_inputs] + hidden_layer + [number_of_outputs]
        print("LAYERS: {} \n".format(cum_layers))

        # Generate the weights for the layers
        # Calculate the derivative of the layer
        weights = []
        derivatives = []
        for i in range(len(cum_layers) - 1):
            w = np.random.rand(cum_layers[i], cum_layers[i + 1])
            weights.append(w)

            d = np.zeros((cum_layers[i], cum_layers[i + 1]))
            derivatives.append(d)

        self.weights = weights
        self.derivatives = derivatives
        print("WEIGHTS : {} \n".format(weights))
        print("DERIVATES : {} \n".format(derivatives))
        

        # Get the activation of the layers
        activations = []
        for i in range(len(cum_layers)):
            a = np.zeros(cum_layers[i])
            activations.append(a)
        self.activations = activations
        print("ACTIVATIONS : {} \n".format(activations))

    def get_feature_importance(self, j, n, mlp_clf):
        mlp_clf.fit(Xtest, ytest)
        s = accuracy_score(ytest, mlp_clf.predict_proba(Xtest)[:,1].round()) # baseline score
        total = 0.0
        for i in range(n):
            perm = np.random.permutation(range(Xtest.shape[0]))
            X_test_ = Xtest.copy()
            X_test_[:, j] = Xtest[perm, j]
            y_pred_ = clf.predict(X_test_)
            s_ij = accuracy_score(ytest, y_pred_)
            total += s_ij
        return s - total / n

    def roc_auc_cross_validation(self, mlp_clf):
        """
        roc_auc
        """
        
        mlp_clf.fit(Xtrain, ytrain)
        # calculate roc curve
        fpr_rf, tpr_rf, thresholds_rf = roc_curve(ytest, mlp_clf.predict_proba(Xtest)[:,1])
        print("FPR RF with sklearn model: {} \n".format(fpr_rf))
        print("TPR RF with sklearn model: {} \n".format(tpr_rf))
        print("THRESHOLD RF with sklearn model: {} \n".format(thresholds_rf))

        """
        Plot ROC 
        """
        skplt.metrics.plot_roc_curve(ytest, mlp_clf.predict_proba(Xtest))
        plt.show()

        """
        AUC
        """
        auc = roc_auc_score(ytest, mlp_clf.predict_proba(Xtest)[:,1])
        print('AUC with sklearn model: %.3f \n' % auc)

        """
        Accuracy score 
        """
        accuracy_score_value = accuracy_score(ytest, mlp_clf.predict_proba(Xtest)[:,1].round(), normalize=True, sample_weight=None)
        print("Accuracy Score with sklearn model: {}".format(accuracy_score_value))

        """
        Cross Validation
        """
        _scoring = ['accuracy', 'precision', 'recall', 'f1']
        results = cross_validate(estimator=model,X=X,y=Y,cv=6,scoring=_scoring,return_train_score=True)

        return {"Training Accuracy scores": results['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean()
              }

    def train_test_curve(self, mlp_clf):
        """
        Train test curve
        """

        train_sizes, train_scores, test_scores = learning_curve(estimator = mlp_clf,X = Xtrain, y = ytrain, train_sizes = [1, 14, 17], cv = 5)
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

    # def bagging(self, x, y, number_samples=5): 
    #     bag = []
    #     acc_old_output = 0
    #     for i in range(number_samples):
    #         sampling_x = []
    #         sampling_y = []
    #         for item in range(y.size):
    #             r_index = random.randrange(y.size)
    #             sampling_x.append(x[r_index,:])
    #             sampling_y.append(y[r_index])
    #         self.train(sampling_x, sampling_y)
    #         output = self.forward_propagate(sampling_x)
    #         acc = accuracy_score(sampling_y, output.round(), normalize=True, sample_weight=None)

    #         if (acc_old_output < acc):
    #             acc_old_output = acc
    #     return acc_old_output, sampling_y
        
    def bagging_sklearn(self, mlp_clf):
        """
        bagging
        """
        bag_clf = BaggingClassifier(
        mlp_clf,
        n_estimators=13,
        max_samples=5,
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


    def train(self, inputs, targets, epochs = 5, learning_rate = 1):
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
    Y = smoke_copy['Fire Alarm'].values 
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=2)

    sc = StandardScaler()
    sc.fit(Xtrain)
    Xtrain = sc.transform(Xtrain)
    Xtest = sc.transform(Xtest)

    print(f"The Shape of train set = {Xtrain.shape} \n")
    print(f"The Shape of test set = {Xtest.shape} \n")
    print(f"The Shape of train label = {ytrain.shape} \n")
    print(f"The Shape of test labels = {ytest.shape} \n")

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.90, random_state=2, shuffle=True)
    num_rows, num_cols = Xtrain.shape

    # Find the best hidden layer
    post_score = 0
        
    find_hiddle_layers = [[13], [14], [11]]
    hiddle_layer = find_hiddle_layers[0]
    for i in range(len(find_hiddle_layers)):
        clf = MLPClassifier(hidden_layer_sizes=find_hiddle_layers[i], random_state=1, max_iter=300).fit(Xtrain, ytrain)
        score = clf.score(Xtrain, ytrain)
        if (post_score < score) :
            hiddle_layer = find_hiddle_layers[i]
        post_score = score
    
    mlp = ANN(num_cols, hiddle_layer, 1)
    
    # model
    model = mlp.model_ann(hiddle_layer)

    # Feature importances - - (For feature importance please incomment the following lines of code)
    """
    # Start 
    f = []
    for j in range(Xtest.shape[1]):
        feature = mlp.get_feature_importance(j, 100, model)
        f.append(feature)
    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(Xtest.shape[1]), f, color="r", alpha=0.7)
    plt.xticks(ticks=range(Xtest.shape[1]), labels = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]',
       'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5',
       'NC0.5', 'NC1.0', 'NC2.5', 'CNT'])
    # data = pd.DataFrame({"column1": ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]',
    #    'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5',
    #    'NC0.5', 'NC1.0', 'NC2.5', 'CNT']})
    # data.plot(xticks=data.column1)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature importances")
    plt.show()

    #end
    """
    details = mlp.roc_auc_cross_validation(model)

    print("\n----Cross Validation----\n")
    print(details)
    print("\n")

    mlp.train_test_curve(model)

    """
    Output
    """
    output = mlp.forward_propagate(Xtest)
    
    accuracy_score_own_model = accuracy_score(ytest, output.round(), normalize=True, sample_weight=None)
    print("Accuracy Score with own model: {}".format(accuracy_score_own_model))
    print()
    print("=========Testing with inputs {} and output got is {}=========".format(Xtest, output.round()))
    

    """
    SK learn bagging
    """
    print("\n\n================================SK BAGGING=======================================\n\n")
    mlp_bagging_sk = ANN(num_cols, hiddle_layer, 1)
    y_prediction = mlp_bagging_sk.bagging_sklearn(model)

    # Error after bagging
    testing_error = mlp_bagging_sk.compute_error(ytest, y_prediction)
    print("Testing error after bagging with Sk learn model : {}".format(testing_error))

    """
    Own implementation bagging
    """
    # mlp_bagging_own = ANN(num_cols, hiddle_layer, 1)
    # y_prediction_own_bagging, sampling_y = mlp_bagging_own.bagging(Xtrain, ytrain)
    # Error after bagging
    # testing_error_own_imp = mlp_bagging_own.compute_error(sampling_y, y_prediction_own_bagging)
    # print("Testing error after bagging with own learn model : {}".format(testing_error_own_imp))

    