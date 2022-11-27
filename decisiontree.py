from __future__ import print_function
from matplotlib import pyplot as py_plot
import math
import numpy as np
from random import randrange

import graphviz
import os
import random

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)
    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE

    partition_dictX = {}
    try: 
        #numpy.unique gives a more compact way of writing this function
        #gives the unique elements in a vector as a list
        for i in np.unique(x):  
            partition_dictX.update({i : (x == i).nonzero()[0]})   
        return partition_dictX
    except:
        raise Exception('partition function error!')
    return partition_dictX
    raise Exception('Function not yet implemented!')


def entropy(y, weights=None):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z
    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    try:
        length_y = len(y)
        dictionary_y  = partition(y)
        suprise = 0
        if weights is None:
            for key in list(dictionary_y.keys()):
                probablity = dictionary_y[key].size/length_y          
            suprise -= probablity*math.log2(probablity)
            return suprise
        
        if weights is not None:
            sum_weights = np.sum(weights)
            for key in list(dictionary_y.keys()):
                wei_probability = 0
                for ind in list(dictionary_y[key]):
                    wei_probability += weights[ind]/sum_weights
                probablity = wei_probability     
            suprise -= probablity*math.log2(probablity)
            return suprise

    except:
        raise Exception('entropy function error!')
    return suprise
    raise Exception('Function not yet implemented!')

def mutual_information(x, y, weights=None):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.
    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    try: 
        if weights is None:
            entropy_class  = entropy(y)
            valuex, countx = np.unique(x,return_counts = True)
    
            mutual_info = 0.0
            for i in range(len(valuex)):
                prob = countx[i].astype('float')/len(x)
                mutual_info += prob*entropy(y[x==valuex[i]])
            return entropy_class - mutual_info
        
        if weights is not None:
            entropy_class = entropy(y,weights)
            weight_sum = np.sum(weights)
            partitions = partition(x)
            mutual_info = 0.0
            for part in partitions.keys():
                sum_prob = 0.0
                for index in partitions[part]:
                    sum_prob += weights[index]/weight_sum
                prob = sum_prob
                mutual_info += prob*entropy(y[x==part],weights)
            return entropy_class - mutual_info
    except: 
        raise Exception('mutual_information function error!')
    raise Exception('Function not yet implemented!')

def compute_error(y_true, y_pred, weights = None):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)
    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    if weights is None:
        n = len(y_true)
        error_y = [y_true[ele] != y_pred[ele] for ele in range(n)]
        return sum(error_y)/n
    
    if weights is not None:
        sum_weights = np.sum(weights)
        n = len(y_true)
        error_y = [int(y_true[i] != y_pred[i]) for i in range(n)]
        sum_error = 0
        for i in range(n):
            sum_error += error_y[i]*weights[i]
        return sum_error/sum_weights

    raise Exception('Function not yet implemented!')


def prettyPrint(decision_tree, depth_tree=0):
   
    if depth_tree == 0:
        print('TREE')

    for index, split_criterion in enumerate(decision_tree):
        sub_branches_trees = decision_tree[split_criterion]

        print('|\t' * depth_tree, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        if type(sub_branches_trees) is dict:
            prettyPrint(sub_branches_trees, depth_tree + 1)
        else:
            print('|\t' * (depth_tree + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_branches_trees))



def renderFile(blackbox, save_file, image_format='png'):

    if type(blackbox).__name__ != 'str':
        raise TypeError('Error in tree.\n')

    table_graph = graphviz.Source(blackbox)
    table_graph.format = image_format
    table_graph.render(save_file, view=True)


def diagGraphviz(tree, blackbox='', unique_id=-1, depth=0):

    unique_id += 1       # Running index of node ids across recursion
    node_id = unique_id  # Node id of this node

    if depth == 0:
        blackbox += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_branches_trees = tree[split_criterion]
        attr_i = split_criterion[0]
        attr_val = split_criterion[1]
        branch_condition = split_criterion[2]

        if not branch_condition:
            # Alphabetically, False comes first
            blackbox += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attr_i, attr_val)

        if type(sub_branches_trees) is dict:
            if not branch_condition:
                blackbox, right_child, unique_id = diagGraphviz(sub_branches_trees, blackbox=blackbox, unique_id=unique_id, depth=depth + 1)
                blackbox += '    node{0} -> node{1} [label = "False"];\n'.format(node_id, right_child)
            else:
                blackbox, left_child, unique_id = diagGraphviz(sub_branches_trees, blackbox=blackbox, unique_id=unique_id, depth=depth + 1)
                blackbox += '    node{0} -> node{1} [label = "True"];\n'.format(node_id, left_child)

        else:
            unique_id += 1
            blackbox += '    node{0} [label = "y = {1}"];\n'.format(unique_id, sub_branches_trees)
            if not branch_condition:
                blackbox += '    node{0} -> node{1} [label = "False"];\n'.format(node_id, unique_id)
            else:
                blackbox += '    node{0} -> node{1} [label = "True"];\n'.format(node_id, unique_id)

    if depth == 0:
        blackbox += '}\n'
        return blackbox
    else:
        return blackbox, node_id, unique_id

def errPlot(x_training,ytrn,xtst,ytst,j):
    from sklearn.metrics import confusion_matrix
    confusions={}
    d={}
    for i in range(1,11):
        decision_tree=id3(x_training,ytrn,depth = 0,max_depth=i)

        y_pred_training = [predict_example(x, decision_tree) for x in x_training]

        err_trn=compute_error(y_pred_training,ytrn)
        y_pred_tst = [predict_example(x, decision_tree) for x in xtst]
        if i in [1, 2]:
            confusions[i]=confusion_matrix(ytst,y_pred_tst)
            dot_str = diagGraphviz(decision_tree)

            renderFile(dot_str, './error_plot_treeDepth/treeDepth'+str(i))
            import scikitplot as skp

            skp.metrics.plot_confusion_matrix(ytst,y_pred_tst)
            py_plot.title('DEPTH:'+str(i))
            py_plot.show()
        err_tst = compute_error(y_pred_tst, ytst)
        d[i]=(err_trn,err_tst)
    plotData(d,j)
    return confusions

def plotData(d,j):
    py_plot.figure()
    py_plot.title("DIAGRAM"+str(j))
    trnerr=[]
    tsterr=[]
    depths=[]
    all=[]
    for i in range(1,11):
        (trn,tst)=d[i]
        trnerr.append(trn)
        tsterr.append(tst)
        depths.append(i)
        all.append(trn)
        all.append(tst)

    py_plot.plot(depths,trnerr, marker='o', linewidth=3, markersize=12)
    py_plot.plot(depths,tsterr, marker='s', linewidth=3, markersize=12)
    py_plot.ylabel('Train/Test error', fontsize=16)
    py_plot.xlabel('max_depth', fontsize=16)

    py_plot.legend(['Train Error', 'Test Error'], fontsize=16)

    py_plot.show()

def skLearn(x,y,xtst,ytst):

    confusions={}
    for i in [1,2,3,4,5,6,7,8,9,10]:
        from sklearn import tree
        from sklearn.metrics import confusion_matrix, accuracy_score
        
        dtree = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=i)
        dtree = dtree.fit(x, y)
        dot_str = tree.export_graphviz(dtree, out_file=None)
        renderFile(str(dot_str), './skLearn/sklearn_tree of depth '+str(i))

        dtree = dtree.fit(x, y)
        dtree = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=i)
        dtree=dtree.fit(x,y)
        y_pred = dtree.predict(xtst)
        import scikitplot as skp
        skp.metrics.plot_confusion_matrix(ytst, y_pred)
        py_plot.title('depth fig:' + str(i))
        py_plot.show()

        confusions[i]=confusion_matrix(ytst,y_pred)
        print(accuracy_score(ytst,y_pred))

    return confusions

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=3, weights=None):

    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.
    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.
    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels
    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    try: 
        tree = {}
        if attribute_value_pairs is None:
            attribute_value_pairs = np.vstack([[(ind, val) for val in np.unique(x[:, ind])] for ind in range(x.shape[1])])
        values_y, count_y = np.unique(y, return_counts=True)
        if len(values_y) == 1:      
            return values_y[0]
        if len(attribute_value_pairs) == 0 or depth == max_depth:   
            return values_y[np.argmax(count_y)]
        
        mutual_info = np.array([mutual_information(np.array(x[:, i] == val).astype(int), y,weights)
                                    for (i, val) in attribute_value_pairs])
        
        (best_attr, best_val) = attribute_value_pairs[np.argmax(mutual_info)]   
        partitions = partition(np.array(x[:, best_attr] == best_val).astype(int))   
        
        delete_index = np.all(attribute_value_pairs == (best_attr, best_val), axis=1)    
        
        attribute_value_pairs = np.delete(attribute_value_pairs, np.argwhere(delete_index), 0)
            
        for split_val, indices in partitions.items():
            split_x = x.take(indices, axis=0)
            split_y = y.take(indices, axis=0)
            decision = bool(split_val)
            
            if weights is None:
                tree[(best_attr, best_val, decision)] = id3(split_x, split_y, attribute_value_pairs=attribute_value_pairs,
                                            max_depth=max_depth, depth=depth + 1)
        
            if weights is not None:
                take_weights = weights.take(indices, axis=0)
                tree[(best_attr, best_val, decision)] = id3(split_x, split_y, attribute_value_pairs=attribute_value_pairs,
                                            max_depth=max_depth, depth=depth + 1, weights = take_weights)
            
    except: 
        raise Exception('id3 function error!')
    return tree
    raise Exception('Function not yet implemented!')

def bagging(x, y, max_depth, num_trees):

    bag = []
    
    for i in range(num_trees):
        sampling_x = []
        sampling_y = []
        for item in range(y.size):
            r_index = random.randrange(y.size)
            sampling_x.append(x[r_index,:])
            sampling_y.append(y[r_index])
        dt = id3(np.array(sampling_x),np.array(sampling_y),depth = 0, max_depth = max_depth)
        bag.append(dt)
        
    return bag

def prediction_labels(x, tree):
   
    for val_split in tree:
        if val_split[2] == (x[val_split[0]] == val_split[1]):
            if type(tree[val_split]) is dict:
                label = prediction_labels(x, tree[val_split])
            else:
                label = tree[val_split]

            return label

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.
    Returns the predicted label of x according to tree
    """

    for splitting_value, tree_sub_part in tree.items():
        a_index = splitting_value[0]
        a_value = splitting_value[1]
        a_decision = splitting_value[2]

        if a_decision == (x[a_index] == a_value):
            if type(tree_sub_part) is dict:
                label = predict_example(x, tree_sub_part)
            else:
                label = tree_sub_part

            return label
    
    raise Exception('Function not yet implemented!')

def drawConfusionMatrix(value_true, value_pred):

  unique_true = len(np.unique(value_true)) 
  matric = np.zeros((unique_true, unique_true))

  for ind in range(len(value_true)):
    matric[value_true[ind]][value_pred[ind]] += 1

  return matric.astype(int)

def boosting(x, y, max_depth,num_stumps):

    boosting=[]
    
    n = y.size
    dl = np.array([])
   
    for i in range(n):
        dl = np.append(dl,1/n)
    
    for num in range(num_stumps):
        dt = id3(x, y, depth=0, max_depth=max_depth, weights = dl)
        y_pred = [predict_example(row, dt) for row in x]
        
        false_pred = [int(y[i] != y_pred[i]) for i in range(len(y))]    
        for val in range(len(false_pred)):
            if(false_pred[val] == 0):
                false_pred[val] = -1
        error = compute_error(y,y_pred,dl)                          
        stage = 0.5*math.log((1 - error)/error)                           
        boosting.append((stage,dt))                                  
        
        dl = np.array([dl[i]*math.exp(stage*false_pred[i]) for i in range(len(dl))]) 
    
    return boosting

def predictExamplesBagging(x_value_test, bags):
    res = 0
    labels = []
    for ind in range(len(bags)):
        labels.append(predict_example(x_value_test,bags[ind]))
    res = max(labels, key=labels.count)
    return res
    
def predictEnsembles(x_value_test, ensemble):
    res = 0
    for ind in range(len(ensemble)):
         for splitval, samp_trees in ensemble[ind][1].items():
             attribute_number = splitval[0]
             attribute_value = splitval[1]
             attribute_decision = splitval[2]

         if attribute_decision == (x_value_test[attribute_number] == attribute_value):
            if type(samp_trees) is dict:
                tree_label = predict_example(x_value_test, samp_trees)
            else:
                tree_label = samp_trees
            
            res += int(tree_label)*ensemble[ind][0]
    
    if(res > 0.5):
        return 1
    else:
        return 0
    
    raise Exception('Function not yet implemented!')

def sklearnBagging(Xtrn, ytrn, Xtst, ytst):
     print("\n\n\nscikit-learn Bagging :------------------->")   
     for max_depth in (3,5):
        for num_trees in (10,20):
            dt = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=max_depth)
            bagged_tree = BaggingClassifier(base_estimator = dt, n_estimators = num_trees)
            bagged_tree.fit(Xtrn,ytrn)
            y_prediction = bagged_tree.predict(Xtst)
            testing_error = compute_error(ytst, y_prediction)
            print("Number of trees : \n",num_trees)
            print("Max depth of tree : \n",max_depth)
            print('Test Error : {0:4.2f}%'.format(testing_error * 100))
            print("Confusion Matrix:\n",drawConfusionMatrix(ytst, y_prediction))
            print("======================================")

def sklearnBoosting(Xtrn, ytrn, Xtst, ytst):
     print("\n\n\nscikit-learn Boosting :------------------->")   
     for max_depth in (1,2):
        for num_trees in (20,40):
            dt = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=max_depth, random_state=2)
            boost_tree = AdaBoostClassifier(base_estimator = dt, n_estimators = num_trees, random_state=5)
            boost_tree.fit(Xtrn,ytrn)
            y_prediction = boost_tree.predict(Xtst)
            testing_error = compute_error(ytst, y_prediction)
            print("Number of trees : \n",num_trees)
            print("Max depth of tree : \n",max_depth)
            print('Test Error : {0:4.2f}%'.format(testing_error * 100))
            print("Confusion Matrix : \n",drawConfusionMatrix(ytst, y_prediction))
            print("======================================")

if __name__ == '__main__':

    import pandas as pd

    # Load the training data
    smoke = pd.read_csv("./smoke.csv")

    smoke_copy = smoke.copy()
    X = smoke_copy.drop(columns=['Fire Alarm']).values
    Y = smoke_copy['Fire Alarm'].values 
    Xtrn, Xtst, ytrn, ytst = train_test_split(X, Y, test_size=0.2, random_state=2)

    print("Bagging :------------------->")   
    for max_depth in (3,5):
        for num_trees in (10,20):
            res=bagging(Xtrn, ytrn, max_depth, num_trees)
            y_prediction = [predictExamplesBagging(x_value_test,res) for x_value_test in Xtst]
            testing_error = compute_error(ytst, y_prediction)
            print("No. of trees : \n",num_trees)
            print("Max depth of tree : \n",max_depth)
            
            print('Test Error : {0:4.2f}%'.format(testing_error * 100))
            print("Confusion Matrix : \n",drawConfusionMatrix(ytst, y_prediction))
            print("======================================")

    sklearnBagging(Xtrn, ytrn,Xtst,ytst)
    
    print("\n\n\nBoosting :------------------->") 
    for max_depth in (1,2):
        for num_trees in (20,40):
            res=boosting(Xtrn, ytrn, max_depth, num_trees)
            y_prediction = [predictEnsembles(x_value_test,res) for x_value_test in Xtst]
            testing_error = compute_error(ytst, y_prediction)
            print("Number of trees : \n",num_trees)
            print("Max depth of tree : \n",max_depth)
            
            print('Test Error : {0:4.2f}%'.format(testing_error * 100))
            print("Confusion Matrix : \n",drawConfusionMatrix(ytst, y_prediction))
            print("======================================")
            
    sklearnBoosting(Xtrn, ytrn,Xtst,ytst)