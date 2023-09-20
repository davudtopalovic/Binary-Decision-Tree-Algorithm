#!/usr/bin/env python
# coding: utf-8

# # Binary Decision Tree Classifier

# # Import tools
# We are importing all the necessary libraries. 

# In[103]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#scikit learn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from collections import Counter
import random
import time


# ## Constructing a model (algorithm)

# In[104]:


class TreeNode:
    def __init__(self, X, y, split_index=None, threshold=None, left=None, right=None, info_gain=None, curr_depth=0, root=False, leaf=False,
                 classes=None, features = None):
        #DECISION NODE
        #index of the feature selected for splitting
        self.split_index = split_index
        #threshold value - certain value of the feature selected for the split
        self.threshold = threshold
        #left child
        self.left = left
        #right child
        self.right = right
        #information gain provided with the split (essentially this decide which split we will be making)
        self.info_gain = info_gain
        
        #LEAF NODE
        self.leaf = leaf
        
        #for BOTH NODES
        #depth of a node (this is only used for printing the tree)
        self.curr_depth = curr_depth
        #features in the current node
        self.X = X
        #classes in the current node
        self.y = y
        
        #string classes
        self.classes = classes
        
        
    def predict(self, X):
        "function for predicting classes"
        if self.split_index is None:
            return np.array([np.argmax(np.bincount(self.y)) for _ in range(X.shape[0])])
        pred = np.zeros(X.shape[0])
        left_idx = X[:, self.split_index] <= self.threshold
        pred[left_idx] = self.left.predict(X[left_idx])
        pred[~left_idx] = self.right.predict(X[~left_idx])
        return pred.astype(int)

    def print_split_idxs(self, node=None):
        "function for printing all split indices of the tree"
        if not node:
            node = self
        if node.split_index is not None:
            print(node.split_index, end=" ")
        if node.leaf == False:
            self.print_split_idxs(node.left)
            self.print_split_idxs(node.right)
    
    def print_split_idxs_and_thrshld(self, node=None):
        "function for printing all split indices of the tree"
        if not node:
            node = self
        if node.split_index is not None:
            print(f"X_{node.split_index}: {node.threshold}", end=" ")
        if node.leaf == False:
            self.print_split_idxs_and_thrshld(node.left)
            self.print_split_idxs_and_thrshld(node.right)

    def print_tree(self, tree=None, indent=" "):
        "function for visualizing the tree"
        
        if not tree:
            tree = self
            
        if tree.leaf:
            print(np.argmax(np.bincount(tree.y)), f"(#{len(tree.y)})")

        else:
            print(" X_\033[0m"+str(tree.split_index), "≤", np.round(tree.threshold,3), "?" ,np.round(tree.info_gain,3), f"(#{len(tree.y)})")
            print(tree.curr_depth + 1,":","%sleft: " % (indent), end="")
            self.print_tree(tree.left, indent + "   ")
            print(tree.curr_depth + 1 ,":","%sright: " % (indent), end="")
            self.print_tree(tree.right, indent + "   ") 
        

    
    def print_colorized_tree_with_original_labels(self, tree=None, indent=" ", features=None):
        "function for visualizing the tree"
        
        if not tree:
            tree = self
            
        if tree.leaf:
            class_counts = np.bincount(tree.y)
            max_class_count = np.max(class_counts)
            max_class_index = np.argmax(class_counts)
            num_classes = len(class_counts)

            # Define colors for different classes (you can customize these)
            colors = ["\033[95m", "\033[94m", "\033[93m", "\033[92m", "\033[91m"]

            # Assign a color based on the class with the highest count
            color_index = max_class_index % len(colors)
            class_color = colors[color_index]

            # Print the class label in the assigned color
            print(f"{class_color}{tree.classes[max_class_index]} ({max_class_count})\033[0m",f"{class_counts}")
            

        else:
            class_counts = np.bincount(tree.y)
            if features is not None:
                splitting_feature = features[tree.split_index]
                self.features = features
            else: 
                splitting_feature = str(tree.split_index)
                self.features = None
            print("\033[1mX_\033[0m"+f"\033[1m {splitting_feature}\033[0m", "\033[1m≤\033[0m", f"\033[1m{np.round(tree.threshold,3)}\033[0m", "?" ,np.round(tree.info_gain,3), f"(#{len(tree.y)})", f"{class_counts}")
            print(tree.curr_depth + 1,":","%sleft: " % (indent), end="")
            self.print_colorized_tree_with_original_labels(tree.left, indent + "   ", features = self.features)
            print(tree.curr_depth + 1 ,":","%sright: " % (indent), end="")
            self.print_colorized_tree_with_original_labels(tree.right, indent + "   ", features = self.features) 

def all_columns(X, rand):
    "this function returns the range of the features"
    return range(X.shape[1])


def to_int(y):
    "function to convert string classes to integer classes"
    y = y.copy()
    cl = np.unique(y)
    for i,cl in enumerate(cl):
        y[y==cl] = i
    return y.astype(int).flatten()


class Tree:
    def __init__(self, 
                 rand=None,
                 get_candidate_columns=all_columns,
                 min_samples=2, 
                 max_depth = float('inf')):
        
        if rand is None:
            rand = random.Random(42)
        self.rand = rand
        self.get_candidate_columns = get_candidate_columns #needed for random forests
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.classes = None

    def to_int(self, y):
        "function to convert string classes to integer classes"
        y = y.copy()
        cl = np.unique(y)
        self.classes = cl
        for i,cl in enumerate(cl):
            y[y==cl] = i
        return y.astype(int).flatten()
    
    
    #MOST IMPORTANT FUNCTION - recursive function for building a binary tree using a recursive function. 
    #This function takes dataset as an input, performs a best split of the dataset - creating left and right child,
    #which either can be pure leaf node (node with only data points with one class) 
    #or a node with the remaining data and the condition that performs further splits of the data in that node.
    def build(self, X, y, curr_depth=0):
        '''recursive function to build the tree''' 
        
        #converting string classes to integer classes
        if curr_depth == 0:
            y = self.to_int(y)

        
        num_samples, num_features = np.shape(X)

        
        #store selected features we want to consider for finding the split
        candidates = np.array(self.get_candidate_columns(X, self.rand))
        self.rand.shuffle(candidates)

        
        #Now split until stopping conditions are met
        
        #if there's only one class present stop splitting and return leaf node
        if np.unique(y).shape[0] == 1:
            return TreeNode(X, y, curr_depth=curr_depth, leaf=True, classes = self.classes)
        #if the node doesn't have less than minimum number samples go for finding the best split
        if len(X) >= self.min_samples and curr_depth < self.max_depth:
            
            #on current data point apply best_split and obtain the split index, split threshold and information gain.
            split_index, split_threshold, split_info_gain = self.best_split(X, y, candidates)
            
            
            #now create the indices of the left and right child with the function split
            left_idxs, right_idxs = self.split(X[:, split_index], split_threshold)
        
            #if the information gain is positive create the left and right child and on each of the child recursively
            #call the build function. At the end return the current decision node.
        
            ##print("NEXT NODE left- recursion")
            # recur left
            left = self.build(X[left_idxs,:], y[left_idxs], curr_depth + 1)
            #recur right
            right = self.build(X[right_idxs,:], y[right_idxs], curr_depth + 1)


            if curr_depth == 0:

                #returning the root node
                return TreeNode(X, y, split_index, split_threshold, 
                                left, right, split_info_gain, curr_depth=0, root=True)
            
            #returing the decision node
            return TreeNode(X, y, split_index, split_threshold, left, right, split_info_gain, curr_depth=curr_depth)
        

        
        #returning the leaf node

        return TreeNode(X, y, curr_depth=curr_depth, leaf=True, classes = self.classes)

    def best_split(self, X, y, candidates):
        ''' function to find the best split '''
        #initialize the important variables
        split_index, split_threshold, split_info_gain = None, None, None
        max_info_gain = -float("inf")
        
        #loop through every feature
        for feature_index in candidates:
            #create a column of the current feature values of all data points
            X_column = X[:, feature_index]
            
            #find the unique values of that feature, one of the values might be chosen for the split
            possible_thresholds = np.unique(X_column)

            
            # loop over all the unique feature values present in the data
            # Iterate over adjacent unique sorted values to find midpoints
            thresholds = []
            for i in range(len(possible_thresholds) - 1):

                # Calculate midpoint between adjacent values
                threshold = (possible_thresholds[i] + possible_thresholds[i + 1]) / 2
                thresholds.append(threshold)
                
                
                
                #for this threshold value capture the information gain by calling the information_gain function.
                #This function will split the data points with respect to the threshold and calculate the infogain
                #of such split
                curr_info_gain = self.information_gain(y, X_column, threshold)
                #if the current info gain is larger than the last maximum info gain update the important variables.
                if curr_info_gain > max_info_gain: 
                    split_index = feature_index
                    split_threshold = threshold
                    split_info_gain = curr_info_gain
                    max_info_gain = curr_info_gain
            thresholds = []
                    
        #when two for loops are done return the best values of important variables.
        return split_index, split_threshold, split_info_gain
    
    def information_gain(self, y, X_column, split_threshold):
        "function to calculate the information gain of the split"

        #generate split, that is get the left and right indices 
        left_idxs, right_idxs = self.split(X_column, split_threshold)
        
        #if either left or right indices are zero, there is no information gain because we end up with
        #one empty node and the same node we perfomed a split on.
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0 
        
        #calculate the information gain of the split with gini index (Gini impurity)
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        weight_l = n_l / n
        weight_r = n_r / n
        gain = self.gini_index(y) - (weight_l*self.gini_index(y[left_idxs]) + weight_r*self.gini_index(y[right_idxs]))
        
        #return information gain
        return gain

    def split(self, X_column, split_threshold):
        "function to split the data"
        
        #left indices are the indices of data points which feature value is less than or equal to threshold
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        
        #right indices are the indices of data points which feature value is larger than threshold
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        
        return left_idxs, right_idxs
        
    def gini_index(self, y):
        ''' function to compute gini index '''
        #gini_index = 1 - ∑p_i**2, where p_i = probability of class i 
        #Why would we use gini function? Unlike entropy function, gini doesn't have logarithmic part,
        #so by choosing gini function we have actually done a favor to us which is saving computation time - 
        #(it is easier to find square of a quantity than to find the logarithm.
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini

