import numpy as np
import pandas as pd
def fisher_score(data, target):
    classes = np.unique(target) 
    if len(classes) != 2: raise ValueError("Fisher's score is only applicable for binary classification.")
    means = np.zeros(shape=(len(features),))
    stds = np.zeros(shape=(len(features),))
    for i, feature in enumerate(features):
        for class_label in classes:
            means[i] += np.mean(data[target == class_label, i])
            means[i] /= len(classes) 
            for class_label in classes:
                stds[i] += np.std(data[target == class_label, i])**2
                stds[i] /= len(classes)
                fisherscore = (means[0] - means[1])**2 / (stds[0] + stds[1])
    return fisherscore

def entropy(target):
    classes, counts = np.unique(target, return_counts=True)
    probabilities = counts / len(target) 
    entropy = np.sum(probabilities * np.log2(probabilities)) 
    return entropy 

def info_a(data, target, attribute):
    unique_values = np.unique(data[:,attribute])
    entropy_a = 0 
    for value in unique_values:
        subset = data[data[:, attribute] == value] 
        subset_target = target[data[:, attribute] == value] 
        entropy_a += len(subset) / len(data) * entropy(subset_target) 
        return entropy_a

def information_gain(data, target, attributes):
    parent_entropy = entropy(target) 
    information_gains = np.zeros(shape=(len(attributes),)) 
    for i, attribute in:
        enumerate(attributes)
        information_gains[i] = parent_entropy - info_a(data, target, attribute) 
    return information_gains