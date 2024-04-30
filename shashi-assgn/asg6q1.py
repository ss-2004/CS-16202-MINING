import pandas as pd 
import numpy as np
data = pd.read_csv("vehicle.csv")
def information_gain(data, attribute_name, target_name):
    entropy_total = calculate_entropy(data[target_name])
    values, counts = np.unique(data[attribute_name],return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) *
    calculate_entropy(data.where(data[attribute_name] ==
    values[i]).dropna()[target_name])
    for i in range(len(values))])
    return entropy_total - weighted_entropy

def gini_index(data, attribute_name, target_name):
    gini_total = calculate_gini(data[target_name]) 
    values, counts = np.unique(data[attribute_name], return_counts=True)
    weighted_gini = np.sum([(counts[i] / np.sum(counts)) * calculate_gini(data.where(data[attribute_name] == values[i]).dropna()[target_name])
    for i in range(len(values))])
    
    return gini_total - weighted_gini

def gain_ratio(data, attribute_name, target_name):
    info_gain = information_gain(data, attribute_name, target_name)
    split_info = calculate_entropy(data[attribute_name])
    return info_gain / split_info

def calculate_entropy(target):
    entropy = 0
    values, counts = np.unique(target, return_counts=True)
    for i in range(len(values)):
        prob = counts[i] / np.sum(counts)
        entropy -= prob * np.log2(prob)
        return entropy

def calculate_gini(target):
    gini = 1
    values, counts = np.unique(target, return_counts=True)
    for i in range(len(values)):
        prob = counts[i] / np.sum(counts)
        gini -= prob ** 2
    return gini

split_points_info_gain = {col: information_gain(data, col, 'Fuel_Type') for col in data.columns if col != 'Fuel_Type'}
split_points_gini_index = {col: gini_index(data, col, 'Fuel_Type') for col in data.columns if col != 'Fuel_Type'} 
split_points_gain_ratio = {col: gain_ratio(data, col, 'Fuel_Type') for col in data.columns if col != 'Fuel_Type'}

print("Split points using Information Gain:", split_points_info_gain) 
print("Split points using Gini Index:", split_points_gini_index) 
print("Split points using Gain Ratio:", split_points_gain_ratio)