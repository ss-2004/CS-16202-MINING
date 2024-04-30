import pandas as pd 
import numpy as np

df = pd.read_csv("Buy_Computer.csv")
def count_class_labels(df, attribute):
    counts = {} 
    for value in df[attribute].unique(): 
        counts[value] = df[df[attribute] == value]['Buy_Computer'].value_counts().to_dict()
    return counts

counts_dict = count_class_labels(df,'age') 
print("Counts for 'age':",counts_dict)

def compute_class_label_probabilities(df, attribute):
    probabilities = {}
    total_instances = len(df) 
    for value in df[attribute].unique():
        counts = df[df[attribute] == value]['Buy_Computer'].value_counts().to_dict()
        probabilities[value] = { 'yes': counts.get('yes', 0) / total_instances, 'no': counts.get('no', 0) / total_instances} 
    return probabilities

probabilities_dict = compute_class_label_probabilities(df, 'age')
print("Probabilities for 'age':", probabilities_dict)

def compute_instance_probability(df, instance):
    total_instances = len(df)
    probabilities = {} 
    for label in ['yes', 'no']:
        label_probability = len(df[df['Buy_Computer'] == label]) / total_instances
        instance_probability = 1 
        for attribute, value in instance.items():
            attribute_probabilities = df[df['Buy_Computer'] == label][attribute].value_counts(normalize=True).to_dict()
            attribute_probability = attribute_probabilities.get(value, 0)
            instance_probability *= attribute_probability 
            probabilities[label] = instance_probability * label_probability 
    return probabilities

instance = {'age': 'youth', 'income': 'medium', 'student':'yes','credit_rating': 'fair'} 
instance_probabilities = compute_instance_probability(df, instance) 
print("Probability of instance X:", instance_probabilities)

def justify_instance_probability(instance_probabilities):
    if instance_probabilities['yes'] > instance_probabilities['no']:
        return "The probability of the instance belonging to 'yes' class is higher." 
    elif instance_probabilities['no'] > instance_probabilities['yes']:
        return "The probability of the instance belonging to 'no' class is higher." 
    else:
        return "The probabilities are equal."

justification = justify_instance_probability(instance_probabilities)
print("Justification:", justification)