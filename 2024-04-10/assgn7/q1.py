# a. Prepare the dictionary which counts the number of “yes” and “no” class labels for each nominal value in
# an attribute.

import pandas as pd
def count_class_labels(df, attribute):
    
    counts_dict = {}
    unique_values = df[attribute].unique()
    for value in unique_values:
        counts_dict[value] = {
            'yes': df[(df[attribute] == value) & (df['buys_computer'] == 'yes')].shape[0],
            'no': df[(df[attribute] == value) & (df['buys_computer'] == 'no')].shape[0]}
    return counts_dict

df = pd.read_csv('buys_computer.csv')

counts_dict = count_class_labels(df, 'age')
print(counts_dict)
