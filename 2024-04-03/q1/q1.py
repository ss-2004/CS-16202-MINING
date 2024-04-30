# Note: Feature Selection- Select k useful features out of n futures in a dataset, where k<n.
# Objective 1:
# Load page_block.csv numeric (Beginner's Guide Page-block Classification (kaggle.com)) dataset (make
# appropriate preprocessing if required). Use Fisher’s Score (f) ranking method for assigning an
# appropriate rank to each feature and select the top k ranked features.

# Where μ1, μ2, σ1, σ2 are means and standard deviations for negative and positive class respectively.
# Write the equivalent function in python for the following:
# 1. Compute the mean of attribute values against both class labels.
# 2. Compute the standard deviation of attribute values against both class labels.
# 3. ComputeFisher’s score (f) the for each attribute.
# 4. Assigned rank for each attribute (High f value has a high ranking).

import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("page_block.data")
X = data.drop("Class", axis=1)  
y = data["Class"] 

X = X.fillna(X.mean())  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def class_wise_stats(data, attribute):
    class_0_data = data[data["Class"] == 0][attribute]
    class_1_data = data[data["Class"] == 1][attribute]
    mean_0 = class_0_data.mean()
    mean_1 = class_1_data.mean()
    std_0 = class_0_data.std()
    std_1 = class_1_data.std()
    return mean_0, mean_1, std_0, std_1

def fisher_score_wrapper(X, y):
    scores = []
    for attr in X.columns:
        mean_0, mean_1, std_0, std_1 = class_wise_stats(data, attr)
        if std_0 + std_1 == 0:  
            score = 0
        else:
            score = (mean_1 - mean_0) ** 2 / (std_0 ** 2 + std_1 ** 2)
        scores.append(score)
    return scores

scores = fisher_score_wrapper(X_scaled, y)
ranks = np.argsort(scores)[::-1] 

top_k = 5  
top_k_features = X.columns[ranks[:top_k]]

print("Top", top_k, "features with highest Fisher's score:")
print(top_k_features)
