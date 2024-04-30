# Implementation of classification using j-48 algorithm in python on ‘weather.nominal.arff’
# (storm.cis.fordham.edu/~gweiss/data-mining/weka-data/weather.nominal.arff) dataset & verify the result
# with Objective 1.

import pandas as pd
from weka.classifiers import Classifier
from weka.core.converters import Loader
from weka.classifiers import Evaluation
from weka.core.classes import Random

# Load dataset
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file("weather.nominal.arff")
data.class_is_last()
# Preprocess the data (not necessary for this dataset)
# Implement J48 algorithm
cls = Classifier(classname="weka.classifiers.trees.J48")
cls.build_classifier(data)
# Verify the result
evaluation = Evaluation(data)
evaluation.crossvalidate_model(cls, data, 10, Random(1))
# Print classification results
print("Classification Results:")
print(evaluation.summary())
# Print entropy values and Kappa statistic
print("Entropy:", evaluation.mean_entropy())
print("Kappa:", evaluation.kappa())
# Extract if-then rules
print("If-then rules:")
print(cls)

# from sklearn import datasets
# from sklearn.model_selection import train_test_split 
# from sklearn.preprocessing import OneHotEncoder 
# from sklearn import tree
# import pandas as pd
# # Load dataset
# filename = '/Users/ShresthS/Desktop/CSE/ASSGN/SEM6/MINE/2024-04-24/assgn8/Buy_Computer.csv'
# df = pd.read_csv(filename)
# # Preprocess dataset
# enc = OneHotEncoder() 
# enc.fit(weather.data)
# X = enc.transform(weather.data)
# # Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(X, weather.target, test_size=0.3)
# # Create Decision Tree classifer object 
# clf = tree.DecisionTreeClassifier()
# # Train Decision Tree Classifer 
# clf = clf.fit(X_train,y_train)
# # Predict the response for test dataset 
# y_pred = clf.predict(X_test)
