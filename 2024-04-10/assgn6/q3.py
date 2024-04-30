# Design module which predicts the class label of unknown and unseen data using tree
# traversal or any other techniques.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('/Users/ShresthS/Desktop/CSE/ASSGN/SEM6/MINE/2024-04-10/assgn6/vehicle.csv')

df.dropna(inplace=True)

X = df.drop(columns=['make'])
y = df['model']

X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
