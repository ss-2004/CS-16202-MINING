import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_graphviz
import graphviz

df = pd.read_csv("/Users/ShresthS/Desktop/CSE/ASSGN/SEM6/MINE/2024-04-10/assgn6/vehicle.csv")

print("Missing values before preprocessing:")
print(df.isnull().sum())

df.dropna(inplace=True)

print("\nMissing values after preprocessing:")
print(df.isnull().sum())

X = df.drop(columns=['make'])
y = df['make']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

dot_data = export_graphviz(clf, out_file=None,
                          feature_names=X.columns,
                          class_names=y.unique(),
                          filled=True, rounded=True,
                          special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("vehicle_decision_tree", format='png', cleanup=True)