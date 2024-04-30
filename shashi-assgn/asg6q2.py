from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split 
import graphviz
import pandas as pd

data = pd.read_csv('vehicles.csv')

X = data.drop(columns=['Fuel_Type'])
X = pd.get_dummies(X, columns=['Car_Name', 'Seller_Type', 'Transmission']) 
y = data['Fuel_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def build_decision_tree(X_train, y_train, binary=True):
    if binary:
        tree = DecisionTreeClassifier(splitter='best')
    else:
        tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    return tree

def visualize_decision_tree(tree, feature_names, class_names, filename):
    dot_data = export_graphviz(tree, out_file=None,
    feature_names=feature_names,
    class_names=class_names,
    filled=True, rounded=True,
    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(filename, format='png')
    binary_tree = build_decision_tree(X_train, y_train, binary=True)
    visualize_decision_tree(binary_tree, feature_names=X_train.columns,
    class_names=y_train.unique(), filename='binary_tree')
    general_tree = build_decision_tree(X_train, y_train, binary=False)
    visualize_decision_tree(general_tree, feature_names=X_train.columns,
    class_names=y_train.unique(), filename='general_tree')

accuracy = accuracy_score(y_test, binary_tree_predictions)
print("Accuracy for Binary Tree Model:", accuracy)
# Calculate accuracy for General Tree Model accuracy =
accuracy_score(y_test, general_tree_predictions)
print("Accuracy for General Tree Model:", accuracy)