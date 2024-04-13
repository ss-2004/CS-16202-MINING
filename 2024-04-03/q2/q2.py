import numpy as np

def entropy(dataset):
  class_labels, counts = np.unique(dataset[:, -1], return_counts=True)
  probabilities = counts / len(dataset)
  return -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))

def info_a(dataset, attribute):
  unique_values, counts = np.unique(dataset[:, attribute], return_counts=True)
  probabilities = counts / len(dataset)
  entropy_values = entropy(dataset[dataset[:, attribute] == False])
  return np.sum(probabilities * entropy_values)

def information_gain(dataset, attribute):
  parent_entropy = entropy(dataset)
  info_a_value = info_a(dataset.copy(), attribute)
  return parent_entropy - info_a_value

def select_k_best(dataset, k):
  num_attributes = dataset.shape[1] - 1
  information_gains = [information_gain(dataset, i) for i in range(num_attributes)]
  best_attribute_indices = np.argsort(information_gains)[-k:]
  return best_attribute_indices.tolist()

data = np.array([
  [1, 0, "Yes"],
  [2, 1, "No"],
  [3, 0, "Yes"],
  [4, 1, "No"],
  [5, 0, "Yes"],
])

dataset_entropy = entropy(data)
print("Entropy of the dataset:", dataset_entropy)

info_a_values = [info_a(data.copy(), i) for i in range(data.shape[1] - 1)]
print("InfoA values for each attribute:", info_a_values)

information_gains = [information_gain(data.copy(), i) for i in range(data.shape[1] - 1)]
print("Information gain for each attribute:", information_gains)

top_k_attributes = select_k_best(data.copy(), 2)
print("Indices of top 2 attributes:", top_k_attributes)
